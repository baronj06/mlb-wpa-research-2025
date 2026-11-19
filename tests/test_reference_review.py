"""Real cached evidence plus explicitly synthetic mutations; no HTTP."""

from copy import deepcopy
import hashlib
import json

import pandas as pd
import pytest

from mlb_wpa.attribution import build_ledger, intentional_walk_terminal
from mlb_wpa.collection import CachedClient, read_cached_body
from mlb_wpa.final_validation import parse_query, opportunity_state
from mlb_wpa.reference_review import (audit, inactive_lineup_entries, crosswalk_v04,
    ledger_v04, opportunity_v04, label_v04)
from mlb_wpa.sample import ROOT
from mlb_wpa.verify_sources import build_crosswalk


def real_game(pk):
    record=json.loads((ROOT/f'data/checkpoints/season_2025/games/{pk}.json').read_text())
    client=CachedClient(offline=True)
    metadata=json.loads(read_cached_body(client.root,record['sources']['metadata']))
    plays=json.loads(read_cached_body(client.root,record['sources']['wp']))
    box=parse_query(read_cached_body(client.root,record['sources']['fangraphs_box']),'scores/box-score')
    return metadata,plays,box


@pytest.fixture(scope='module')
def reviewed():
    return audit()


def test_real_reference_denominators_and_all_original_values_preserved(reviewed):
    data,report=reviewed
    assert len(data)==4800 and report['reference_games']==65
    assert report['validated_pa']==4765 and report['recovered_original_pa']==118
    assert report['remaining_unresolved_pa']==35
    assert not report['lost_previously_validated_pa']
    assert not any(report['previous_primary_value_changes'].values())
    assert report['network_requests']==0 and not report['metrics_and_rankings_allowed']
    assert data.loc[~data.validated,['batter_attributed_wpa','pre_pa_li']].isna().all().all()


def test_real_zero_pa_pitchers_are_not_batter_identity_gaps():
    meta,plays,box=real_game(778515)
    omitted=inactive_lineup_entries(meta,box)
    assert len(omitted)==4 and all(p['official_pa']==0 for p in omitted)
    with pytest.raises(ValueError,match='Unmatched MLB'):build_crosswalk(meta,box)
    mapping,_=crosswalk_v04(meta,box)
    assert not {p['mlbam_id'] for p in omitted}&set(mapping.values())
    # Synthetic adverse mutation: an actual PA prevents ignoring the entry.
    altered=deepcopy(meta);p=omitted[0]
    altered['liveData']['boxscore']['teams'][p['side']]['players'][p['roster_key']]['stats']['batting']['plateAppearances']=1
    with pytest.raises(ValueError,match='Unmatched MLB'):crosswalk_v04(altered,box)


def test_real_intentional_walk_terminal_has_four_observed_balls():
    _,plays,_=real_game(778516);p=next(p for p in plays if p['about']['atBatIndex']==59)
    assert intentional_walk_terminal(p)==3
    state,_=opportunity_v04(plays,plays.index(p))
    old,blockers=build_ledger([p],{59:state})
    assert not old and blockers[0]['reason']=='missing_terminal_event_index'
    units,blockers=ledger_v04([p],{59:state})
    assert not blockers and len(units)==1 and units[0].action_index==3
    altered=deepcopy(p);altered['playEvents'][-1]['count']['balls']=3
    assert intentional_walk_terminal(altered) is None


def test_real_automatic_runner_before_no_pitch_walk_is_observed():
    _,plays,_=real_game(778539);pos=next(i for i,p in enumerate(plays) if p['about']['atBatIndex']==76)
    with pytest.raises(ValueError,match='automatic_runner'):opportunity_state(plays,pos)
    state,method=opportunity_v04(plays,pos)
    assert state.mask=='_2_' and state.bases[1]==668804 and 'intentional' in method
    altered=deepcopy(plays);placement=next(e for e in altered[pos]['playEvents'] if e.get('details',{}).get('eventType')=='runner_placed')
    placement['player']['id']=-1  # Synthetic wrong actor must fail.
    with pytest.raises(ValueError,match='identity'):opportunity_v04(altered,pos)


def test_real_multiple_runner_wild_pitch_keeps_whole_action_separate(reviewed):
    data,_=reviewed
    r=data.loc[(data.game_pk==778555)&(data.at_bat_index==52)].iloc[0]
    assert r.recovered and r.independent_event_count==1
    events=pd.read_parquet(ROOT/'reports/reference_review_v04/reference_events.parquet')
    events=events.loc[(events.game_pk==778555)&(events.at_bat_index==52)]
    assert list(events.kind)==['runner','batter']
    assert r.batter_attributed_wpa==events.iloc[-1].reference_wpa
    assert r.pre_pa_li==events.iloc[0].reference_li
    assert r.batter_reference_sequence!=r.pre_pa_li_reference_sequence
    _,plays,_=real_game(778555);pos=next(i for i,p in enumerate(plays) if p['about']['atBatIndex']==52)
    state,_=opportunity_v04(plays,pos);p=deepcopy(plays[pos])
    action=next(e for e in p['playEvents'] if e.get('details',{}).get('eventType')=='wild_pitch')
    action.pop('player')  # Synthetic absent owner is never guessed.
    _,blockers=ledger_v04([p],{52:state})
    assert any(b['reason']=='multi_runner_or_missing_credit_requires_review' for b in blockers)


def test_real_pinch_runner_replacement_preserves_state_identity(reviewed):
    data,_=reviewed
    assert data.loc[(data.game_pk==778551)&(data.at_bat_index==60),'recovered'].item()
    _,plays,_=real_game(778551);pos=next(i for i,p in enumerate(plays) if p['about']['atBatIndex']==60)
    state,_=opportunity_v04(plays,pos);p=deepcopy(plays[pos]);p['playEvents'][0]['replacedPlayer']['id']=-1
    _,blockers=ledger_v04([p],{60:state})
    assert any(b['reason']=='pinch_runner_replacement_unverified' for b in blockers)


@pytest.mark.parametrize('pk,index',[(778520,24),(778538,14),(778548,63)])
def test_real_reference_abbreviations_do_not_create_source_gaps(reviewed,pk,index):
    data,_=reviewed;r=data.loc[(data.game_pk==pk)&(data.at_bat_index==index)].iloc[0]
    assert r.recovered and r.old_exclusion_reason=='missing_reference_event'


def test_true_source_gaps_and_base_conflicts_remain_null(reviewed):
    data,_=reviewed
    for pk,index in [(778519,57),(778549,36),(778559,28)]:
        r=data.loc[(data.game_pk==pk)&(data.at_bat_index==index)].iloc[0]
        assert not r.validated and pd.isna(r.pre_pa_li) and pd.isna(r.batter_attributed_wpa)


def test_provider_policy_blocks_live_fangraphs_before_http(tmp_path):
    class DeniedSession:
        def get(self,*args,**kwargs):raise AssertionError('No real or synthetic HTTP allowed')
    client=CachedClient(tmp_path,session=DeniedSession(),sleep=lambda _:None)
    with pytest.raises(PermissionError,match='provider policy'):
        client.get('https://www.fangraphs.com/api/synthetic')
    assert client.network_requests==0


def test_synthetic_prestate_diff_detects_attribution_shift_despite_game_sum():
    from mlb_wpa.review_evidence import prestate_shift_counterexample
    r=prestate_shift_counterexample()
    assert r['both_game_sums_equal']
    assert r['correct_pa_changes']!=r['backward_diff_assigned_to_current_indices']
    assert r['backward_diff_assigned_to_current_indices'][0]==0
    assert r['correct_pa_changes'][0]==pytest.approx(.1)


def test_real_root_cause_audit_and_mlb_accounting_limitations(reviewed):
    from mlb_wpa.review_evidence import main
    r=main()
    assert r['original_cases']==153 and r['recovered']==118 and r['network_requests']==0
    assert r['remaining_root_causes']['cross_source_base_state_conflict']==2
    assert r['remaining_root_causes']['reference_omission_automatic_timer_walk']==1
    assert r['mlb_accounting_discrepancy_gt_0_000001']==1
    assert r['mlb_accounting_max_absolute_residual']==pytest.approx(.002)
    assert r['season_api_li_missing_pa']==r['first_game_pa_li_missing']==2430
    assert not r['independently_validated_calculation_available']


def test_all_original_datasets_checkpoints_and_formula_are_unchanged():
    manifest=json.loads((ROOT/'reports/reference_review_v04/protected_inputs.json').read_text())
    assert len(manifest)>24000
    for name,expected in manifest.items():
        assert hashlib.sha256((ROOT/name).read_bytes()).hexdigest()==expected,name


def test_provider_policy_still_allows_existing_verified_cache():
    record=json.loads((ROOT/'data/checkpoints/season_2025/games/778563.json').read_text())
    receipt=record['sources']['fangraphs_log'];client=CachedClient()
    raw,cached=client.get(receipt['requested_url'],receipt.get('params'))
    assert hashlib.sha256(raw).hexdigest()==receipt['sha256']==cached['sha256']
    assert client.network_requests==0
