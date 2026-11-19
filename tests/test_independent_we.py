"""Independent WE/LI mechanics and real POC holdout regression tests."""

from copy import deepcopy
from dataclasses import replace
import hashlib
import json

import numpy as np
import pandas as pd
import pytest

from mlb_wpa.attribution import State
from mlb_wpa.collection import CachedClient, read_cached_body
from mlb_wpa.independent_we import (RidgeWE, ExpectedOutcomeLI, normalize_state,
    attributed_wpa, feature_vector, extract_game)
from mlb_wpa.sample import ROOT
from mlb_wpa.we_poc import validate_split, calibration_table

FOLDER=ROOT/'reports/independent_we_poc'


def state(inning=1,half='top',outs=0,bases=(None,None,None),home=0,away=0):
    """Explicitly synthetic state for software tests."""
    return State(inning,half,outs,bases,home,away)


@pytest.mark.parametrize('s,winner',[
    (state(9,'bottom',0,home=1),1.),
    (state(9,'top',3,home=1),1.),
    (state(9,'bottom',3,away=1),0.),
    (state(12,'bottom',1,home=2,away=1),1.)])
def test_terminal_outcomes_follow_game_state_not_future_winner(s,winner):
    active,value=normalize_state(s)
    assert active is None and value==winner


def test_third_out_resets_half_and_automatic_runner_has_no_fake_pa():
    s,t=normalize_state(state(8,'top',3,bases=(1,2,3)))
    assert t is None and (s.inning,s.half,s.outs,s.mask)==(8,'bottom',0,'___')
    s,t=normalize_state(state(9,'bottom',3))
    assert t is None and (s.inning,s.half,s.outs,s.mask)==(10,'top',0,'_2_')
    s,t=normalize_state(state(10,'top',3))
    assert t is None and s.half=='bottom' and s.mask=='_2_'


def test_extra_innings_are_not_the_regulation_ninth_feature():
    assert not np.array_equal(feature_vector(state(9)),feature_vector(state(10)))
    # Stationary automatic-runner extra-inning regime is explicitly pooled.
    assert np.array_equal(feature_vector(state(10)),feature_vector(state(11)))


class ToyWE:
    """Synthetic probability map, never a fitted baseball estimate."""
    training_games={1,2}
    def predict(self,states):
        result=[]
        for s in states:
            active,terminal=normalize_state(s)
            if terminal is not None:result.append(terminal)
            else:result.append(.5+.05*(active.home_score-active.away_score))
        return np.array(result)


def test_away_and_home_event_wpa_use_the_correct_sign_and_current_event():
    model=ToyWE();away=state();home=state(half='bottom')
    assert attributed_wpa(model,away,replace(away,away_score=1))==pytest.approx(.05)
    assert attributed_wpa(model,home,replace(home,home_score=1))==pytest.approx(.05)
    assert attributed_wpa(model,home,replace(home,home_score=2))==pytest.approx(.10)
    with pytest.raises(ValueError):attributed_wpa(model,away,home)


def toy_kernel_rows():
    s=state()
    return [dict(game_pk=1,supported=True,start_state=s,batter_before=s,
                 after_state=replace(s,outs=1),outcome=(0,1,'___')),
            dict(game_pk=2,supported=True,start_state=s,batter_before=s,
                 after_state=replace(s,away_score=1),outcome=(1,0,'___'))]


def test_li_integrates_all_training_outcomes_not_the_realized_test_result():
    li=ExpectedOutcomeLI(ToyWE(),min_pa=1,min_games=1).fit(toy_kernel_rows())
    expected,info=li.expected_change(state())
    assert expected==pytest.approx(.025)  # .5 * zero change + .5 * .05 change
    assert info['outcome_support']==2 and info['kernel_games']==2
    assert li.predict(state())[0]==pytest.approx(1.)
    # Query accepts only the pre-state: changing a held-out result cannot enter it.
    assert li.predict(state())[0]==li.predict(deepcopy(state()))[0]


def test_unsupported_transition_states_are_null_and_never_neutral_imputations():
    li=ExpectedOutcomeLI(ToyWE(),min_pa=1,min_games=1).fit(toy_kernel_rows())
    value,info=li.predict(state(outs=2))
    assert value is None and info['kernel_pa']==0


def test_kernel_and_calibration_reject_overlapping_or_heldout_game_ids():
    rows=toy_kernel_rows();rows[0]['game_pk']=999
    with pytest.raises(ValueError,match='leakage'):ExpectedOutcomeLI(ToyWE(),1,1).fit(rows)
    model=RidgeWE();model.training_games={1}
    with pytest.raises(ValueError,match='overlap'):model.calibrate([state()],[1],[1])
    with pytest.raises(ValueError,match='leakage'):
        validate_split({'training_games':[1],'calibration_games':[2],'test_reference_games':[1],'selected_games':3})


def test_censored_walkoff_is_not_a_transition_donor():
    rows=toy_kernel_rows();last=deepcopy(rows[0]);last['start_state']=last['batter_before']=state(9,'bottom')
    last['after_state']=state(9,'bottom',home=1);rows.append(last)
    li=ExpectedOutcomeLI(ToyWE(),1,1).fit(rows)
    assert li.excluded_walkoff_donors==1 and sum(li.counts[(0,'___')].values())==2


def test_calibration_bins_are_disjoint_at_endpoints():
    q=pd.DataFrame({'game_pk':[1]*11,'independent_pre_pa_home_we':np.arange(11)/10,'home_won':[0]*11})
    table=calibration_table(q)
    assert table.pa.sum()==11 and len(table)==10 and table.iloc[-1].pa==2


def test_real_extraction_ignores_supplied_numeric_model_fields():
    rec=json.loads((ROOT/'data/checkpoints/season_2025/games/778563.json').read_text());client=CachedClient(offline=True)
    meta=json.loads(read_cached_body(client.root,rec['sources']['metadata']));plays=json.loads(read_cached_body(client.root,rec['sources']['wp']))
    first=extract_game(778563,meta,plays,'test')
    altered=deepcopy(plays)
    for p in altered:
        p['homeTeamWinProbability']=-999;p['homeTeamWinProbabilityAdded']=999;p['leverageIndex']=-999
        for e in p['playEvents']:
            e['homeTeamWinProbabilityAdded']=999;e['leverageIndex']=-999
    second=extract_game(778563,meta,altered,'test')
    assert first==second and len(first)==74 and client.network_requests==0


def test_real_uncertain_pa_boundaries_are_not_promoted_to_we_or_li_inputs():
    rec=json.loads((ROOT/'data/checkpoints/season_2025/games/778551.json').read_text());client=CachedClient(offline=True)
    meta=json.loads(read_cached_body(client.root,rec['sources']['metadata']));plays=json.loads(read_cached_body(client.root,rec['sources']['wp']))
    row=next(r for r in extract_game(778551,meta,plays,'test') if r['at_bat_index']==16)
    assert row['start_state'] is None and not row['supported']
    assert row['reason']=='runner_action_pa_boundary_unverified'


@pytest.fixture(scope='module')
def result():
    # The real POC is executed explicitly; fixtures reuse its frozen artifacts.
    return json.loads((FOLDER/'results.json').read_text())


def test_real_poc_scope_denominators_and_nonpromotion(result):
    plan=json.loads((FOLDER/'design.json').read_text());splits=validate_split(plan)
    assert [len(x) for x in splits]==[120,30,65] and sum(map(len,splits))==215
    data=pd.read_parquet(FOLDER/'independent_predictions.parquet')
    assert data.game_pk.nunique()==215
    assert data.loc[data.role=='test'].shape[0]==4800
    assert not data.duplicated(['game_pk','at_bat_index']).any()
    assert data.loc[~data.supported,'independent_batter_wpa'].isna().all()
    assert result['reference_validated_pa']==4765 and result['reference_unresolved_pa']==35
    assert result['network_requests']==0 and not result['adequately_validated_for_rankings']
    available=data.independent_batter_wpa.notna()
    assert np.allclose(data.loc[available,'independent_batter_wpa']+data.loc[available,'independent_runner_wpa'],
                       data.loc[available,'independent_whole_pa_wpa'])


def test_real_model_has_score_monotonicity_and_exact_terminal_boundaries(result):
    weights=json.loads((FOLDER/'model.json').read_text());model=RidgeWE()
    model.beta=np.array(weights['coefficient_vector']);model.calibration=np.array(weights['calibration'])
    assert model.beta[1]>=0 and model.calibration[1]>=0
    preds=model.predict([state(8,home=i) for i in range(5)])
    assert np.all(np.diff(preds)>=0)
    assert model.predict([state(9,'bottom',home=1)])[0]==1
    assert model.predict([state(9,'bottom',3,away=1)])[0]==0
    assert result['structural_diagnostics']['score']['violations_gt_1e_10']==0


def test_real_first_pa_predictions_are_independent_not_filled_official_li(result):
    q=pd.read_csv(FOLDER/'heldout_reference_comparison.csv')
    first=q.sort_values('at_bat_index').groupby('game_pk').head(1)
    assert len(first)==65 and first.independent_pre_pa_li.notna().all()
    old=pd.read_parquet(ROOT/'data/processed/season_2025_cleaned_pas.parquet')
    original=old.sort_values('at_bat_index').groupby('game_pk').head(1)
    assert original.api_li_timing_unverified.isna().all()


def test_model_design_and_formula_hashes_are_preserved(result):
    p=json.loads((FOLDER/'provenance.json').read_text())
    assert p['design_sha256']==hashlib.sha256((FOLDER/'design.json').read_bytes()).hexdigest()
    assert p['formula_sha256']==json.loads((ROOT/'reports/attribution_provenance.json').read_text())['formula_config_sha256']
    for name,expected in p['outputs_sha256'].items():
        assert hashlib.sha256((ROOT/name).read_bytes()).hexdigest()==expected,name
