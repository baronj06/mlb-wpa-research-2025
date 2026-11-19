"""Offline v0.4 reference review. Preserve all season files and checkpoints."""

from collections import Counter
from copy import deepcopy
from dataclasses import replace
import hashlib
import json
import re

import pandas as pd

from .attribution import build_ledger, intentional_walk_terminal, start_state, RUNNER_TYPES
from .collection import CachedClient, atomic_json, read_cached_body, utc_now
from .final_validation import validate_one, opportunity_state, parse_query, match_units_by_state, event_label_matches
from .sample import ROOT
from .verify_sources import build_crosswalk

OUT=ROOT/'reports/reference_review_v04'
VERSION='attribution-validation v0.4; no metric-formula revision'


def inactive_lineup_entries(metadata,box):
    """Ignore only unmatched, zero-PA pitchers absent from offensive events."""
    plays=metadata['liveData']['plays']['allPlays']
    actors={p['matchup']['batter']['id'] for p in plays}
    actors.update(r['details']['runner']['id'] for p in plays for r in p.get('runners',[]))
    actors.update(e['player']['id'] for p in plays for e in p['playEvents']
                  if e.get('details',{}).get('eventType') in RUNNER_TYPES and e.get('player'))
    omitted=[]
    for side in ('home','away'):
        fg={(p['BatOrder'],p['SubNum']) for p in box[side]['bat'] if p['RowType']==0}
        for key,p in metadata['liveData']['boxscore']['teams'][side]['players'].items():
            order=p.get('battingOrder')
            if not order:continue
            slot=(int(order)//100,int(order)%100)
            # Require an explicitly recorded zero; missing statistics are not zero.
            if (slot not in fg and p.get('stats',{}).get('batting',{}).get('plateAppearances')==0
                and p.get('position',{}).get('code')=='1' and p['person']['id'] not in actors):
                omitted.append({'side':side,'roster_key':key,'mlbam_id':p['person']['id'],
                                'lineup_slot':slot,'official_pa':0,
                                'evidence':'unmatched zero-PA pitcher; no batter/runner appearance in official plays'})
    return omitted


def crosswalk_v04(metadata,box):
    working=deepcopy(metadata)
    for p in inactive_lineup_entries(metadata,box):
        del working['liveData']['boxscore']['teams'][p['side']]['players'][p['roster_key']]
    return build_crosswalk(working,box)


def opportunity_v04(plays,position):
    p=plays[position]
    placements=[e for e in p['playEvents'] if e.get('details',{}).get('eventType')=='runner_placed']
    terminal=intentional_walk_terminal(p)
    if not placements or terminal is None:return opportunity_state(plays,position)
    state=start_state(plays,position)
    starts=[e['index'] for e in p['playEvents'] if e.get('isPitch') or e.get('details',{}).get('code')=='VB']
    if (len(placements)!=1 or state.inning<10 or state.outs!=0 or any(state.bases)
        or not starts or placements[0]['index']>=min(starts)):
        raise ValueError('unverified_automatic_runner_placement')
    actor=placements[0].get('player',{}).get('id')
    if actor is None or not (p['matchup'].get('postOnSecond',{}).get('id')==actor or any(
        r['details']['runner']['id']==actor and r['movement'].get('originBase')=='2B' for r in p.get('runners',[]))):
        raise ValueError('automatic_runner_identity_unverified')
    return replace(state,bases=(None,actor,None)),'explicit_automatic_runner_before_intentional_balls'


def ledger_v04(plays,initial_states=None):
    return build_ledger(plays,initial_states,reviewed_rules=True)


def label_v04(unit,event):
    if event_label_matches(unit,event):return True
    text=event['playDesc'].lower()
    return bool((unit.event_type=='grounded_into_double_play' and re.search(r'\b(?:gidp|gdp)\b',text))
                or (unit.event_type=='field_out' and 'batter interference' in text))


def match_v04(units,events,crosswalk):
    return match_units_by_state(units,events,crosswalk,label_matcher=label_v04)


def review_one(game,record,client):
    return validate_one(game,record,client,crosswalk_builder=crosswalk_v04,
                        state_builder=opportunity_v04,ledger_builder=ledger_v04,reference_matcher=match_v04)


def audit():
    OUT.mkdir(exist_ok=True)
    baseline=pd.read_parquet(ROOT/'data/processed/season_2025_cleaned_pas.parquet')
    selected=baseline.loc[baseline.reference_complete]
    assert len(selected)==4800 and int(selected.validated.sum())==4647
    games=json.loads((ROOT/'config/season_2025_games.json').read_text())['games']
    client=CachedClient(offline=True)
    rows=[];events=[];reports=[];inputs={};omissions=[]
    for game in games:
        pk=game['game_pk']
        if pk not in set(selected.game_pk):continue
        record=json.loads((ROOT/f'data/checkpoints/season_2025/games/{pk}.json').read_text())
        inputs[pk]=record['sources']
        meta=json.loads(read_cached_body(client.root,record['sources']['metadata']))
        box=parse_query(read_cached_body(client.root,record['sources']['fangraphs_box']),'scores/box-score')
        omissions.extend({'game_pk':pk,**r} for r in inactive_lineup_entries(meta,box))
        result=review_one(game,record,client)
        rows.extend(result[0]);events.extend(result[1]);reports.append(result[2])
    data=pd.DataFrame(rows)
    old=selected[['game_pk','at_bat_index','batter_id','validated','exclusion_reason','batter_attributed_wpa','pre_pa_li']].rename(
        columns={c:'old_'+c for c in ['batter_id','validated','exclusion_reason','batter_attributed_wpa','pre_pa_li']})
    data=data.merge(old,on=['game_pk','at_bat_index'],validate='one_to_one',how='outer',indicator=True)
    if not data._merge.eq('both').all() or len(data)!=4800:raise ValueError('Reference denominator changed')
    if not data.batter_id.eq(data.old_batter_id).all():raise ValueError('Official PA ownership changed')
    if ((~data.validated)&data[['batter_attributed_wpa','pre_pa_li']].notna().any(axis=1)).any():raise ValueError('Excluded primary inputs are nonnull')
    unchanged=data.old_validated & data.validated
    value_changes={c:int((data.loc[unchanged,c]-data.loc[unchanged,'old_'+c]).abs().gt(1e-12).sum())
                   for c in ['batter_attributed_wpa','pre_pa_li']}
    lost=int((data.old_validated & ~data.validated).sum())
    if lost or any(value_changes.values()):raise ValueError('Previously accepted values changed; review before publication')
    data['recovered']=~data.old_validated & data.validated
    data.to_parquet(OUT/'reference_pas.parquet',index=False)
    data.to_csv(OUT/'reference_pas.csv',index=False)
    data.loc[~data.old_validated].to_csv(OUT/'original_153_disposition.csv',index=False)
    data.loc[~data.validated].to_csv(OUT/'remaining_unresolved.csv',index=False)
    pd.DataFrame(events).to_parquet(OUT/'reference_events.parquet',index=False)
    pd.DataFrame(omissions).to_csv(OUT/'ignored_nonbatting_lineup_entries.csv',index=False)
    count=int(data.validated.sum());n=len(baseline)
    report={'generated_at_utc':utc_now(),'method':VERSION,'network_requests':client.network_requests,
            'official_season_games':int(baseline.game_pk.nunique()),'official_season_pa':n,
            'reference_games':len(reports),'reference_pa':len(data),'old_validated_pa':4647,
            'validated_pa':count,'reference_validation_percentage':100*count/len(data),
            'recovered_original_pa':int(data.recovered.sum()),'remaining_unresolved_pa':len(data)-count,
            'lost_previously_validated_pa':lost,'previous_primary_value_changes':value_changes,
            'season_validated_percentage':100*count/n,'reference_unavailable_pa':n-len(data),
            'season_excluded_pa':n-count,'metrics_and_rankings_allowed':False,
            'stop_reasons':['reference_collection_incomplete','independent_numeric_model_validation_not_established'],
            'original_153_causes':data.loc[~data.old_validated].old_exclusion_reason.value_counts().to_dict(),
            'recovered_by_original_cause':data.loc[data.recovered].old_exclusion_reason.value_counts().to_dict(),
            'remaining_causes':data.loc[~data.validated].reason.value_counts().to_dict(),
            'games':reports}
    atomic_json(OUT/'validation.json',report)
    outputs={str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest()
             for p in OUT.glob('*') if p.suffix in ('.csv','.parquet')}
    atomic_json(OUT/'provenance.json',{'generated_at_utc':utc_now(),'method':VERSION,
        'inputs':inputs,'baseline_sha256':hashlib.sha256((ROOT/'data/processed/season_2025_cleaned_pas.parquet').read_bytes()).hexdigest(),
        'formula_sha256':hashlib.sha256((ROOT/'config/research_v1.json').read_bytes()).hexdigest(),
        'code_sha256':{str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest() for p in (ROOT/'src/mlb_wpa').glob('*.py')},
        'outputs_sha256':outputs,'primary_values':'Observed cached FanGraphs only; no imputation or model substitution'})
    print(json.dumps({k:v for k,v in report.items() if k!='games'},indent=2))
    return data,report


if __name__=='__main__':audit()
