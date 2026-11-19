"""Explain unresolved source cases and audit MLB-only feasibility, offline."""

import json
import hashlib

import pandas as pd

from .attribution import state_matches
from .collection import CachedClient, atomic_json, read_cached_body, utc_now
from .final_validation import parse_query
from .reference_review import (OUT, crosswalk_v04, opportunity_v04, ledger_v04,
                                label_v04)
from .sample import ROOT


def prestate_shift_counterexample():
    """Synthetic software audit, never a recorded MLB game or fitted model."""
    pre=[.5,.6,.4];final=1.0
    expected=[b-a for a,b in zip(pre,pre[1:]+[final])]
    backward=[b-a for a,b in zip([.5]+pre,pre+[final])]
    return {'fixture':'SYNTHETIC WE sequence, not MLB observations',
            'pre_pa_home_we':pre,'final_home_outcome':final,
            'correct_pa_changes':expected,
            'backward_diff_assigned_to_current_indices':backward[:-1],
            'synthetic_terminal_row_delta':backward[-1],
            'both_game_sums_equal':abs(sum(expected)-sum(backward))<1e-12,
            'conclusion':'A telescoping game total does not validate per-PA attribution or index alignment.'}


def main():
    data=pd.read_parquet(OUT/'reference_pas.parquet')
    baseline=pd.read_parquet(ROOT/'data/processed/season_2025_cleaned_pas.parquet')
    client=CachedClient(offline=True);contexts={};cases=[];identities=[]
    for pk,group in data.groupby('game_pk'):
        rec=json.loads((ROOT/f'data/checkpoints/season_2025/games/{pk}.json').read_text())
        plays=json.loads(read_cached_body(client.root,rec['sources']['wp']))
        meta=json.loads(read_cached_body(client.root,rec['sources']['metadata']))
        box=parse_query(read_cached_body(client.root,rec['sources']['fangraphs_box']),'scores/box-score')
        events=parse_query(read_cached_body(client.root,rec['sources']['fangraphs_log']),'scores/play-log')
        crosswalk,_=crosswalk_v04(meta,box)
        byindex=group.set_index('at_bat_index')
        for pos,p in enumerate(plays):
            index=p['about']['atBatIndex']
            if index not in byindex.index:continue
            r=byindex.loc[index]
            try:
                state,_=opportunity_v04(plays,pos)
                units,blockers=ledger_v04([p],{index:state})
            except ValueError:
                units=[];blockers=['unverified_state']
            if not r.old_validated:
                evidence={'game_pk':pk,'at_bat_index':index,'original_reason':r.old_exclusion_reason,
                          'recovered':bool(r.recovered),'remaining_reason':r.reason or '',
                          'mlb_result':p['result']['description'],
                          'observed_action_codes':';'.join(sorted({str(e.get('details',{}).get('eventType')) for e in p['playEvents'] if not e.get('isPitch')}))}
                if r.recovered:
                    evidence['root_cause']='implementation_conservatism_or_label_coverage'
                    evidence['evidence']='Recovered by exact cached actor/state/type/sequence matching and provider box accounting; source numeric values unchanged.'
                elif r.reason=='missing_reference_event' and units:
                    unit=next(u for u in units if u.kind=='batter')
                    candidates=[e for e in events if crosswalk.get(int(e['playerId']))==unit.actor_id
                                and int(e['inning'])==unit.before.inning
                                and e['ih']==('0' if unit.before.half=='top' else '1')
                                and int(e['outs'])==unit.before.outs and e['score']==f'{unit.before.home_score}-{unit.before.away_score}'
                                and label_v04(unit,e)]
                    if len(candidates)==1 and candidates[0]['base']!=unit.before.mask:
                        e=candidates[0];evidence.update(root_cause='cross_source_base_state_conflict',
                            evidence=f"MLB base={unit.before.mask}; FanGraphs base={e['base']}; original sequence={e['gameSeq']}. No base-state repair or numeric substitution.")
                    elif p['result']['eventType']=='walk' and any(
                        'Pitcher Pitch Timer Violation' in e.get('details',{}).get('description','') and e.get('count',{}).get('balls')==4
                        for e in p['playEvents']):
                        evidence.update(root_cause='reference_omission_automatic_timer_walk',
                            evidence='Recorded fourth automatic ball from pitcher timer violation; no matching reference batter event. Box residual is not a replacement value.')
                    else:
                        evidence.update(root_cause='reference_identity_or_state_unmatched',evidence='No unique exact reference match.')
                else:
                    evidence.update(root_cause=r.reason or 'unclassified',evidence='Original action ledger remains unsupported or source accounting/state cannot be verified; primary values null.')
                cases.append(evidence)
            if not r.validated or blockers:continue
            independent=[u for u in units if u.kind=='runner']
            # First game WP and automatic-runner boundary adjustments have no
            # independently observed preceding opportunity WP in this audit.
            if not pos or any(e.get('details',{}).get('eventType')=='runner_placed' for e in p['playEvents']):continue
            action_by_index={e['index']:e for e in p['playEvents']}
            deltas=[action_by_index[u.action_index].get('homeTeamWinProbabilityAdded') for u in independent]
            before=plays[pos-1].get('homeTeamWinProbability');after=p.get('homeTeamWinProbability')
            supplied=p.get('homeTeamWinProbabilityAdded')
            complete=all(x is not None for x in deltas+[before,after,supplied])
            residual=None if not complete else (after-before-sum(deltas)-supplied)/100
            identities.append({'game_pk':pk,'at_bat_index':index,'independent_actions':len(independent),
                               'action_wp_complete':complete,'home_wpa_accounting_residual':residual,
                               'role':'same-provider accounting only; not independent WE/LI model validation'})
    cases=pd.DataFrame(cases)
    assert len(cases)==153 and int(cases.recovered.sum())==118
    cases.to_csv(OUT/'case_evidence_153.csv',index=False)
    checks=pd.DataFrame(identities);checks.to_csv(OUT/'mlb_wpa_accounting.csv',index=False)
    pairs=checks.loc[checks.action_wp_complete]
    # Season metadata availability, never promoted to primary WPA/LI.
    missing_li=baseline.api_li_timing_unverified.isna()
    first=baseline.groupby('game_pk').at_bat_index.transform('min').eq(baseline.at_bat_index)
    states=baseline.dropna(subset=['outs_before','bases_before','home_score_before','away_score_before']).copy()
    states['score_diff']=states.home_score_before-states.away_score_before
    keys=['inning','half_inning','outs_before','bases_before','score_diff']
    sizes=states.groupby(keys).size()
    bygame=baseline.groupby('game_pk').size().rename('official_pa').to_frame()
    bygame['reference_pa']=data.groupby('game_pk').size().reindex(bygame.index,fill_value=0)
    bygame['validated_pa']=data.loc[data.validated].groupby('game_pk').size().reindex(bygame.index,fill_value=0)
    byplayer=baseline.groupby('batter_id').size().rename('official_pa').to_frame()
    byplayer['reference_pa']=data.groupby('batter_id').size().reindex(byplayer.index,fill_value=0)
    byplayer['validated_pa']=data.loc[data.validated].groupby('batter_id').size().reindex(byplayer.index,fill_value=0)
    byevent=baseline.groupby('result').size().rename('official_pa').to_frame()
    byevent['reference_pa']=data.groupby('result').size().reindex(byevent.index,fill_value=0)
    byevent['validated_pa']=data.loc[data.validated].groupby('result').size().reindex(byevent.index,fill_value=0)
    for label,table in [('game',bygame),('player',byplayer),('event',byevent)]:
        table['excluded_pa']=table.official_pa-table.validated_pa
        table['missing_rate']=table.excluded_pa/table.official_pa
        table.to_csv(OUT/f'quality_by_{label}.csv')
    report={'generated_at_utc':utc_now(),'network_requests':client.network_requests,
        'original_cases':153,'recovered':118,'remaining_root_causes':cases.loc[~cases.recovered].root_cause.value_counts().to_dict(),
        'mlb_accounting_tested_pa':len(pairs),'mlb_accounting_with_independent_action_pa':int(pairs.independent_actions.gt(0).sum()),
        'mlb_accounting_unavailable_pa':int((~checks.action_wp_complete).sum()),
        'mlb_accounting_discrepancy_gt_0_000001':int(pairs.home_wpa_accounting_residual.abs().gt(1e-6).sum()),
        'mlb_accounting_max_absolute_residual':float(pairs.home_wpa_accounting_residual.abs().max()),
        'season_api_li_missing_pa':int(missing_li.sum()),'first_game_pa_li_missing':int((missing_li&first).sum()),
        'observed_usable_pre_state_pa':len(states),'distinct_pre_state_buckets':len(sizes),
        'state_buckets_n_lt_50':int(sizes.lt(50).sum()),
        'pa_in_state_buckets_n_lt_50':int(sizes.loc[sizes.lt(50)].sum()),
        'independently_validated_calculation_available':False,
        'interpretation':'Same-provider arithmetic cannot establish independent numeric validity. Counterfactual WE, transition probabilities, LI normalization and independently tested timing are not supplied by this accounting check.',
        'synthetic_source_algorithm_check':prestate_shift_counterexample()}
    atomic_json(OUT/'evidence_and_model_feasibility.json',report)
    provenance=json.loads((OUT/'provenance.json').read_text())
    provenance['code_sha256']={str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest()
                              for p in (ROOT/'src/mlb_wpa').glob('*.py')}
    provenance['outputs_sha256'].update({str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest()
        for p in OUT.glob('*') if p.suffix in ('.csv','.parquet') or p.name=='evidence_and_model_feasibility.json'})
    provenance['document_review_receipts']='reports/reference_review_v04/source_documents/receipts.json'
    atomic_json(OUT/'provenance.json',provenance)
    print(json.dumps(report,indent=2))
    return report


if __name__=='__main__':main()
