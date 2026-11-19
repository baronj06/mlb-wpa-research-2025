"""Create a reproducible season snapshot, cleaned data and missingness reports."""

from collections import Counter
import fcntl
import hashlib
import json
import numpy as np

import pandas as pd

from .collection import CachedClient, atomic_json, read_cached_body, utc_now
from .final_validation import opportunity_state
from .phase3_quality import evaluate_quality
from .sample import ROOT
from .season import GAME_DIR, TABLE_DIR, official_player_stats


def exclusion_class(reason):
    reason=str(reason or "")
    if "429" in reason or "cooldown" in reason:return "reference_rate_limited"
    if "reference_deferred" in reason:return "reference_not_collected"
    if "403" in reason:return "reference_access_blocked"
    if "simultaneous" in reason:return "simultaneous_batting_running"
    if "multi_runner" in reason:return "joint_running_credit_unresolved"
    if "boundary" in reason:return "pa_boundary_unverified"
    if "box_accounting" in reason:return "reference_accounting_unresolved"
    if "missing_reference" in reason:return "reference_event_missing"
    if "lineup" in reason:return "player_identity_mapping_unresolved"
    if "runner_start_identity" in reason:return "runner_identity_unverified"
    if "automatic_runner" in reason:return "automatic_runner_unverified"
    if "missing_terminal_event_index" in reason:return "terminal_event_index_missing"
    return "other_validation_failure"


def validate_primary_table(data):
    """Reject unsafe promotion, while allowing explicitly excluded null rows."""
    if data.duplicated(['game_pk','at_bat_index']).any():
        raise ValueError('Duplicate season PA key')
    if data.batter_id.isna().any():
        raise ValueError('Official PA actor is ambiguous')
    valid=data.loc[data.validated]
    if not (np.isfinite(valid.batter_attributed_wpa.astype(float)).all()
            and valid.batter_attributed_wpa.between(-1,1).all()
            and np.isfinite(valid.pre_pa_li.astype(float)).all()
            and valid.pre_pa_li.ge(0).all()):
        raise ValueError('Invalid primary WPA/LI value')
    if data.loc[~data.validated,['batter_attributed_wpa','pre_pa_li']].notna().any().any():
        raise ValueError('Excluded PA contains primary values')


def main():
    manifest=json.loads((ROOT/'config/season_2025_games.json').read_text())
    client=CachedClient(offline=True)
    all_rows=[];stats=[];game_rows=[];lineage={};unresolved_actions=[];official_identity_errors=[];official_games=reference_games=0
    for game in manifest['games']:
        pk=game['game_pk'];path=GAME_DIR/f'{pk}.json'
        if not path.exists():
            game_rows.append({'game_pk':pk,'date':game['date'],'status':'not_collected','official_pa':None,'validated_pa':0})
            continue
        with (GAME_DIR/f'{pk}.lock').open('a') as lock:
            fcntl.flock(lock,fcntl.LOCK_SH)
            record=json.loads(path.read_text());sources=record.get('sources',{})
            for case in record.get('quality',{}).get('unresolved_cases',[]):
                unresolved_actions.append({'game_pk':pk,**case})
            lineage[pk]={'sources':sources,'validation_fingerprint':record.get('validation_fingerprint'),
                         'error':record.get('error'),'validation_status':record.get('validation_status')}
            if 'metadata' not in sources:
                game_rows.append({'game_pk':pk,'date':game['date'],'status':'metadata_missing','official_pa':None,'validated_pa':0})
                continue
            meta=json.loads(read_cached_body(client.root,sources['metadata']))
            if meta['gameData']['game']['pk']!=pk or meta['gameData']['game']['season']!='2025':
                raise ValueError('Season metadata identity mismatch')
            players=official_player_stats(meta);stats.extend(players)
            official_pa=sum(p['PA'] for p in players)
            complete_official={'metadata','wp'}<=sources.keys()
            complete_reference={'fangraphs_box','fangraphs_log'}<=sources.keys()
            reference_games+=int(complete_reference)
            wp=json.loads(read_cached_body(client.root,sources['wp'])) if 'wp' in sources else []
            wp_by_index={p['about']['atBatIndex']:p for p in wp} if isinstance(wp,list) else {}
            table=TABLE_DIR/f'{pk}_pas.parquet'
            prior=pd.read_parquet(table).to_dict('records') if table.exists() else []
            valid_by_index={int(r['at_bat_index']):r for r in prior}
            names={p['player_id']:p['player_name'] for p in players}
            plays=meta['liveData']['plays']['allPlays'];game_valid=represented=0
            identity_ok=isinstance(wp,list) and len(wp)==len(plays) and all(
                a['about']['atBatIndex']==b['about']['atBatIndex']
                and a['matchup']['batter']['id']==b['matchup']['batter']['id']
                and a['result']['eventType']==b['result']['eventType']
                for a,b in zip(plays,wp))
            if not identity_ok:official_identity_errors.append(pk)
            complete_official=complete_official and identity_ok
            official_games+=int(complete_official)
            for pos,p in enumerate(plays):
                credits={c['player']['id'] for c in wp_by_index.get(p['about']['atBatIndex'],{}).get('credits',[]) if c.get('credit')=='b_pa'}
                if not credits:continue
                index=p['about']['atBatIndex'];actor=next(iter(credits)) if len(credits)==1 else None
                old=valid_by_index.get(index,{})
                passed=old.get('validated') is True
                if passed and (old.get('batter_id')!=actor or pd.isna(old.get('batter_attributed_wpa')) or pd.isna(old.get('pre_pa_li'))):
                    raise ValueError('Promoted input has invalid actor/WPA/LI')
                reason='' if passed else str(old.get('reason') or record.get('error') or 'validation_not_completed')
                row={'game_pk':pk,'date':game['date'],'season':2025,'at_bat_index':index,'pa_id':f'{pk}:{index}',
                     'batter_id':actor,'batter_name':names.get(actor),'batter_at_pitch_id':p['matchup']['batter']['id'],
                     'inning':p['about']['inning'],'half_inning':p['about']['halfInning'],
                     'batting_team_id':meta['gameData']['teams']['away' if p['about']['isTopInning'] else 'home']['id'],
                     'result':p['result']['eventType'],'home_score_after':p['result']['homeScore'],
                     'away_score_after':p['result']['awayScore'],'outs_after':p['count']['outs'],
                     'validated':passed,'excluded':not passed,'exclusion_reason':reason,
                     'exclusion_class':'' if passed else exclusion_class(reason),
                     'batter_attributed_wpa':old.get('batter_attributed_wpa') if passed else None,
                     'pre_pa_li':old.get('pre_pa_li') if passed else None,
                     'li_at_batter_event':old.get('li_at_batter_event') if passed else None,
                     'independent_event_count':old.get('independent_event_count') if passed else None,
                     'batter_reference_sequence':old.get('batter_reference_sequence') if passed else None,
                     'pre_pa_li_reference_sequence':old.get('pre_pa_li_reference_sequence') if passed else None,
                     'reference_complete':complete_reference,'primary_reference':'FanGraphs',
                     'state_derivation_error':None}
                try:
                    state,method=opportunity_state(plays,pos)
                    row.update({'outs_before':state.outs,'bases_before':state.mask,'runner_on_first':state.bases[0],
                                'runner_on_second':state.bases[1],'runner_on_third':state.bases[2],
                                'home_score_before':state.home_score,'away_score_before':state.away_score,'state_method':method})
                except Exception as exc:
                    row['state_derivation_error']=str(exc)
                w=wp_by_index.get(index,{})
                delta=w.get('homeTeamWinProbabilityAdded');after=w.get('homeTeamWinProbability')
                row['api_home_wp_after']=None if after is None else after/100
                row['api_supplied_offense_wpa_unverified']=None if delta is None else (-1 if p['about']['isTopInning'] else 1)*delta/100
                row['api_li_timing_unverified']=w.get('leverageIndex')
                all_rows.append(row);represented+=1;game_valid+=int(passed)
            game_rows.append({'game_pk':pk,'date':game['date'],'official_complete':complete_official,
                              'reference_complete':complete_reference,'official_pa':official_pa,'represented_pa':represented,
                              'validated_pa':game_valid,'excluded_pa':official_pa-game_valid,
                              'missing_rate':(official_pa-game_valid)/official_pa if official_pa else None,
                              'validation_status':record.get('validation_status'),'source_error':record.get('error')})
        if len(game_rows)%250==0:print('Snapshot games',len(game_rows),flush=True)
    data=pd.DataFrame(all_rows)
    for col in ['game_pk','season','at_bat_index','batter_id','batter_at_pitch_id','inning','batting_team_id',
                'outs_before','outs_after','runner_on_first','runner_on_second','runner_on_third',
                'home_score_before','away_score_before','home_score_after','away_score_after']:
        if col in data:data[col]=data[col].astype('Int64')
    validate_primary_table(data)
    stats=pd.DataFrame(stats);fields=['PA','AB','H','HR','BB','HBP','SF']
    derived=stats.groupby('player_id')[fields].sum()
    derived['player_name']=stats.groupby('player_id')['player_name'].last()
    derived['teams']=stats.groupby('player_id')['team'].agg(lambda x:'; '.join(sorted(set(x))))
    derived['validated_pa']=data.loc[data.validated].groupby('batter_id').size().reindex(derived.index,fill_value=0)
    derived['excluded_pa']=derived.PA-derived.validated_pa
    derived['missing_rate']=derived.excluded_pa/derived.PA.replace(0,float('nan'))
    actor_counts=data.groupby('batter_id').size()
    derived['represented_pa']=actor_counts.reindex(derived.index,fill_value=0)
    actor_mismatches=int(derived.represented_pa.ne(derived.PA).sum())
    actor_mismatches+=len(actor_counts.loc[~actor_counts.index.isin(derived.index)])
    totals_receipt=json.loads((ROOT/'reports/season_2025_official_totals_receipt.json').read_text())
    totals=json.loads(read_cached_body(client.root,totals_receipt))['stats'][0]['splits']
    aliases={'PA':'plateAppearances','AB':'atBats','H':'hits','HR':'homeRuns','BB':'baseOnBalls','HBP':'hitByPitch','SF':'sacFlies'}
    official=pd.DataFrame([{'player_id':s['player']['id'],**{k:s['stat'][v] for k,v in aliases.items()}} for s in totals]).set_index('player_id')
    comparison=official.join(derived[fields],lsuffix='_season_api',rsuffix='_game_sum')
    mismatch=pd.Series(False,index=comparison.index)
    for field in fields:
        comparison[field+'_difference']=comparison[field+'_game_sum']-comparison[field+'_season_api']
        mismatch|=comparison[field+'_difference'].isna() | comparison[field+'_difference'].ne(0)
    extra_nonzero=derived.loc[(~derived.index.isin(official.index)) & (derived.PA>0)]
    mismatches=int(mismatch.sum())+len(extra_nonzero)
    official_pa=int(official.PA.sum());valid=int(data.validated.sum())
    gate=evaluate_quality(official_games,reference_games,len(manifest['games']),official_pa,valid,len(data),mismatches,actor_mismatches)
    bygame=pd.DataFrame(game_rows)
    byevent=data.groupby('result').agg(official_observed_pa=('excluded','size'),excluded_pa=('excluded','sum'))
    byevent['missing_rate']=byevent.excluded_pa/byevent.official_observed_pa
    excluded=data.loc[data.excluded]
    byreason=excluded.groupby(['exclusion_class','result']).size().rename('excluded_pa')
    path=ROOT/'data/processed'
    data.to_parquet(path/'season_2025_cleaned_pas.parquet',index=False)
    data.to_csv(path/'season_2025_cleaned_pas.csv',index=False)
    data.loc[data.validated].to_parquet(path/'season_2025_validated_pas.parquet',index=False)
    excluded.to_csv(path/'season_2025_excluded_events.csv',index=False)
    derived.to_csv(path/'season_2025_official_player_stats.csv')
    bygame.to_csv(ROOT/'reports/season_2025_quality_by_game.csv',index=False)
    derived.to_csv(ROOT/'reports/season_2025_quality_by_player.csv')
    byevent.to_csv(ROOT/'reports/season_2025_quality_by_event.csv')
    byreason.to_csv(ROOT/'reports/season_2025_exclusion_reasons.csv')
    comparison.to_csv(ROOT/'reports/season_2025_official_totals_comparison.csv')
    pd.DataFrame(unresolved_actions).to_csv(ROOT/'reports/season_2025_unresolved_actions.csv',index=False)
    ref_pa=int(data.reference_complete.sum());ref_valid=int((data.reference_complete & data.validated).sum())
    verified=data.loc[data.validated]
    api_wpa_pairs=verified.dropna(subset=['api_supplied_offense_wpa_unverified','batter_attributed_wpa'])
    api_li_pairs=verified.dropna(subset=['api_li_timing_unverified','pre_pa_li'])
    wpa_delta=api_wpa_pairs.api_supplied_offense_wpa_unverified-api_wpa_pairs.batter_attributed_wpa
    li_delta=api_li_pairs.api_li_timing_unverified-api_li_pairs.pre_pa_li
    timing_pairs=verified.dropna(subset=['pre_pa_li','li_at_batter_event'])
    diagnostics={'warning':'Different providers/models/timing: differences are not established attribution errors or interchangeable values.',
                 'api_wpa_reference_pairs':len(wpa_delta),'api_wpa_difference_gt_0_0001':int(wpa_delta.abs().gt(.0001).sum()),
                 'api_wpa_max_absolute_difference':float(wpa_delta.abs().max()) if len(wpa_delta) else None,
                 'api_li_reference_pairs':len(li_delta),'api_li_difference_gt_0_0001':int(li_delta.abs().gt(.0001).sum()),
                 'api_li_max_absolute_difference':float(li_delta.abs().max()) if len(li_delta) else None,
                 'verified_pas_with_independent_actions':int(verified.independent_event_count.fillna(0).gt(0).sum()),
                 'pre_pa_li_differs_from_batter_event_li':int((timing_pairs.pre_pa_li-timing_pairs.li_at_batter_event).abs().gt(.0001).sum())}
    report={'generated_at_utc':utc_now(),'expected_games':len(manifest['games']),
            'official_source_complete_games':official_games,'reference_source_complete_games':reference_games,
            'official_season_pa':official_pa,'represented_pa':len(data),'validated_pa':valid,
            'validation_percentage':100*valid/official_pa,'primary_excluded_pa':official_pa-valid,
            'reference_unavailable_observed_pa':int((~data.reference_complete).sum()),
            'reference_subset_pa':ref_pa,'reference_subset_validated_pa':ref_valid,
            'reference_subset_validation_percentage':100*ref_valid/ref_pa if ref_pa else None,
            'official_api_players':len(official),'official_400_pa_players':int((official.PA>=400).sum()),
            'official_player_stat_mismatches':mismatches,'quality_gate':gate,
            'player_pa_identity_count_mismatches':actor_mismatches,
            'official_feed_wp_identity_error_games':official_identity_errors,
            'state_derivation_error_pas':int(data.state_derivation_error.notna().sum()),
            'cross_provider_diagnostics':diagnostics,
            'rankings_generated':False,'metrics_generated':False,
            'exclusion_classes':excluded.exclusion_class.value_counts().to_dict(),
            'interpretation':'Reference availability losses are not measured attribution errors. Missing primary inputs are null; API fields are diagnostics only.'}
    atomic_json(ROOT/'reports/phase3_data_quality.json',report)
    files=['data/processed/season_2025_cleaned_pas.parquet','data/processed/season_2025_validated_pas.parquet']
    atomic_json(ROOT/'reports/phase3_provenance.json',{
        'generated_at_utc':utc_now(),'game_receipts':lineage,'official_totals_receipt':totals_receipt,
        'research_formula_sha256':hashlib.sha256((ROOT/'config/research_v1.json').read_bytes()).hexdigest(),
        'collection_manifest_sha256':hashlib.sha256((ROOT/'config/season_2025_games.json').read_bytes()).hexdigest(),
        'code_sha256':{str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest() for p in (ROOT/'src/mlb_wpa').glob('*.py')},
        'outputs_sha256':{f:hashlib.sha256((ROOT/f).read_bytes()).hexdigest() for f in files}})
    print(json.dumps(report,indent=2))


if __name__=='__main__':main()
