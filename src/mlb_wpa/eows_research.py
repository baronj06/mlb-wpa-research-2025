"""First exploratory 2025 EOWS evaluation with uncertainty and sensitivity."""

import hashlib
import json

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

from .collection import atomic_json, utc_now
from .eows_apply import OUT, DATA
from .eows_metrics import (classify_hitters, scoring_table, summarize_events,
                           original_components, composite)
from .eows_bootstrap import bootstrap
from .sample import ROOT

TABLES=ROOT/'outputs/tables/eows_2025_exploratory'


def coverage_tables(data,stats):
    q=data.copy();li=q.diagnostic_start_li.to_numpy(float)
    q['category']=np.where(~np.isfinite(li),'Unknown',np.where(li<.85,'Low',np.where(li<2,'Medium','High')))
    byplayer=q.groupby('batter_id').agg(official_observed_pa=('batter_id','size'),eligible_PA=('metric_eligible','sum'))
    byplayer['excluded_PA']=byplayer.official_observed_pa-byplayer.eligible_PA
    byplayer['coverage_fraction']=byplayer.eligible_PA/byplayer.official_observed_pa
    byplayer=stats.set_index('player_id')[['player_name','PA']].join(byplayer)
    byplayer.to_csv(OUT/'coverage_by_player.csv')
    bycategory=q.groupby(['batter_id','category']).agg(opportunities_with_this_state=('batter_id','size'),
        eligible_pa=('metric_eligible','sum'))
    bycategory['excluded_pa']=bycategory.opportunities_with_this_state-bycategory.eligible_pa
    bycategory['retention_fraction']=bycategory.eligible_pa/bycategory.opportunities_with_this_state
    bycategory.to_csv(OUT/'coverage_by_player_category.csv')
    q.groupby('category').agg(opportunities=('batter_id','size'),eligible_pa=('metric_eligible','sum')).to_csv(OUT/'coverage_by_category.csv')
    return byplayer


def scenario_comparison(events,primary,all_data,minimum):
    ids=primary.player_id.to_numpy();complete_ids=all_data.groupby('game_pk').metric_eligible.all()
    complete=events.loc[events.game_pk.isin(complete_ids.loc[complete_ids].index)]
    scenarios=[('contextual_LI',events,events.alternative_pre_pa_li.to_numpy(),2.,(.4,.3,.3)),
        ('high_cut_1p8',events,None,1.8,(.4,.3,.3)),('high_cut_2p2',events,None,2.2,(.4,.3,.3)),
        ('LI_minus_0p10',events,np.maximum(0,events.estimated_pre_pa_li.to_numpy()-.1),2.,(.4,.3,.3)),
        ('LI_plus_0p10',events,events.estimated_pre_pa_li.to_numpy()+.1,2.,(.4,.3,.3)),
        ('complete_games_only',complete,None,2.,(.4,.3,.3)),
        ('weights_50_25_25',events,None,2.,(.5,.25,.25)),
        ('weights_34_33_33',events,None,2.,(.34,.33,.33))]
    rows=[];summaries=[];point=primary.set_index('player_id')
    for name,ev,li,high,weights in scenarios:
        table,bench=scoring_table(ev,ids,li,high,weights)
        finite=table.EOWS_C.notna();sub=table.loc[finite].copy();common=sub.player_id
        base=point.loc[common].copy()
        if len(sub)!=len(primary):
            # Recompute percentiles on the identical finite comparison set.
            bs,_=scoring_table(events,common.to_numpy());base=bs.set_index('player_id').loc[common]
        sub['primary_rank_common']=base.Rank.to_numpy()
        sub['rank_change']=sub.Rank-sub.primary_rank_common
        sub['below_primary_category_floor']=sub[['Low_PA','Medium_PA','High_PA']].min(axis=1)<minimum
        sub['scenario']=name;rows.append(sub)
        rho=spearmanr(sub.Rank,sub.primary_rank_common).statistic if len(sub)>2 else np.nan
        top=set(sub.loc[sub.Rank<=10,'player_id']);base_top=set(base.loc[base.Rank<=10].index)
        summaries.append({'scenario':name,'players_compared':len(sub),'players_unscored_empty_category':int((~finite).sum()),
            'players_below_primary_category_floor':int(sub.below_primary_category_floor.sum()),
            'rank_spearman':float(rho),'mean_absolute_rank_shift':float(sub.rank_change.abs().mean()),
            'max_absolute_rank_shift':float(sub.rank_change.abs().max()),'top10_overlap':len(top&base_top),
            'eligible_league_pa':bench['eligible_league_pa'],'p_low':bench['category_proportions'][0],
            'p_medium':bench['category_proportions'][1],'p_high':bench['category_proportions'][2],
            'high_league_mean_wpa':bench['high_league_mean_wpa']})
    return pd.concat(rows,ignore_index=True),pd.DataFrame(summaries),int(complete_ids.sum())


def run():
    TABLES.mkdir(parents=True,exist_ok=True)
    design=json.loads((OUT/'design.json').read_text());data=pd.read_parquet(DATA)
    stats=pd.read_csv(ROOT/'data/processed/season_2025_official_player_stats.csv')
    classified,classification=classify_hitters(stats)
    coverage=coverage_tables(data,stats)
    events=data.loc[data.metric_eligible].copy()
    all_scores,league=scoring_table(events,classified.player_id.to_numpy())
    qualified=classified.merge(all_scores,on='player_id',validate='one_to_one')
    qualified['valid_pa_fraction']=qualified.eligible_PA/qualified.PA
    qualified['category_minimum']=qualified[['Low_PA','Medium_PA','High_PA']].min(axis=1)
    sufficient=(qualified.valid_pa_fraction>=design['minimum_total_valid_pa_fraction']) & (qualified.category_minimum>=design['minimum_valid_pa_each_category'])
    excluded=qualified.loc[~sufficient].copy()
    excluded['reason']=np.where(excluded.valid_pa_fraction<design['minimum_total_valid_pa_fraction'],'coverage_below_95pct','category_below_20_pa')
    excluded.to_csv(TABLES/'excluded_pa400_players.csv',index=False)
    ids=qualified.loc[sufficient,'player_id'].to_numpy()
    primary,league=scoring_table(events,ids)
    primary=classified.merge(primary,on='player_id',validate='one_to_one')
    primary['valid_pa_fraction']=primary.eligible_PA/primary.PA
    primary['Estimated_WPA_per_100_eligible_PA']=100*primary.Estimated_WPA/primary.eligible_PA
    primary['WPA_source']='Independent frozen Markov estimate on eligible observed PAs; not FanGraphs'
    primary['Ranking_status']='Exploratory; uncertainty and sensitivity attached'
    primary=primary.sort_values('player_id').reset_index(drop=True)
    long,sensitivity,n_complete=scenario_comparison(events,primary,data,design['minimum_valid_pa_each_category'])
    long.to_csv(TABLES/'sensitivity_rankings_long.csv',index=False);sensitivity.to_csv(TABLES/'sensitivity_summary.csv',index=False)
    maximum=long.groupby('player_id').rank_change.agg(lambda x:x.abs().max()).rename('max_sensitivity_rank_shift')
    primary=primary.merge(maximum,on='player_id',validate='one_to_one')
    for label,mask in [('LI',~long.scenario.str.startswith('weights_')&long.scenario.ne('complete_games_only')),
                       ('weight',long.scenario.str.startswith('weights_')),('complete_game',long.scenario.eq('complete_games_only'))]:
        change=long.loc[mask].groupby('player_id').rank_change.agg(lambda x:x.abs().max()).rename(f'max_{label}_rank_shift')
        primary=primary.merge(change,on='player_id',validate='one_to_one')
    print('Point estimates and sensitivity complete:',len(primary),'players',flush=True)
    intervals,group_intervals,boot=bootstrap(data,primary,design['bootstrap_reps'],design['bootstrap_seed'])
    primary=primary.merge(intervals,on='player_id',validate='one_to_one')
    primary['bootstrap_EOWS_C_bias']=primary.bootstrap_mean_EOWS_C-primary.EOWS_C
    primary['bootstrap_rank_interval_width']=primary.Rank_ci_high-primary.Rank_ci_low
    primary['wide_rank_uncertainty']=primary.bootstrap_rank_interval_width>design['rank_flags']['bootstrap_rank_interval_width_over']
    primary['sensitivity_flag']=primary.max_sensitivity_rank_shift>design['rank_flags']['max_scenario_rank_shift_over']
    primary['stability_note']=np.select([primary.wide_rank_uncertainty&primary.sensitivity_flag,primary.wide_rank_uncertainty,
        primary.sensitivity_flag],['Wide bootstrap interval and sensitive scenario ranks','Wide bootstrap rank interval',
        'Sensitive to scenario choices'],default='No large shift under the displayed diagnostic rules; not proof of stability')
    primary=primary.sort_values(['Rank','player_id'])
    display=['Rank','player_id','player_name','teams','PA','eligible_PA','valid_pa_fraction','HR','HR_per_PA','OBP',
        'Estimated_WPA','Estimated_WPA_per_100_eligible_PA','EOWS_100','DI','CR','E','D','C','EOWS_C','Group',
        'Low_PA','Medium_PA','High_PA','Rank_ci_low','Rank_ci_high','bootstrap_top10_probability',
        'bootstrap_mean_EOWS_C','bootstrap_EOWS_C_bias','bootstrap_median_rank',
        'max_sensitivity_rank_shift','max_LI_rank_shift','max_weight_rank_shift','max_complete_game_rank_shift',
        'wide_rank_uncertainty','sensitivity_flag','stability_note',
        'EOWS_100_ci_low','EOWS_100_ci_high','DI_ci_low','DI_ci_high','CR_ci_low','CR_ci_high',
        'EOWS_C_ci_low','EOWS_C_ci_high','WPA_source','Ranking_status']
    primary[display].to_csv(TABLES/'preliminary_eows_c_rankings.csv',index=False)
    for metric in ['EOWS_100','DI','CR','Estimated_WPA']:
        primary.sort_values([metric,'player_id'],ascending=[False,True]).to_csv(TABLES/f'ranking_{metric}.csv',index=False)
    classified.to_csv(TABLES/'all_pa400_player_classification.csv',index=False)
    floors=[]
    for minimum in [10,30]:
        keep=(qualified.valid_pa_fraction>=.95)&qualified.category_minimum.ge(minimum)
        table,_=scoring_table(events,qualified.loc[keep,'player_id'].to_numpy())
        table['minimum_category_PA']=minimum;table['comparison_population_n']=len(table);floors.append(table)
    pd.concat(floors).to_csv(TABLES/'category_minimum_sensitivity.csv',index=False)
    group_rows=[]
    for group,q in primary.groupby('Group'):
        for metric in ['Estimated_WPA','EOWS_100','DI','CR','EOWS_C','HR_per_PA','OBP']:
            group_rows.append({'Group':group,'metric':metric,'n_players':len(q),'mean':q[metric].mean(),
                'median':q[metric].median(),'sd':q[metric].std()})
    groups=pd.DataFrame(group_rows).merge(group_intervals.rename(columns={'comparison':'Group'}),on=['Group','metric','n_players'],how='left')
    groups['bootstrap_bias']=groups.bootstrap_mean-groups['mean']
    groups['resampling_range_excludes_point']=groups['mean'].lt(groups.ci_low)|groups['mean'].gt(groups.ci_high)
    groups.to_csv(TABLES/'group_descriptive_comparison.csv',index=False)
    differences=[]
    A=primary.loc[primary.Group.eq('Power')];B=primary.loc[primary.Group.eq('On-Base')]
    for metric in ['EOWS_100','DI','CR','EOWS_C']:
        ci=group_intervals.loc[group_intervals.comparison.eq('Power minus On-Base')&group_intervals.metric.eq(metric)].iloc[0]
        differences.append({'metric':metric,'Power_n':len(A),'OnBase_n':len(B),
            'mean_difference_Power_minus_OnBase':float(A[metric].mean()-B[metric].mean()),
            'ci_low':ci.ci_low,'ci_high':ci.ci_high,'bootstrap_mean_difference':ci.bootstrap_mean})
    pd.DataFrame(differences).to_csv(TABLES/'power_minus_onbase_bootstrap.csv',index=False)
    primary[['EOWS_100','DI','CR','EOWS_C','Estimated_WPA','HR_per_PA','OBP']].corr(method='spearman').to_csv(TABLES/'component_spearman_correlations.csv')
    primary[['player_id','Rank','Rank_ci_low','Rank_ci_high','bootstrap_top10_probability']].to_csv(TABLES/'bootstrap_rank_uncertainty.csv',index=False)
    classified.merge(qualified[['player_id','eligible_PA','valid_pa_fraction','Low_PA','Medium_PA','High_PA']],on='player_id').to_csv(TABLES/'pa400_category_coverage.csv',index=False)
    report={'generated_at_utc':utc_now(),'framework':'Unmodified EOWS v1.0 / EOWS-C v1.0',
        'model':'Independent frozen Markov v0.3; estimated, not official FanGraphs',
        'official_pa':len(data),'eligible_league_pa':len(events),'excluded_pa':len(data)-len(events),
        'official_pa400_players':len(classified),'primary_ranked_players':len(primary),'excluded_pa400_players':len(excluded),
        'classification':classification,'league':league,'bootstrap':boot,'complete_games_for_missingness_stress':n_complete,
        'ranked_group_counts':primary.Group.value_counts().to_dict(),
        'wide_bootstrap_rank_intervals':int(primary.wide_rank_uncertainty.sum()),
        'large_sensitivity_rank_shifts':int(primary.sensitivity_flag.sum()),
        'top10':primary.head(10)[['Rank','player_id','player_name','EOWS_100','DI','CR','EOWS_C','Group',
            'Rank_ci_low','Rank_ci_high','bootstrap_top10_probability']].to_dict('records'),
        'power_minus_onbase':differences,'ranking_release':'Exploratory estimates only; no claim of stable exact order, causality or superiority.',
        'model_refitted':False}
    atomic_json(OUT/'research_results.json',report);atomic_json(OUT/'bootstrap_design_and_completion.json',boot)
    atomic_json(OUT/'scoring_provenance.json',{'generated_at_utc':utc_now(),
        'input_dataset_sha256':hashlib.sha256(DATA.read_bytes()).hexdigest(),
        'formula_sha256':hashlib.sha256((ROOT/'config/research_v1.json').read_bytes()).hexdigest(),
        'design_sha256':hashlib.sha256((OUT/'design.json').read_bytes()).hexdigest(),
        'code_sha256':{str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest() for p in (ROOT/'src/mlb_wpa').glob('eows*.py')},
        'outputs_sha256':{str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest() for p in TABLES.iterdir() if p.is_file()}})
    print(json.dumps(report,indent=2),flush=True)
    return primary,report


if __name__=='__main__':run()
