"""Conclude one model expansion; produce supported season descriptives offline.

No EOWS calculation is run when the predeclared model acceptance gate fails.
Official batting counts, cached reference WPA and unverified API diagnostics
are separate columns and have separate denominators.
"""

import hashlib
import json
import os

import numpy as np
import pandas as pd

from .collection import atomic_json, utc_now
from .sample import ROOT

OUT=ROOT/'reports/expanded_we_once'
TABLES=ROOT/'outputs/tables/preliminary_2025'
CHARTS=ROOT/'outputs/charts/preliminary_2025'


def assess_model(result,policy,confusion):
    errors=result['reference_cross_model_agreement'];reference=result['reference_validated_pa']
    tp=float(confusion.loc['High','High']);predicted_high=float(confusion['High'].sum());actual_high=float(confusion.loc['High'].sum())
    values={'li_mae':errors['li']['mae'],'li_category_agreement':errors['li']['category_agreement'],
        'high_li_precision':tp/predicted_high if predicted_high else 0.,
        'high_li_recall':tp/actual_high if actual_high else 0.,'wpa_mae':errors['wpa']['mae'],
        'we_ece':result['heldout_we']['ece_10bin_pa_weighted'],
        'absolute_inferred_we_bias':abs(errors['we_inferred_reference']['bias']),
        'model_wpa_reference_coverage':errors['wpa']['n']/reference,
        'model_li_reference_coverage':errors['li']['n']/reference,
        'added_runner_probability_violation':result['structural_diagnostics']['added_baserunner']['max_probability_violation']}
    checks={}
    for key,limit in policy.items():
        if key.endswith('_max'):checks[key]=values[key[:-4]]<=limit
        if key.endswith('_min'):checks[key]=values[key[:-4]]>=limit
    checks['brier_better_than_training_prior']=result['heldout_we']['brier_game_balanced']<result['heldout_we']['prior_baseline_brier_game_balanced']
    return {'generated_at_utc':utc_now(),'metrics_and_rankings_allowed':all(checks.values()),
            'values':values,'checks':checks,'failed_checks':[k for k,v in checks.items() if not v],
            'policy':'Exploratory operational criteria frozen before expansion results, not universal accuracy standards.',
            'next_action':'No further model iterations. Produce descriptive fallback if any check fails.'}


def batting_rates(stats):
    result=stats.copy()
    denominator=result.AB+result.BB+result.HBP+result.SF
    result['OBP']=(result.H+result.BB+result.HBP)/denominator.replace(0,np.nan)
    result['HR_per_PA']=result.HR/result.PA.replace(0,np.nan)
    return result


def attach_wpa_sources(stats,season,reference):
    output=stats.set_index('player_id').copy()
    api=season.groupby('batter_id').agg(mlb_api_wpa_observed_pa=('api_supplied_offense_wpa_unverified','count'))
    api['mlb_api_supplied_wpa_unverified_attribution']=season.groupby('batter_id').api_supplied_offense_wpa_unverified.sum(min_count=1)
    verified=reference.loc[reference.validated]
    sample=verified.groupby('batter_id').agg(reference_sample_verified_pa=('batter_attributed_wpa','size'),
        reference_sample_batter_wpa_sum=('batter_attributed_wpa','sum'))
    output=output.join(api).join(sample)
    for c in ('mlb_api_wpa_observed_pa','reference_sample_verified_pa'):output[c]=output[c].fillna(0).astype(int)
    output['reference_sample_fraction_of_season_pa']=output.reference_sample_verified_pa/output.PA.replace(0,np.nan)
    output['full_season_reference_batter_wpa']=output.reference_sample_batter_wpa_sum.where(
        output.PA.gt(0)&output.reference_sample_verified_pa.eq(output.PA))
    output['full_season_wpa_status']=np.where(output.full_season_reference_batter_wpa.notna(),
        'complete_observed_reference','unavailable_complete_batter_attributed_reference')
    return output.reset_index()


def make_charts(qualified,reference,gate):
    os.environ.setdefault('MPLCONFIGDIR',str(ROOT/'.cache/matplotlib'))
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    plt.rcParams.update({'axes.spines.top':False,'axes.spines.right':False,'font.size':10})
    fig,axes=plt.subplots(1,3,figsize=(15,4.8),layout='constrained')
    axes[0].hist(qualified.HR,bins=np.arange(0,66,5),color='#176a82',edgecolor='white')
    axes[0].set(xlabel='Home runs',ylabel='Players',title='2025 HR distribution (PA >= 400)')
    axes[1].hist(qualified.OBP,bins=14,color='#d98f34',edgecolor='white')
    axes[1].set(xlabel='On-base percentage',ylabel='Players',title='2025 OBP distribution (PA >= 400)')
    axes[2].scatter(qualified.HR_per_PA,qualified.OBP,s=25,color='#176a82',alpha=.7)
    selected=pd.concat([qualified.nlargest(3,'HR'),qualified.nlargest(3,'OBP')]).drop_duplicates('player_id')
    for p in selected.itertuples():axes[2].annotate(p.player_name,(p.HR_per_PA,p.OBP),xytext=(3,4),textcoords='offset points',fontsize=7)
    axes[2].set(xlabel='HR / PA',ylabel='OBP',title='Power and on-base output can overlap')
    fig.suptitle('2025 MLB batting descriptives — official counts, no EOWS rankings')
    fig.supxlabel('Source: previously cached MLB game boxes; player IDs joined across team stints. OBP includes home runs.',fontsize=9)
    fig.savefig(CHARTS/'hr_obp_descriptives.png',dpi=220);plt.close(fig)
    fig,axes=plt.subplots(1,2,figsize=(11,4.6),layout='constrained')
    verified=reference.loc[reference.validated,'batter_attributed_wpa']
    axes[0].hist(verified,bins=60,color='#176a82')
    axes[0].set(xlabel='Verified batter WPA per PA (wins)',ylabel='Reference-sample PAs',
                title=f'Observed FanGraphs sample only (n={len(verified):,})')
    axes[1].hist(qualified.mlb_api_supplied_wpa_unverified_attribution,bins=20,color='#9c9c9c')
    axes[1].set(xlabel='Sum of API supplied offensive-event WPA (wins)',ylabel='Players (PA >= 400)',
                title='Full-season API diagnostic: attribution unverified')
    fig.suptitle('WPA source populations remain separate')
    fig.supxlabel('Left: 65 selected games, not representative season coverage. Right: not validated batter WPA; do not use as a leaderboard.',fontsize=8)
    fig.savefig(CHARTS/'wpa_source_descriptives.png',dpi=220);plt.close(fig)


def run():
    TABLES.mkdir(parents=True,exist_ok=True);CHARTS.mkdir(parents=True,exist_ok=True)
    result=json.loads((OUT/'results.json').read_text());policy=json.loads((OUT/'acceptance_policy.json').read_text())
    confusion=pd.read_csv(OUT/'li_category_confusion.csv',index_col=0)
    gate=assess_model(result,policy,confusion);atomic_json(OUT/'acceptance_result.json',gate)
    stats=pd.read_csv(ROOT/'data/processed/season_2025_official_player_stats.csv')
    season=pd.read_parquet(ROOT/'data/processed/season_2025_cleaned_pas.parquet')
    reference=pd.read_parquet(ROOT/'reports/reference_review_v04/reference_pas.parquet')
    stats=batting_rates(stats);stats=attach_wpa_sources(stats,season,reference)
    primary=stats.loc[stats.PA.ge(400)].copy()
    stats.sort_values('player_id').to_csv(TABLES/'all_player_batting_and_wpa_sources.csv',index=False)
    primary.sort_values(['HR','player_id'],ascending=[False,True]).to_csv(TABLES/'players_pa400_hr_obp_wpa_sources.csv',index=False)
    primary.sort_values(['HR','player_id'],ascending=[False,True]).head(10).to_csv(TABLES/'top10_hr.csv',index=False)
    primary.sort_values(['OBP','player_id'],ascending=[False,True]).head(10).to_csv(TABLES/'top10_obp_pa400.csv',index=False)
    stats[['PA','HR','HR_per_PA','OBP']].describe().to_csv(TABLES/'all_players_distribution_summary.csv')
    primary[['PA','HR','HR_per_PA','OBP']].describe().to_csv(TABLES/'pa400_distribution_summary.csv')
    verified=reference.loc[reference.validated]
    verified.groupby('result').agg(verified_pa=('batter_attributed_wpa','size'),
        mean_batter_wpa=('batter_attributed_wpa','mean'),median_batter_wpa=('batter_attributed_wpa','median'),
        sum_batter_wpa=('batter_attributed_wpa','sum')).to_csv(TABLES/'reference_sample_wpa_by_event.csv')
    totals=stats[['PA','AB','H','HR','BB','HBP','SF']].sum()
    league_obp=float((totals.H+totals.BB+totals.HBP)/(totals.AB+totals.BB+totals.HBP+totals.SF))
    summary={'generated_at_utc':utc_now(),'season':2025,'official_games':int(season.game_pk.nunique()),
        'official_pa':int(totals.PA),'official_hr':int(totals.HR),'league_obp':league_obp,
        'players_with_pa':int(stats.PA.gt(0).sum()),'pa400_players':len(primary),
        'pa400_mean_hr':float(primary.HR.mean()),'pa400_median_hr':float(primary.HR.median()),
        'pa400_mean_obp':float(primary.OBP.mean()),'pa400_median_obp':float(primary.OBP.median()),
        'hr_obp_spearman_pa400':float(primary.HR_per_PA.corr(primary.OBP,method='spearman')),
        'reference_verified_pa':len(verified),'reference_sample_games':int(reference.game_pk.nunique()),
        'reference_sample_wpa_mean':float(verified.batter_attributed_wpa.mean()),
        'complete_reference_wpa_players_pa400':int(primary.full_season_reference_batter_wpa.notna().sum()),
        'api_diagnostic_wpa_observed_pa':int(season.api_supplied_offense_wpa_unverified.notna().sum()),
        'eows_rankings_generated':False,'hitter_group_winning_comparisons_generated':False,
        'model_acceptance':gate,'interpretation':'HR/OBP are official-count descriptives. Reference WPA is partial-sample only; full-season API WPA is an unverified diagnostic. No reliable EOWS or winning-contribution group conclusion is claimed.'}
    atomic_json(OUT/'descriptive_results.json',summary)
    make_charts(primary,reference,gate)
    original=json.loads((ROOT/'reports/independent_we_poc/results.json').read_text())
    rows=[]
    for label,path in [('WPA MAE',('reference_cross_model_agreement','wpa','mae')),
                       ('LI MAE',('reference_cross_model_agreement','li','mae')),
                       ('LI category agreement',('reference_cross_model_agreement','li','category_agreement')),
                       ('Game-balanced Brier',('heldout_we','brier_game_balanced'))]:
        a=original;b=result
        for k in path:a=a[k];b=b[k]
        rows.append({'measure':label,'original':a,'expanded_once':b})
    pd.DataFrame(rows).to_csv(OUT/'model_comparison.csv',index=False)
    atomic_json(OUT/'descriptive_provenance.json',{'generated_at_utc':utc_now(),
        'inputs':{name:hashlib.sha256((ROOT/name).read_bytes()).hexdigest() for name in [
            'data/processed/season_2025_official_player_stats.csv','data/processed/season_2025_cleaned_pas.parquet',
            'reports/reference_review_v04/reference_pas.parquet','config/research_v1.json']},
        'outputs':{str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest() for folder in (TABLES,CHARTS) for p in folder.iterdir() if p.is_file()},
        'code_sha256':hashlib.sha256((ROOT/'src/mlb_wpa/preliminary_descriptives.py').read_bytes()).hexdigest()})
    print(json.dumps(summary,indent=2))
    return summary


if __name__=='__main__':run()
