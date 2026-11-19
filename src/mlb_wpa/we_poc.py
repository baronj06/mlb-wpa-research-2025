"""Run the frozen 215-game offline independent WE/LI proof of concept."""

from collections import Counter
import hashlib
import json

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

from .collection import CachedClient, atomic_json, read_cached_body, utc_now
from .independent_we import RidgeWE, ExpectedOutcomeLI, extract_game, attributed_wpa, structural_diagnostics, FEATURE_NAMES
from .sample import ROOT

OUT=ROOT/'reports/independent_we_poc'


def plot_results(test,bins):
    import os
    os.environ.setdefault('MPLCONFIGDIR',str(ROOT/'.cache/matplotlib'))
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig,axes=plt.subplots(1,3,figsize=(15,4.7),layout='constrained')
    ax=axes[0];ax.plot([0,1],[0,1],'--',color='gray',linewidth=1)
    ax.plot(bins.mean_prediction,bins.home_win_fraction,color='#166d82')
    ax.scatter(bins.mean_prediction,bins.home_win_fraction,s=20+bins.pa/15,color='#166d82')
    ax.set(xlabel='Mean predicted home win probability',ylabel='Observed home-win frequency',
           title='Held-out calibration (65 games)',xlim=(0,1),ylim=(0,1))
    for ax,(col,ref,title,unit) in zip(axes[1:],[
        ('independent_batter_wpa','fangraphs_batter_wpa','Batter WPA agreement','WPA (wins)'),
        ('independent_pre_pa_li','fangraphs_pre_pa_li','Pre-PA LI agreement','Leverage Index')]):
        q=test.loc[test.reference_validated].dropna(subset=[col,ref])
        ax.scatter(q[ref],q[col],s=7,alpha=.18,color='#166d82',rasterized=True)
        lo=min(0,float(q[[col,ref]].min().min()));hi=float(q[[col,ref]].max().max())
        ax.plot([lo,hi],[lo,hi],'--',color='gray',linewidth=1)
        ax.set(xlabel='FanGraphs '+unit,ylabel='Independent model '+unit,title=title+f' (n={len(q):,})')
    fig.suptitle('Independent WE/LI proof of concept — no player rankings',fontsize=14)
    fig.supxlabel('Cached MLB events; separate training/calibration/reference games. FanGraphs values are comparison data only.',fontsize=9)
    fig.savefig(OUT/'validation_diagnostics.png',dpi=200)
    plt.close(fig)


def validate_split(plan):
    sets=[set(plan[k]) for k in ('training_games','calibration_games','test_reference_games')]
    if any(sets[i]&sets[j] for i in range(3) for j in range(i)):
        raise ValueError('Game leakage between splits')
    if sum(map(len,sets))!=plan['selected_games']:raise ValueError('Unexpected POC scope')
    return sets


def paired_errors(pred,reference):
    pred=np.asarray(pred,float);reference=np.asarray(reference,float)
    if not len(pred):return {'n':0}
    delta=pred-reference;rho=spearmanr(pred,reference).statistic if len(pred)>2 else None
    return {'n':len(pred),'mae':float(np.abs(delta).mean()),'rmse':float(np.sqrt(np.mean(delta**2))),
            'bias':float(delta.mean()),'spearman':None if rho is None or not np.isfinite(rho) else float(rho)}


def bootstrap_mean_ci(frame,column,seed=20251117,n=500,game_balanced=False):
    grouped=frame.groupby('game_pk')[column].agg(['sum','count','mean'])
    if not len(grouped):return None
    rng=np.random.default_rng(seed);idx=rng.integers(0,len(grouped),size=(n,len(grouped)))
    if game_balanced:means=grouped['mean'].to_numpy()[idx].mean(axis=1)
    else:means=grouped['sum'].to_numpy()[idx].sum(axis=1)/grouped['count'].to_numpy()[idx].sum(axis=1)
    return [float(x) for x in np.quantile(means,[.025,.975])]


def calibration_table(frame):
    rows=[]
    categories=np.minimum(9,np.floor(frame.independent_pre_pa_home_we*10).astype(int))
    for b in range(10):
        q=frame.loc[categories==b]
        if len(q):rows.append({'lower':b/10,'pa':len(q),'games':q.game_pk.nunique(),
            'mean_prediction':float(q.independent_pre_pa_home_we.mean()),'home_win_fraction':float(q.home_won.mean())})
    return pd.DataFrame(rows)


def role_for(pk,sets):
    return ('train' if pk in sets[0] else 'calibration' if pk in sets[1] else 'test')


def run():
    plan=json.loads((OUT/'design.json').read_text());sets=validate_split(plan)
    client=CachedClient(offline=True);rows=[];receipts={};test_events={}
    for pk in sorted(set.union(*sets)):
        record=json.loads((ROOT/f'data/checkpoints/season_2025/games/{pk}.json').read_text())
        src=record['sources'];role=role_for(pk,sets)
        meta=json.loads(read_cached_body(client.root,src['metadata']))
        plays=json.loads(read_cached_body(client.root,src['wp']))
        if meta['gameData']['game']['pk']!=pk or meta['gameData']['game']['type']!='R':raise ValueError('POC game identity')
        official=meta['liveData']['plays']['allPlays']
        if len(official)!=len(plays) or any(a['about']['atBatIndex']!=b['about']['atBatIndex']
            or a['matchup']['batter']['id']!=b['matchup']['batter']['id'] for a,b in zip(official,plays)):
            raise ValueError('POC feed/WP identity')
        rows.extend(extract_game(pk,meta,plays,role))
        receipts[pk]={k:src[k] for k in ('metadata','wp')}
        if role=='test':
            # Read only for later evaluation; never supplied to fitting.
            from .final_validation import parse_query
            ev=parse_query(read_cached_body(client.root,src['fangraphs_log']),'scores/play-log')
            test_events[pk]={int(e['gameSeq']):e for e in ev}
            receipts[pk]['reference_log']=src['fangraphs_log']
    active=[r for r in rows if r['start_state'] is not None]
    train=[r for r in active if r['role']=='train'];cal=[r for r in active if r['role']=='calibration']
    model=RidgeWE(plan['ridge_penalty']).fit([r['start_state'] for r in train],
        [r['home_won'] for r in train],[r['game_pk'] for r in train])
    model.calibrate([r['start_state'] for r in cal],[r['home_won'] for r in cal],[r['game_pk'] for r in cal])
    li=ExpectedOutcomeLI(model,plan['min_kernel_pa'],plan['min_kernel_games']).fit(train)
    if model.training_games & sets[2] or model.calibration_games & sets[2] or li.normalization_games & sets[2]:
        raise ValueError('Held-out game leaked into fitted components')
    predictions=[];flat_inputs=[]
    for r in rows:
        s=r['start_state'];entry={k:r[k] for k in ['game_pk','at_bat_index','role','batter_id','home_won','result','supported','reason']}
        entry.update(independent_pre_pa_home_we=None,independent_raw_home_we=None,
            independent_batter_wpa=None,independent_runner_wpa=None,independent_whole_pa_wpa=None,
            independent_pre_pa_li=None,li_reason='start_state_unavailable',kernel_pa=None,kernel_games=None,
            kernel_expected_home_we_drift=None)
        features={**entry}
        if s is not None:
            features.update(inning=s.inning,half=s.half,outs=s.outs,bases=s.mask,
                            home_score=s.home_score,away_score=s.away_score)
            entry['independent_pre_pa_home_we']=model.predict([s])[0]
            entry['independent_raw_home_we']=model.predict([s],calibrated=False)[0]
            value,info=li.predict(s);entry.update(independent_pre_pa_li=value,
                li_reason='' if value is not None else 'insufficient_training_transition_support',
                kernel_pa=info['kernel_pa'],kernel_games=info['kernel_games'],
                kernel_expected_home_we_drift=info.get('expected_home_we_drift'))
            if r['supported']:
                sign=1 if s.half=='bottom' else -1
                before=model.predict([r['batter_before']])[0];after=model.predict([r['after_state']])[0]
                runner=sum(attributed_wpa(model,a,b) for a,b in r['runner_states'])
                batter=attributed_wpa(model,r['batter_before'],r['after_state']);whole=sign*(after-entry['independent_pre_pa_home_we'])
                if abs(batter+runner-whole)>1e-10:raise ValueError('Independent event accounting failed')
                entry.update(independent_batter_wpa=batter,independent_runner_wpa=runner,
                             independent_whole_pa_wpa=whole)
        predictions.append(entry);flat_inputs.append(features)
    state_frame=pd.DataFrame(flat_inputs)
    data=pd.DataFrame(predictions).merge(state_frame[['game_pk','at_bat_index','inning','half','outs','bases','home_score','away_score']],
                                       on=['game_pk','at_bat_index'],validate='one_to_one')
    state_frame.to_parquet(OUT/'input_states.parquet',index=False)
    data.to_parquet(OUT/'independent_predictions.parquet',index=False)
    data.to_csv(OUT/'independent_predictions.csv',index=False)
    test=data.loc[data.role=='test'].copy()
    ref=pd.read_parquet(ROOT/'reports/reference_review_v04/reference_pas.parquet')
    test=test.merge(ref[['game_pk','at_bat_index','batter_id','validated','batter_attributed_wpa','pre_pa_li','pre_pa_li_reference_sequence']],
                    on=['game_pk','at_bat_index','batter_id'],validate='one_to_one',how='left')
    if len(test)!=4800 or test.validated.isna().any():raise ValueError('Reference denominator mismatch')
    test=test.rename(columns={'batter_attributed_wpa':'fangraphs_batter_wpa','pre_pa_li':'fangraphs_pre_pa_li',
                             'validated':'reference_validated'})
    test['fangraphs_pre_pa_home_we_inferred']=np.nan
    for r in test.loc[test.reference_validated].itertuples():
        e=test_events[r.game_pk][int(r.pre_pa_li_reference_sequence)]
        # Provider's home post-WE minus signed credited change: inferred, not
        # independently observed pre-WE. Never use for training/calibration.
        sign=1 if e['ih']=='1' else -1
        test.loc[r.Index,'fangraphs_pre_pa_home_we_inferred']=float(e['we'])-sign*float(e['wpa'])
    test['fangraphs_pre_pa_home_we_inferred']=pd.to_numeric(test.fangraphs_pre_pa_home_we_inferred)
    test.to_csv(OUT/'heldout_reference_comparison.csv',index=False)
    scored=test.dropna(subset=['independent_pre_pa_home_we']).copy()
    scored['brier']=(scored.independent_pre_pa_home_we-scored.home_won)**2
    scored['raw_brier']=(scored.independent_raw_home_we-scored.home_won)**2
    scored['baseline_brier']=(model.baseline-scored.home_won)**2
    prob=np.clip(scored.independent_pre_pa_home_we,1e-12,1-1e-12)
    scored['log_loss']=-(scored.home_won*np.log(prob)+(1-scored.home_won)*np.log1p(-prob))
    bins=calibration_table(scored);bins.to_csv(OUT/'heldout_calibration.csv',index=False)
    comparison={}
    for label,a,b in [('wpa','independent_batter_wpa','fangraphs_batter_wpa'),
                      ('li','independent_pre_pa_li','fangraphs_pre_pa_li'),
                      ('we_inferred_reference','independent_pre_pa_home_we','fangraphs_pre_pa_home_we_inferred')]:
        pairs=test.loc[test.reference_validated].dropna(subset=[a,b]).copy()
        comparison[label]=paired_errors(pairs[a],pairs[b]);pairs['absolute_error']=abs(pairs[a]-pairs[b])
        comparison[label]['mae_game_cluster_bootstrap_95ci']=bootstrap_mean_ci(pairs,'absolute_error',n=plan['bootstrap_replicates'])
        comparison[label]['paired_games']=int(pairs.game_pk.nunique())
        if label=='li':
            cat=lambda x:np.where(x<.85,'Low',np.where(x<2,'Medium','High'))
            pairs['reference_category']=cat(pairs[b]);pairs['model_category']=cat(pairs[a])
            pd.crosstab(pairs.reference_category,pairs.model_category).to_csv(OUT/'li_category_confusion.csv')
            comparison[label]['category_agreement']=float(pairs.reference_category.eq(pairs.model_category).mean())
            comparison[label]['reference_high_leverage_pairs']=int(pairs[b].ge(2).sum())
    supports=[]
    for key,counts in sorted(li.counts.items()):
        supports.append({'outs':key[0],'bases':key[1],'training_pa':sum(counts.values()),
                         'training_games':len(li.games[key]),'outcome_support':len(counts),
                         'supported':sum(counts.values())>=li.min_pa and len(li.games[key])>=li.min_games and len(counts)>=2})
    pd.DataFrame(supports).to_csv(OUT/'training_transition_support.csv',index=False)
    stratified=[]
    eligible=test.loc[test.reference_validated].copy()
    eligible['reference_leverage']=np.where(eligible.fangraphs_pre_pa_li<.85,'Low',np.where(eligible.fangraphs_pre_pa_li<2,'Medium','High'))
    for label,q in eligible.groupby('reference_leverage'):
        wp=q.dropna(subset=['independent_batter_wpa']);lp=q.dropna(subset=['independent_pre_pa_li'])
        stratified.append({'reference_leverage':label,'reference_pa':len(q),
            'paired_wpa_pa':len(wp),'wpa_mae':float(abs(wp.independent_batter_wpa-wp.fangraphs_batter_wpa).mean()),
            'paired_li_pa':len(lp),'li_mae':float(abs(lp.independent_pre_pa_li-lp.fangraphs_pre_pa_li).mean())})
    pd.DataFrame(stratified).to_csv(OUT/'errors_by_reference_leverage.csv',index=False)
    exclusions=data.loc[~data.supported].groupby(['role','reason']).size().rename('pa')
    exclusions.to_csv(OUT/'attribution_exclusions.csv')
    summaries=[]
    for role,q in data.groupby('role'):
        summaries.append({'role':role,'games':int(q.game_pk.nunique()),'official_pa':len(q),
            'we_available':int(q.independent_pre_pa_home_we.notna().sum()),
            'batter_wpa_available':int(q.independent_batter_wpa.notna().sum()),
            'pre_pa_li_available':int(q.independent_pre_pa_li.notna().sum())})
    report={'generated_at_utc':utc_now(),'version':plan['version'],'selected_games':215,
        'network_requests':client.network_requests,'split_summary':summaries,
        'reference_validated_pa':int(test.reference_validated.sum()),
        'reference_unresolved_pa':int((~test.reference_validated).sum()),
        'model_attribution_available_but_reference_unresolved':int((~test.reference_validated & test.independent_batter_wpa.notna()).sum()),
        'heldout_we':{'games':int(scored.game_pk.nunique()),'pa':len(scored),
            'brier_pa_weighted':float(scored.brier.mean()),'brier_game_balanced':float(scored.groupby('game_pk').brier.mean().mean()),
            'brier_game_cluster_95ci':bootstrap_mean_ci(scored,'brier',n=plan['bootstrap_replicates'],game_balanced=True),
            'uncalibrated_brier_pa_weighted':float(scored.raw_brier.mean()),
            'training_prior_home_win_rate':model.baseline,'prior_baseline_brier':float(scored.baseline_brier.mean()),
            'prior_baseline_brier_game_balanced':float(scored.groupby('game_pk').baseline_brier.mean().mean()),
            'log_loss':float(scored.log_loss.mean()),
            'ece_10bin_pa_weighted':float((bins.pa*abs(bins.mean_prediction-bins.home_win_fraction)).sum()/bins.pa.sum())},
        'reference_cross_model_agreement':comparison,'li_normalizer_training_only':li.normalizer,
        'calibration_intercept_slope':[float(x) for x in model.calibration],
        'optimizer':model.optimization,'adequately_validated_for_rankings':False,
        'fitted_we_training_games':len(model.training_games),'fitted_calibration_games':len(model.calibration_games),
        'home_win_fractions':{role:float(q.groupby('game_pk').home_won.first().mean()) for role,q in data.groupby('role')},
        'excluded_censored_walkoff_kernel_donors':li.excluded_walkoff_donors,
        'structural_diagnostics':structural_diagnostics(model),
        'heldout_mean_absolute_expected_home_we_drift':float(test.kernel_expected_home_we_drift.abs().mean()),
        'interpretation':'POC only. Calibration/WE/kernel/normalization never use test games or provider values. Cross-provider differences are not absolute truth errors. Bootstrap intervals condition on fitted model; no training uncertainty or temporal validation.'}
    atomic_json(OUT/'results.json',report)
    atomic_json(OUT/'model.json',{'version':plan['version'],'coefficient_vector':model.beta.tolist(),
        'feature_names':FEATURE_NAMES,'fit_game_ids':sorted(model.training_games),
        'calibration_game_ids':sorted(model.calibration_games),'normalization_game_ids':sorted(li.normalization_games),
        'calibration':model.calibration.tolist(),'normalizer':li.normalizer,
        'kernel':[{'outs':key[0],'bases':key[1],'outcomes':[{'runs':o[0],'added_outs':o[1],'after_bases':o[2],'count':n} for o,n in counts.items()]} for key,counts in li.counts.items()]})
    plot_results(test,bins)
    atomic_json(OUT/'provenance.json',{'generated_at_utc':utc_now(),'game_receipts':receipts,
        'design_sha256':hashlib.sha256((OUT/'design.json').read_bytes()).hexdigest(),
        'design_addendum_sha256':hashlib.sha256((OUT/'design_addendum.json').read_bytes()).hexdigest(),
        'formula_sha256':hashlib.sha256((ROOT/'config/research_v1.json').read_bytes()).hexdigest(),
        'reference_pa_sha256':hashlib.sha256((ROOT/'reports/reference_review_v04/reference_pas.parquet').read_bytes()).hexdigest(),
        'code_sha256':{str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest() for p in (ROOT/'src/mlb_wpa').glob('*.py')},
        'outputs_sha256':{str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest() for p in OUT.iterdir() if p.suffix in ('.csv','.parquet','.png')},
        'no_provider_numeric_fit':'All fitting uses official event states and final home winner labels only.'})
    print(json.dumps(report,indent=2))
    return report


if __name__=='__main__':run()
