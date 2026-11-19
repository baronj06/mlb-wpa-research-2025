"""Controlled, offline LI study with non-reference WE selection validation."""

from collections import Counter
import hashlib
import json
import os

import numpy as np
import pandas as pd

from .collection import CachedClient, read_cached_body, atomic_json, utc_now
from .independent_we import extract_game, attributed_wpa
from .li_candidates import MarkovWE, PotentialLI, load_baseline
from .sample import ROOT
from .we_poc import paired_errors, bootstrap_mean_ci, calibration_table

OUT=ROOT/'reports/li_accuracy_study'
CATEGORIES=['Low','Medium','High']


def categories(values):
    a=np.asarray(values,dtype=float)
    return np.where(~np.isfinite(a),'Missing',np.where(a<.85,'Low',np.where(a<2,'Medium','High')))


def evaluate(frame):
    known=frame.loc[frame.reference_validated].copy()
    true=categories(known.reference_li);pred=categories(known.predicted_li)
    confusion=pd.crosstab(pd.Series(true,name='Reference'),pd.Series(pred,name='Predicted')).reindex(
        index=CATEGORIES,columns=CATEGORIES+['Missing'],fill_value=0)
    tp=int(confusion.loc['High','High']);positives=int(confusion.loc['High'].sum())
    predicted=int(confusion['High'].sum())
    paired=known.dropna(subset=['predicted_li']);wpa=known.dropna(subset=['predicted_wpa'])
    r={'reference_pa':len(known),'li_paired_pa':len(paired),'li_missing_pa':int(known.predicted_li.isna().sum()),
       'accuracy':float(np.mean(true==pred)),'high_precision':tp/predicted if predicted else None,
       'high_recall':tp/positives,'high_false_negative_rate':1-tp/positives,
       'low_recall':float(confusion.loc['Low','Low']/confusion.loc['Low'].sum()),
       'medium_recall':float(confusion.loc['Medium','Medium']/confusion.loc['Medium'].sum()),
       'high_reference_pa':positives,'high_false_negatives':positives-tp,
       'li':paired_errors(paired.predicted_li,paired.reference_li),
       'wpa':paired_errors(wpa.predicted_wpa,wpa.reference_wpa)}
    for label,q,a,b in [('li',paired,'predicted_li','reference_li'),('wpa',wpa,'predicted_wpa','reference_wpa')]:
        q=q.copy();q['ae']=abs(q[a]-q[b]);r[label]['mae_cluster_95ci']=bootstrap_mean_ci(q,'ae')
    high=known.loc[known.reference_li.ge(2)].copy();high['hit']=high.predicted_li.ge(2).astype(float)
    r['high_recall_cluster_95ci']=bootstrap_mean_ci(high,'hit')
    return r,confusion


def error_breakdowns(frame,label):
    q=frame.loc[frame.reference_validated].copy();q['correct']=categories(q.predicted_li)==categories(q.reference_li)
    q['high']=q.reference_li.ge(2);q['miss']=q.high & ~q.predicted_li.ge(2)
    diff=q.home_score-q.away_score;q['score_diff']=diff
    q['inning_band']=np.where(q.inning<=6,'1-6',np.where(q.inning<=9,'7-9','10+'))
    q['score_band']=np.where(abs(diff)==0,'Tied',np.where(abs(diff)==1,'1 run',np.where(abs(diff)==2,'2 runs','3+ runs')))
    q['runners_scoring_position']=q.bases.str[1:].ne('__')
    q['two_outs']=q.outs.eq(2);q['bases_loaded']=q.bases.eq('123')
    runners=q.bases.str.count('[123]')
    q['walkoff_opportunity']=(q.half.eq('bottom')&q.inning.ge(9)&diff.le(0)&diff.ge(-runners))
    q['near_085']=abs(q.reference_li-.85)<=.10;q['near_2']=abs(q.reference_li-2)<=.20
    q['high_li_band']=np.where(q.reference_li<2,'Below 2',np.where(q.reference_li<=2.2,'2.0-2.2',np.where(q.reference_li<3,'2.2-3.0','3.0+')))
    q['ae']=abs(q.predicted_li-q.reference_li);q['se']=(q.predicted_li-q.reference_li)**2
    output=[];total_fn=int(q.miss.sum())
    for field in ['inning','inning_band','score_diff','score_band','outs','bases','half','runners_scoring_position',
                  'two_outs','bases_loaded','walkoff_opportunity','unusual_running','near_085','near_2','high_li_band']:
        for value,g in q.groupby(field,dropna=False):
            n_high=int(g.high.sum());fn=int(g.miss.sum())
            output.append({'model':label,'dimension':field,'level':str(value),'reference_pa':len(g),
                'paired_pa':int(g.predicted_li.notna().sum()),'accuracy':float(g.correct.mean()),
                'li_mae':float(g.ae.mean()),'li_rmse':float(np.sqrt(g.se.mean())),
                'high_pa':n_high,'high_false_negatives':fn,
                'high_miss_rate':fn/n_high if n_high else None,'share_of_all_high_misses':fn/total_fn if total_fn else 0})
    q.loc[q.miss].to_csv(OUT/f'{label}_high_false_negatives.csv',index=False)
    return output


def score_we(model,rows):
    active=[r for r in rows if r['start_state'] is not None]
    p=model.predict([r['start_state'] for r in active]);y=np.array([r['home_won'] for r in active])
    q=pd.DataFrame({'game_pk':[r['game_pk'] for r in active],'home_won':y,'independent_pre_pa_home_we':p})
    q['brier']=(p-y)**2
    logp=np.clip(p,1e-12,1-1e-12)
    bins=calibration_table(q)
    return {'pa':len(q),'games':int(q.game_pk.nunique()),
        'brier_game_balanced':float(q.groupby('game_pk').brier.mean().mean()),
        'brier_pa_weighted':float(q.brier.mean()),'log_loss':float(np.mean(-(y*np.log(logp)+(1-y)*np.log1p(-logp)))),
        'ece':float(np.sum(bins.pa*abs(bins.mean_prediction-bins.home_win_fraction))/bins.pa.sum())},bins


def run():
    plan=json.loads((OUT/'design.json').read_text())
    names=['training_games','calibration_games','selection_validation_games','reference_test_games']
    sets={name:set(plan[name]) for name in names}
    if any(sets[a]&sets[b] for i,a in enumerate(names) for b in names[:i]):raise ValueError('Game split leakage')
    client=CachedClient(offline=True);rows_by_role={name:[] for name in names};receipts={}
    roles={pk:role for role,ids in sets.items() for pk in ids}
    for i,(pk,role) in enumerate(sorted(roles.items())):
        record=json.loads((ROOT/f'data/checkpoints/season_2025/games/{pk}.json').read_text());sources=record['sources']
        meta=json.loads(read_cached_body(client.root,sources['metadata']));plays=json.loads(read_cached_body(client.root,sources['wp']))
        if meta['gameData']['game']['pk']!=pk or meta['gameData']['game']['type']!='R':raise ValueError('Game identity')
        rows_by_role[role].extend(extract_game(pk,meta,plays,role))
        receipts[pk]={key:sources[key] for key in ('metadata','wp')}
        if (i+1)%200==0:print('Extracted cached games',i+1,flush=True)
    training=[r for r in rows_by_role['training_games'] if r['start_state'] is not None]
    calibration=[r for r in rows_by_role['calibration_games'] if r['start_state'] is not None]
    validation=rows_by_role['selection_validation_games'];testrows=rows_by_role['reference_test_games']
    original=json.loads((ROOT/'reports/expanded_we_once/model.json').read_text())
    baseline,parents=load_baseline(original)
    markov=MarkovWE(parents,baseline.training_games,plan['max_half_inning_runs'],plan['max_pmf_tail']).calibrate(calibration)
    validation_scores={name:score_we(m,validation)[0] for name,m in [('baseline',baseline),('markov',markov)]}
    selected='markov' if validation_scores['markov']['brier_game_balanced']<validation_scores['baseline']['brier_game_balanced'] else 'baseline'
    selection={'locked_at_utc':utc_now(),'selected_we':selected,'validation_metrics':validation_scores,
        'rule':plan['selection'],'test_reference_values_used':False,'hyperparameter_search':False}
    atomic_json(OUT/'selection_lock.json',selection)
    # Outcome/reference test values have not been used in fitting or selection.
    print('Selection locked:',selected,flush=True)
    global_markov=PotentialLI(markov,parents,training)
    selected_model=markov if selected=='markov' else baseline
    contextual=PotentialLI(selected_model,parents,training,True,plan['hierarchical_prior_pa'])
    old=pd.read_csv(ROOT/'reports/expanded_we_once/heldout_reference_comparison.csv')
    ref=pd.read_parquet(ROOT/'reports/reference_review_v04/reference_pas.parquet')
    metadata=old[['game_pk','at_bat_index','batter_id','inning','half','outs','bases','home_score','away_score',
                  'reference_validated','fangraphs_pre_pa_li','fangraphs_batter_wpa','independent_pre_pa_li','independent_batter_wpa']].rename(
        columns={'fangraphs_pre_pa_li':'reference_li','fangraphs_batter_wpa':'reference_wpa',
                 'independent_pre_pa_li':'predicted_li','independent_batter_wpa':'predicted_wpa'})
    actions=ref[['game_pk','at_bat_index','independent_event_count','reason']].rename(columns={'reason':'reference_exclusion_reason'})
    metadata=metadata.merge(actions,on=['game_pk','at_bat_index'],validate='one_to_one')
    metadata['unusual_running']=metadata.independent_event_count.fillna(0).gt(0)|metadata.result.isin([]) if 'result' in metadata else metadata.independent_event_count.fillna(0).gt(0)
    if len(metadata)!=4800 or metadata.reference_validated.sum()!=4765:raise ValueError('Reference denominator changed')
    results={};matrices={};frames={'baseline':metadata};breakdowns=[]
    candidate_models={'markov_global':(markov,global_markov),'selected_context':(selected_model,contextual)}
    for name,(model,li) in candidate_models.items():
        predictions=[]
        for r in testrows:
            entry={'game_pk':r['game_pk'],'at_bat_index':r['at_bat_index'],'predicted_li':None,
                   'predicted_wpa':None,'model_exclusion_reason':r['reason'],
                   'expected_home_we_drift':None,'context_training_pa':None,'hierarchical_weight':None}
            if r['start_state'] is not None:
                value,drift,info=li.predict(r['start_state'])
                entry.update(predicted_li=value,expected_home_we_drift=drift,
                    context_training_pa=info['context_pa'],hierarchical_weight=info['hierarchical_weight'])
                if r['supported']:entry['predicted_wpa']=attributed_wpa(model,r['batter_before'],r['after_state'])
            predictions.append(entry)
        frames[name]=metadata.drop(columns=['predicted_li','predicted_wpa']).merge(pd.DataFrame(predictions),
                                      on=['game_pk','at_bat_index'],validate='one_to_one')
    for name,frame in frames.items():
        frame.to_csv(OUT/f'{name}_test_predictions.csv',index=False)
        result,confusion=evaluate(frame);results[name]=result;matrices[name]=confusion
        confusion.to_csv(OUT/f'{name}_confusion.csv');breakdowns.extend(error_breakdowns(frame,name))
    pd.DataFrame(breakdowns).to_csv(OUT/'error_breakdowns.csv',index=False)
    calibration_metrics={}
    for name,model in [('baseline',baseline),('markov',markov)]:
        metrics,bins=score_we(model,testrows);calibration_metrics[name]=metrics
        bins.to_csv(OUT/f'{name}_test_calibration.csv',index=False)
    comparison=[]
    for name,r in results.items():
        comparison.append({'model':name,'reference_pa':r['reference_pa'],'li_paired_pa':r['li_paired_pa'],
            'li_mae':r['li']['mae'],'li_rmse':r['li']['rmse'],'category_accuracy':r['accuracy'],
            'high_precision':r['high_precision'],'high_recall':r['high_recall'],
            'high_false_negative_rate':r['high_false_negative_rate'],'low_recall':r['low_recall'],
            'medium_recall':r['medium_recall'],'wpa_paired_pa':r['wpa']['n'],
            'wpa_mae':r['wpa']['mae'],'wpa_rmse':r['wpa']['rmse']})
    pd.DataFrame(comparison).to_csv(OUT/'performance_comparison.csv',index=False)
    support=[]
    for key,counts in contextual.child.items():
        n=sum(counts.values());support.append({'outs':key[0],'bases':key[1],'phase':key[2],'half':key[3],
            'batting_score_sign':key[4],'pa':n,'local_weight':n/(n+contextual.prior_pa)})
    pd.DataFrame(support).to_csv(OUT/'hierarchical_support.csv',index=False)
    report={'generated_at_utc':utc_now(),'design_version':plan['version'],'network_requests':client.network_requests,
        'selection':selection,'test_results':results,'test_we_calibration':calibration_metrics,
        'reference_total_pa':4800,'reference_verified_pa':4765,'reference_unresolved_pa':35,
        'markov_calibration':[float(v) for v in markov.calibration],
        'max_run_distribution_tail':markov.max_truncated_tail,
        'li_normalizers_training_only':{'baseline':original['normalizer'],'markov_global':global_markov.normalizer,
                                      'selected_context':contextual.normalizer},
        'rankings_generated':False,'exploratory_eows_ready':False,
        'interpretation':'Frozen final-reference comparison after non-reference validation selection. Reference set was previously inspected in earlier research; not a fresh blinded confirmation. No LI threshold or scale fitted to reference values.'}
    atomic_json(OUT/'results.json',report)
    atomic_json(OUT/'model_parameters.json',{'markov_calibration':report['markov_calibration'],
        'run_pmf':{f'{key[0]}:{key[1]}':value.tolist() for key,value in markov.run_pmf.items()},
        'context_counts':[{'key':list(k),'outcomes':[{'outcome':list(o),'count':n} for o,n in counts.items()]} for k,counts in contextual.child.items()],
        'selection':selected,'prior_pa':contextual.prior_pa})
    atomic_json(OUT/'provenance.json',{'game_receipts':receipts,'generated_at_utc':utc_now(),
        'design_sha256':hashlib.sha256((OUT/'design.json').read_bytes()).hexdigest(),
        'formula_sha256':hashlib.sha256((ROOT/'config/research_v1.json').read_bytes()).hexdigest(),
        'source_code_sha256':{str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest() for p in (ROOT/'src/mlb_wpa').glob('*.py')},
        'baseline_model_sha256':hashlib.sha256((ROOT/'reports/expanded_we_once/model.json').read_bytes()).hexdigest(),
        'outputs_sha256':{str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest() for p in OUT.iterdir() if p.suffix=='.csv'}})
    print(pd.DataFrame(comparison).to_string(index=False),flush=True)
    return report


if __name__=='__main__':run()
