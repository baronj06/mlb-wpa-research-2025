"""Conditional game-cluster uncertainty for the unchanged EOWS components."""

import numpy as np
import pandas as pd
from scipy.stats import rankdata

from .eows_metrics import leverage_codes, original_components, composite


def bootstrap(data,players,reps=500,seed=2025111801):
    events=data.loc[data.metric_eligible]
    ids=pd.Index(players.player_id);games=pd.Index(sorted(data.game_pk.unique()));G=len(games);P=len(ids)
    g=games.get_indexer(events.game_pk);p=ids.get_indexer(events.batter_id)
    c=leverage_codes(events.estimated_pre_pa_li);w=events.estimated_batter_wpa.to_numpy(float)
    lc=np.zeros((G,3));ls=lc.copy();np.add.at(lc,(g,c),1);np.add.at(ls,(g,c),w)
    pc=np.zeros((G,P*3));ps=pc.copy();pp=pc.copy();keep=p>=0;index=p[keep]*3+c[keep]
    np.add.at(pc,(g[keep],index),1);np.add.at(ps,(g[keep],index),w[keep]);np.add.at(pp,(g[keep],index),(w[keep]>0))
    features=np.concatenate([lc,ls,pc,ps,pp],axis=1)
    rng=np.random.default_rng(seed);weights=rng.multinomial(G,np.full(G,1/G),size=reps).astype(float)
    totals=weights@features
    LC=totals[:,:3];LS=totals[:,3:6]
    start=6;PC=totals[:,start:start+P*3].reshape(reps,P,3);start+=P*3
    PS=totals[:,start:start+P*3].reshape(reps,P,3);start+=P*3
    PP=totals[:,start:start+P*3].reshape(reps,P,3)
    usable=np.all(PC>0,axis=(1,2))&np.all(LC>0,axis=1)
    components=original_components(PC[usable],PS[usable],PP[usable],LC[usable],LS[usable])
    scores,_=composite(components);ranks=rankdata(-scores,method='min',axis=1)
    intervals=pd.DataFrame({'player_id':ids})
    for name,a in zip(['EOWS_100','DI','CR','EOWS_C','Rank'],[*components,scores,ranks]):
        intervals[name+'_ci_low']=np.quantile(a,.025,axis=0);intervals[name+'_ci_high']=np.quantile(a,.975,axis=0)
    intervals['bootstrap_top10_probability']=(ranks<=10).mean(axis=0)
    intervals['bootstrap_mean_EOWS_C']=scores.mean(axis=0)
    intervals['bootstrap_median_rank']=np.median(ranks,axis=0)
    group_rows=[]
    for group in ['Power','On-Base','Hybrid','Other']:
        mask=players.Group.eq(group).to_numpy()
        if not mask.any():continue
        for name,a in zip(['EOWS_100','DI','CR','EOWS_C'],[*components,scores]):
            means=a[:,mask].mean(axis=1)
            group_rows.append({'comparison':group,'metric':name,'n_players':int(mask.sum()),
                'ci_low':float(np.quantile(means,.025)),'ci_high':float(np.quantile(means,.975)),
                'bootstrap_mean':float(means.mean())})
    A=players.Group.eq('Power').to_numpy();B=players.Group.eq('On-Base').to_numpy()
    if A.any() and B.any():
        for name,a in zip(['EOWS_100','DI','CR','EOWS_C'],[*components,scores]):
            diff=a[:,A].mean(axis=1)-a[:,B].mean(axis=1)
            group_rows.append({'comparison':'Power minus On-Base','metric':name,'n_players':int(A.sum()+B.sum()),
                'ci_low':float(np.quantile(diff,.025)),'ci_high':float(np.quantile(diff,.975)),
                'bootstrap_mean':float(diff.mean())})
    return intervals,pd.DataFrame(group_rows),{'requested_reps':reps,'usable_reps':int(usable.sum()),
        'replicas_rejected_for_any_empty_player_category':int((~usable).sum()),'games_per_replica':G,
        'fixed_common_players':P,'seed':seed,'scope':'Game resampling only; model, eligibility and groups fixed; no refitting or missing-value imputation.',
        'interval_interpretation':'Percentile game-bootstrap resampling ranges; not calibrated confidence bounds for nonlinear ranks/composites. Bootstrap bias is reported.'}
