"""Unmodified EOWS v1.0 component/composite calculations and fixed conventions."""

import numpy as np
import pandas as pd
from scipy.stats import rankdata


def leverage_codes(li,high=2.0):
    li=np.asarray(li,float)
    if not np.isfinite(li).all() or np.any(li<0):raise ValueError('LI must be finite and nonnegative')
    return np.where(li<.85,0,np.where(li<high,1,2))


def original_components(counts,wpa_sums,positive_counts,league_counts,league_sums):
    counts=np.asarray(counts,float);sums=np.asarray(wpa_sums,float);positives=np.asarray(positive_counts,float)
    lc=np.asarray(league_counts,float);ls=np.asarray(league_sums,float)
    means=np.divide(sums,counts,out=np.full_like(sums,np.nan),where=counts>0)
    rates=np.divide(positives,counts,out=np.full_like(positives,np.nan),where=counts>0)
    p=np.divide(lc,lc.sum(axis=-1,keepdims=True),out=np.full_like(lc,np.nan),where=lc.sum(axis=-1,keepdims=True)>0)
    benchmark=np.divide(ls[...,2],lc[...,2],out=np.full_like(ls[...,2],np.nan),where=lc[...,2]>0)
    EOWS=100*np.sum(np.expand_dims(p,-2)*means,axis=-1)
    DI=100*(means[...,2]-np.expand_dims(benchmark,-1))
    CR=100*np.sum(np.expand_dims(p,-2)*rates,axis=-1)
    return EOWS,DI,CR


def percentile_scores(values):
    a=np.asarray(values,float)
    if a.shape[-1]<2:raise ValueError('At least two common eligible players required')
    if not np.isfinite(a).all():raise ValueError('Common percentile population must be finite')
    return 100*(rankdata(a,method='average',axis=-1)-1)/(a.shape[-1]-1)


def composite(components,weights=(.4,.3,.3)):
    weights=np.asarray(weights,float)
    if len(weights)!=3 or np.any(weights<0) or not np.isclose(weights.sum(),1):raise ValueError('Invalid weights')
    percentiles=[percentile_scores(a) for a in components]
    integer_percent=np.rint(weights*100)
    if np.allclose(weights*100,integer_percent,atol=1e-12,rtol=0):
        # Algebraically identical to sum(weight * percentile), with one final
        # division so mathematically tied composites do not split by one ULP.
        ranks=[rankdata(np.asarray(a,float),method='average',axis=-1) for a in components]
        score=sum(w*(r-1) for w,r in zip(integer_percent,ranks))/(np.asarray(components[0]).shape[-1]-1)
    else:
        score=sum(w*a for w,a in zip(weights,percentiles))
    return score,percentiles


def summarize_events(events,player_ids,li_values=None,high=2.):
    ids=pd.Index(player_ids);P=len(ids)
    values=events.estimated_pre_pa_li.to_numpy(float) if li_values is None else np.asarray(li_values,float)
    c=leverage_codes(values,high);w=events.estimated_batter_wpa.to_numpy(float)
    if not np.isfinite(w).all():raise ValueError('Eligible WPA must be finite')
    player=ids.get_indexer(events.batter_id)
    selected=player>=0;index=player[selected]*3+c[selected]
    counts=np.bincount(index,minlength=P*3).reshape(P,3)
    sums=np.bincount(index,weights=w[selected],minlength=P*3).reshape(P,3)
    positives=np.bincount(index,weights=(w[selected]>0).astype(int),minlength=P*3).reshape(P,3)
    lc=np.bincount(c,minlength=3);ls=np.bincount(c,weights=w,minlength=3)
    return counts,sums,positives,lc,ls


def scoring_table(events,player_ids,li_values=None,high=2.,weights=(.4,.3,.3)):
    counts,sums,positive,lc,ls=summarize_events(events,player_ids,li_values,high)
    components=original_components(counts,sums,positive,lc,ls)
    finite=np.logical_and.reduce([np.isfinite(a) for a in components])
    output=pd.DataFrame({'player_id':player_ids,'EOWS_100':components[0],'DI':components[1],'CR':components[2],
                         'eligible_PA':counts.sum(axis=1),'Estimated_WPA':sums.sum(axis=1),
                         'Low_PA':counts[:,0],'Medium_PA':counts[:,1],'High_PA':counts[:,2]})
    output['EOWS_C']=np.nan;output['Rank']=np.nan
    for name in ['E','D','C']:output[name]=np.nan
    if finite.sum()>=2:
        score,percentiles=composite([a[finite] for a in components],weights)
        output.loc[finite,'EOWS_C']=score;output.loc[finite,'Rank']=rankdata(-score,method='min')
        for name,a in zip(['E','D','C'],percentiles):output.loc[finite,name]=a
    return output,{'category_proportions':(lc/lc.sum()).tolist(),'league_category_pa':lc.tolist(),
                   'high_league_mean_wpa':float(ls[2]/lc[2]),'eligible_league_pa':int(lc.sum())}


def classify_hitters(stats):
    """Study-qualified >=400 PA population, fixed before model exclusions."""
    q=stats.loc[stats.PA>=400].copy()
    q['HR_per_PA']=q.HR/q.PA
    q['OBP']=(q.H+q.BB+q.HBP)/(q.AB+q.BB+q.HBP+q.SF).replace(0,np.nan)
    h=float(q.HR_per_PA.quantile(.75,interpolation='linear'))
    o=float(q.OBP.quantile(.75,interpolation='linear'))
    power=q.HR_per_PA>=h;onbase=q.OBP>=o
    q['Group']=np.select([power&onbase,power&~onbase,~power&onbase],['Hybrid','Power','On-Base'],default='Other')
    return q,{'population':'official PA>=400 before model exclusions','n':len(q),'HR_per_PA_q75':h,'OBP_q75':o,
              'boundary':'linear quantile; >= includes ties','groups':q.Group.value_counts().to_dict()}
