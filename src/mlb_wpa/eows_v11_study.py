"""Predeclared, role-isolated v1.1 candidates; v1.0 and WE/LI remain frozen."""
import json
from dataclasses import dataclass
import numpy as np
import pandas as pd
from scipy.optimize import minimize_scalar
from scipy.stats import spearmanr,rankdata
from .sample import ROOT
from .eows_metrics import original_components,percentile_scores,leverage_codes,composite
from .eows_interpretation import game_tensors,component_draws

OUT=ROOT/'reports/eows_v11_study';TABLE=ROOT/'outputs/tables/eows_v11_study';CHART=ROOT/'outputs/charts/eows_v11_study'


def cluster_mean_variance(w,n,s):
    """Ratio mean and cluster sandwich sampling variance (not posterior variance)."""
    w=np.atleast_2d(w);N=w@n;S=w@s;G=w@(n>0)
    mean=np.divide(S,N,out=np.full_like(S,np.nan),where=N>0)
    rss=w@(s*s)-2*mean*(w@(s*n))+mean*mean*(w@(n*n))
    var=np.divide(np.maximum(rss,0),N*N,out=np.full_like(S,np.nan),where=N>0)
    var*=np.divide(G,G-1,out=np.full_like(G,np.nan),where=G>1)
    return mean,var,N,G


def fit_tau2(di,variance,valid):
    """Marginal ML variance in DI units, target fixed at league mean (DI=0)."""
    y=np.asarray(di)[valid];v=np.asarray(variance)[valid]
    if len(y)<20:raise ValueError('Insufficient development support for EB')
    if not np.isfinite(y).all() or not np.isfinite(v).all() or np.any(v<0):raise ValueError('Invalid EB inputs')
    # A genuinely zero estimated noise variance is kept, with a numeric lower
    # bound only inside log likelihood; no player observation is manufactured.
    v=np.maximum(v,1e-12)
    loss=lambda t:.5*np.sum(np.log(t+v)+y*y/(t+v))
    result=minimize_scalar(loss,bounds=(0.,10000.),method='bounded',options={'xatol':1e-9})
    return float(min([0.,float(result.x),10000.],key=loss))


def shrink_di(di,variance,tau2):
    di=np.asarray(di,float);v=np.asarray(variance,float)
    if not np.isfinite(tau2).all() or np.any(np.asarray(tau2)<0) or np.any(v[np.isfinite(v)]<0):raise ValueError('Invalid shrinkage variance')
    reliability=np.divide(tau2,tau2+v,out=np.zeros_like(v),where=(tau2+v)>0)
    reliability=np.where(np.isfinite(di)&np.isfinite(v),reliability,np.nan)
    return reliability*di,reliability


def fit_adjustment(cr,obp,opportunities,pa):
    x=np.column_stack([np.ones(len(cr)),obp,opportunities]);keep=np.isfinite(x).all(1)&np.isfinite(cr)&(pa>0)
    root=np.sqrt(pa[keep]);return np.linalg.lstsq(x[keep]*root[:,None],cr[keep]*root,rcond=None)[0]


def adjusted_frequency(cr,obp,opportunities,beta):
    return cr-(beta[0]+beta[1]*obp+beta[2]*opportunities)


def gate_apf(repeatability,n,correlations):
    return bool(n>=100 and np.isfinite(repeatability) and repeatability>=.25 and all(np.isfinite(v) and abs(v)<.90 for v in correlations))


def safe_rho(a,b,mask=None):
    a=np.asarray(a);b=np.asarray(b);ok=np.isfinite(a)&np.isfinite(b)
    if mask is not None:ok&=mask
    if ok.sum()<3 or len(np.unique(a[ok]))<2 or len(np.unique(b[ok]))<2:return np.nan
    return float(spearmanr(a[ok],b[ok]).statistic)


def framework_scores(z,eb,apf,use_d):
    original=[z['EOWS_100'],z['DI'],z['CR']]
    scores={'A_v1_0':composite(original)[0],
      'C_zero_CR_raw_DI':composite(original,(.7,.3,0))[0],
      'C_zero_CR_EB_DI':composite([z['EOWS_100'],eb,z['CR']],(.7,.3,0))[0]}
    if use_d:scores['D_APF_EB_DI']=composite([z['EOWS_100'],eb,apf])[0]
    return scores


@dataclass
class Study:
    data:pd.DataFrame
    players:pd.DataFrame

    def __post_init__(self):
        self.ids=self.players.player_id.to_numpy();self.P=len(self.ids)
        self.games,self.lc,self.ls,self.pc,self.ps,self.pp=game_tensors(self.data,self.ids)
        self.sq=np.zeros_like(self.pc)
        ev=self.data.loc[self.data.metric_eligible];gi=self.games.get_indexer(ev.game_pk);pi=pd.Index(self.ids).get_indexer(ev.batter_id)
        keep=pi>=0;codes=leverage_codes(ev.estimated_pre_pa_li)
        np.add.at(self.sq,(gi[keep],pi[keep],codes[keep]),ev.estimated_batter_wpa.to_numpy()[keep]**2)
        observed=np.zeros((len(self.games),self.P));g=self.games.get_indexer(self.data.game_pk);p=pd.Index(self.ids).get_indexer(self.data.batter_id);k=p>=0
        np.add.at(observed,(g[k],p[k]),1)
        self.complete=(observed==self.pc.sum(2))&(observed>0)
        self.game_positive=(self.ps.sum(2)>0)&self.complete
        self.appeared=observed>0
        self.box=np.zeros((len(self.games),self.P,7));idx=pd.Index(self.ids)
        for g,pk in enumerate(self.games):
            for row in json.loads((ROOT/f'data/processed/season_2025/games/{pk}_official_stats.json').read_text()):
                p=idx.get_indexer([row['player_id']])[0]
                if p>=0:self.box[g,p]+=np.array([row[k] for k in ['PA','H','HR','BB','HBP','SF','AB']])
        stats=pd.read_csv(ROOT/'data/processed/season_2025_official_player_stats.csv').set_index('player_id').loc[self.ids]
        if len(self.games)==2430:
            assert np.array_equal(self.box.sum(0),stats[['PA','H','HR','BB','HBP','SF','AB']])

    def summarize(self,weights):
        w=np.atleast_2d(weights);lc,ls,pc,ps,pp=component_draws(w,self.lc,self.ls,self.pc,self.ps,self.pp)
        e,d,c=original_components(pc,ps,pp,lc,ls)
        prob=lc/lc.sum(1,keepdims=True);means=ps/pc
        sq=(w@self.sq.reshape(len(self.games),-1)).reshape(len(w),self.P,3)/pc
        sd=np.sqrt(np.maximum((prob[:,None,:]*sq).sum(2)-(prob[:,None,:]*means).sum(2)**2,0))
        mean,var,n,g=cluster_mean_variance(w,self.pc[:,:,2],self.ps[:,:,2])
        ng=w@self.complete;gpr=np.divide(100*(w@self.game_positive),ng,out=np.full_like(ng,np.nan),where=ng>0)
        box=(w@self.box.reshape(len(self.games),-1)).reshape(len(w),self.P,7)
        pa,h,hr,bb,hbp,sf,ab=np.moveaxis(box,-1,0)
        obp=(h+bb+hbp)/(ab+bb+hbp+sf)
        validpa=pc.sum(2);opps=validpa/(w@self.appeared)
        return {'EOWS_100':e,'DI':d,'CR':c,'GPR':gpr,'SD_WPA':sd,'OBP':obp,'PA':pa,
          'valid_PA':validpa,'opportunities':opps,'High_PA':n,'High_games':g,'High_mean':mean,'High_variance':var,
          'DI_variance':10000*var,'complete_player_games':ng,'high_benchmark':ls[:,2]/lc[:,2]}


def one(z):return {k:v[0] for k,v in z.items()}


def run():
    design=json.loads((OUT/'design.json').read_text())
    data=pd.read_parquet(ROOT/'data/processed/season_2025_independent_markov_v03.parquet')
    players=pd.read_csv(ROOT/'outputs/tables/eows_2025_exploratory/preliminary_eows_c_rankings.csv').sort_values('player_id').reset_index(drop=True)
    study=Study(data,players);games=study.games;G=len(games)
    devmask=games.isin(design['development_games']);testmask=games.isin(design['evaluation_games'])
    assert not (devmask&testmask).any() and np.all(devmask|testmask)
    dev=one(study.summarize(devmask.astype(float)))
    f1=one(study.summarize(games.isin(design['development_fold1']).astype(float)))
    f2=one(study.summarize(games.isin(design['development_fold2']).astype(float)))
    beta1=fit_adjustment(f1['CR'],f1['OBP'],f1['opportunities'],f1['valid_PA'])
    beta=fit_adjustment(dev['CR'],dev['OBP'],dev['opportunities'],dev['valid_PA'])
    for z,b in [(dev,beta),(f1,beta1),(f2,beta1)]:z['APF']=adjusted_frequency(z['CR'],z['OBP'],z['opportunities'],b)
    support=(dev['High_PA']>=8)&(dev['High_games']>=5)&np.isfinite(dev['DI_variance'])
    tau=fit_tau2(dev['DI'],dev['DI_variance'],support)
    paired=(f1['High_PA']>=8)&(f2['High_PA']>=8)&(f1['complete_player_games']>=20)&(f2['complete_player_games']>=20)
    repeat=safe_rho(f1['APF'],f2['APF'],paired)
    corr=[safe_rho(dev['APF'],dev[k]) for k in ['OBP','EOWS_100','DI']]
    use_d=gate_apf(repeat,int(paired.sum()),corr)
    lock={'tau2_DI_units':tau,'APF_beta':beta.tolist(),'APF_fold1_beta':beta1.tolist(),'APF_dev_half_repeatability':repeat,
      'APF_gate_paired_players':int(paired.sum()),'APF_dev_correlations_OBP_EOWS_DI':corr,'D_gate_passed':use_d,
      'fit_games':design['development_games'],'evaluation_games_used_for_fit':[],
      'D_policy':'No D score if screen fails; still report APF as a diagnostic. Screen passing would not establish independent validity.'}
    (OUT/'selection_lock.json').write_text(json.dumps(lock,indent=2)+'\n')
    # Only after freezing all fitted parameters/selection, evaluate held-out roles.
    test=one(study.summarize(testmask.astype(float)));full=one(study.summarize(np.ones(G)))
    for z in [test,full]:z['APF']=adjusted_frequency(z['CR'],z['OBP'],z['opportunities'],beta)
    eb,reliability=shrink_di(full['DI'],full['DI_variance'],tau)
    dev_eb,_=shrink_di(dev['DI'],dev['DI_variance'],tau)
    valid=support&(test['High_PA']>=8)&(test['High_games']>=5)&np.isfinite(test['High_mean'])
    predictions={'raw_dev_high_mean':dev['High_mean'],'EB_dev_high_mean':dev['high_benchmark']+dev_eb/100,
                 'league_dev_high_mean':np.full(study.P,dev['high_benchmark'])}
    risk=[]
    for name,pred in predictions.items():
        risk.append({'predictor':name,'paired_players':int(valid.sum()),'MSE':float(np.mean((pred[valid]-test['High_mean'][valid])**2)),
          'MAE':float(np.mean(abs(pred[valid]-test['High_mean'][valid]))),'rank_repeatability':safe_rho(pred,test['High_mean'],valid)})
    pd.DataFrame(risk).to_csv(TABLE/'heldout_high_WPA_prediction.csv',index=False)
    candidates=[]
    for name in ['CR','GPR','SD_WPA','APF','EOWS_100','DI']:
        for role,z in [('development',dev),('evaluation',test),('full_descriptive',full)]:
            candidates.append({'metric':name,'role':role,'OBP_spearman':safe_rho(z[name],z['OBP']),
              'EOWS_spearman':safe_rho(z[name],z['EOWS_100']),'DI_spearman':safe_rho(z[name],z['DI']),
              'dev_half_repeatability':safe_rho(f1[name],f2[name],paired),
              'development_to_evaluation_repeatability':safe_rho(dev[name],test[name],valid),
              'paired_dev_halves':int(paired.sum()),'paired_dev_eval':int(valid.sum())})
    pd.DataFrame(candidates).to_csv(TABLE/'consistency_candidate_evaluation.csv',index=False)
    full['EB_DI']=eb;full['EB_reliability']=reliability
    profiles=players[['player_id','player_name','Group','PA']].copy()
    for key,z in full.items():
        if np.ndim(z)>0:profiles[key]=z
    profiles['High_SE']=np.sqrt(full['High_variance']);profiles['High_support_8PA_5games']=(full['High_PA']>=8)&(full['High_games']>=5)
    profiles.to_csv(TABLE/'proposed_v11_player_profiles.csv',index=False)
    cols=['OBP','CR','EOWS_100','DI','EB_DI','GPR','SD_WPA','APF']
    profiles[cols].corr(method='spearman').to_csv(TABLE/'full_component_correlations.csv')
    pd.DataFrame([{'High_minimum':n,'retained_players':int((full['High_PA']>=n).sum()),'excluded_players':int((full['High_PA']<n).sum())} for n in [20,50,80]]).to_csv(TABLE/'High_minimum_support_sensitivity.csv',index=False)
    point_scores=framework_scores(full,eb,full['APF'],use_d)
    # Same full-season draws reproduce v1.0; EB tau is refit using dev games only.
    M=np.random.default_rng(2025111801).multinomial(G,np.full(G,1/G),size=500).astype(float)
    boot=study.summarize(M);bdev=study.summarize(M*devmask)
    taus=[]
    for i in range(len(M)):
        ok=(bdev['High_PA'][i]>=8)&(bdev['High_games'][i]>=5)&np.isfinite(bdev['DI_variance'][i])
        taus.append(fit_tau2(bdev['DI'][i],bdev['DI_variance'][i],ok))
    taus=np.array(taus);beb,brel=shrink_di(boot['DI'],boot['DI_variance'],taus[:,None])
    # APF screening is fixed; resampled coefficients never include eval outcomes.
    bapf=[]
    for i in range(len(M)):
        b=fit_adjustment(bdev['CR'][i],bdev['OBP'][i],bdev['opportunities'][i],bdev['valid_PA'][i])
        bapf.append(adjusted_frequency(boot['CR'][i],boot['OBP'][i],boot['opportunities'][i],b))
    bootstrap_scores=framework_scores(boot,beb,np.array(bapf),use_d)
    # Discovered nuisance-bootstrap bias: provide a fixed-dev-tau sensitivity,
    # not an attempted confidence-interval correction or a tuned alternative.
    fixed_beb,_=shrink_di(boot['DI'],boot['DI_variance'],tau)
    fixed_scores=framework_scores(boot,fixed_beb,np.array(bapf),use_d)
    fixed_rows=[]
    for name,bs in fixed_scores.items():
        rr=rankdata(-bs,method='min',axis=1)
        fixed_rows.append({'framework':name,'hyperparameter_policy':'fixed development tau',
          'median_rank_range_width':float(np.median(np.quantile(rr,.975,axis=0)-np.quantile(rr,.025,axis=0))),
          'mean_absolute_bootstrap_score_bias':float(np.mean(abs(bs.mean(0)-point_scores[name])))})
    pd.DataFrame(fixed_rows).to_csv(TABLE/'fixed_tau_resampling_sensitivity.csv',index=False)
    pd.DataFrame({'replicate':np.arange(500),'refitted_development_tau2':taus,
                  'fixed_point_development_tau2':tau}).to_csv(TABLE/'EB_nuisance_bootstrap_diagnostic.csv',index=False)
    rows=[];ranking_rows=[]
    for name,sc in point_scores.items():
        bs=bootstrap_scores[name];ranks=rankdata(-bs,method='min',axis=1);point_rank=rankdata(-sc,method='min')
        width=np.quantile(ranks,.975,axis=0)-np.quantile(ranks,.025,axis=0)
        if name=='A_v1_0':
            assert np.allclose(sc,players.EOWS_C,atol=1e-10)
            assert np.allclose(np.quantile(ranks,.025,axis=0),players.Rank_ci_low)
        rows.append({'framework':name,'players':study.P,'median_rank_range_width':float(np.median(width)),
          'mean_rank_range_width':float(width.mean()),'width_over50':int((width>50).sum()),
          'mean_absolute_bootstrap_score_bias':float(np.mean(abs(bs.mean(0)-sc))),
          'rank_correlation_v1_0':safe_rho(point_rank,players.Rank),'distinct_point_scores':len(np.unique(sc))})
        for i in range(study.P):ranking_rows.append({'framework':name,'player_id':study.ids[i],
          'point_score':sc[i],'diagnostic_point_rank':point_rank[i],'rank_range_low':np.quantile(ranks[:,i],.025),
          'rank_range_high':np.quantile(ranks[:,i],.975),'score_resampling_sd':bs[:,i].std(ddof=1)})
    pd.DataFrame(rows).to_csv(TABLE/'framework_comparison.csv',index=False)
    pd.DataFrame(ranking_rows).to_csv(TABLE/'diagnostic_composite_comparisons.csv',index=False)
    # Evaluation-game resampling assesses fixed dev predictions (no re-fit).
    E=int(testmask.sum());ew=np.zeros((500,G));ew[:,testmask]=np.random.default_rng(2025111803).multinomial(E,np.full(E,1/E),size=500)
    emean,_,en,_=cluster_mean_variance(ew,study.pc[:,:,2],study.ps[:,:,2]);usable=np.isfinite(emean[:,valid]).all(1)
    losses={name:((emean[usable][:,valid]-pred[valid])**2).mean(1) for name,pred in predictions.items()}
    lossdiff=[]
    for a,b in [('raw_dev_high_mean','EB_dev_high_mean'),('league_dev_high_mean','EB_dev_high_mean')]:
        delta=losses[a]-losses[b]
        point=next(x['MSE'] for x in risk if x['predictor']==a)-next(x['MSE'] for x in risk if x['predictor']==b)
        lossdiff.append({'comparison':a+' minus '+b,'MSE_difference':point,'resampling_low':np.quantile(delta,.025),
          'resampling_high':np.quantile(delta,.975),'usable_game_replicas':int(usable.sum()),'rejected_missing_mean_replicas':int((~usable).sum())})
    pd.DataFrame(lossdiff).to_csv(TABLE/'heldout_MSE_comparison_uncertainty.csv',index=False)
    # Full-profile uncertainty, not an overall rank for B.
    intervalrows=[];group_rows=[]
    bprofile={'EOWS_100':boot['EOWS_100'],'DI':boot['DI'],'EB_DI':beb,'GPR':boot['GPR'],'SD_WPA':boot['SD_WPA'],'APF':np.array(bapf)}
    for name,v in bprofile.items():
        for i in range(study.P):intervalrows.append({'player_id':study.ids[i],'component':name,'estimate':full[name][i],
          'resampling_low':np.quantile(v[:,i],.025),'resampling_high':np.quantile(v[:,i],.975)})
        for group in ['Power','On-Base','Hybrid','Other']:
            m=players.Group.eq(group).to_numpy();a=v[:,m].mean(1)
            group_rows.append({'component':name,'Group':group,'n':int(m.sum()),'mean':float(full[name][m].mean()),
              'resampling_low':np.quantile(a,.025),'resampling_high':np.quantile(a,.975)})
        a=players.Group.eq('Power').to_numpy();b=players.Group.eq('On-Base').to_numpy();delta=v[:,a].mean(1)-v[:,b].mean(1)
        group_rows.append({'component':name,'Group':'Power minus On-Base','n':int(a.sum()+b.sum()),'mean':float(full[name][a].mean()-full[name][b].mean()),
          'resampling_low':np.quantile(delta,.025),'resampling_high':np.quantile(delta,.975)})
    for name,v in bootstrap_scores.items():
        for group in ['Power','On-Base','Hybrid','Other']:
            m=players.Group.eq(group).to_numpy();a=v[:,m].mean(1)
            group_rows.append({'component':name,'Group':group,'n':int(m.sum()),'mean':float(point_scores[name][m].mean()),
              'resampling_low':np.quantile(a,.025),'resampling_high':np.quantile(a,.975)})
    pd.DataFrame(intervalrows).to_csv(TABLE/'profile_component_resampling_ranges.csv',index=False)
    pd.DataFrame(group_rows).to_csv(TABLE/'group_framework_comparison.csv',index=False)
    # Stresses alter only new study views, never original rows/scores.
    stress=[];sdata=data.copy();events=sdata.loc[sdata.metric_eligible & sdata.batter_id.isin(study.ids)&(sdata.estimated_pre_pa_li>=2)].copy()
    events['absolute']=events.estimated_batter_wpa.abs();drop=events.groupby('batter_id').absolute.idxmax()
    sdata.loc[drop,'metric_eligible']=False
    complete_ids=data.groupby('game_pk').metric_eligible.all();complete=data.loc[data.game_pk.isin(complete_ids.loc[complete_ids].index)].copy()
    for label,changed in [('remove_one_extreme_High_PA_per_player',sdata),('complete_games_only',complete)]:
        st=Study(changed,players);z=one(st.summarize(np.ones(len(st.games))))
        z['APF']=adjusted_frequency(z['CR'],z['OBP'],z['opportunities'],beta)
        shrunk,_=shrink_di(z['DI'],z['DI_variance'],tau);zs=framework_scores(z,shrunk,z['APF'],use_d)
        for name,sc in zs.items():
            rr=rankdata(-sc,method='min');orig=rankdata(-point_scores[name],method='min')
            stress.append({'scenario':label,'framework':name,'retained_valid_PA':int(changed.metric_eligible.sum()),
             'players_below20_High':int((z['High_PA']<20).sum()),'rank_spearman':safe_rho(rr,orig),
             'mean_absolute_rank_shift':float(np.mean(abs(rr-orig))),'max_rank_shift':float(max(abs(rr-orig)))})
        for name in ['EOWS_100','DI','EB_DI','GPR','SD_WPA','APF']:
            vv=shrunk if name=='EB_DI' else z[name]
            stress.append({'scenario':label,'framework':'B_profile_'+name,'retained_valid_PA':int(changed.metric_eligible.sum()),
             'players_below20_High':int((z['High_PA']<20).sum()),'rank_spearman':safe_rho(vv,full[name]),
             'mean_absolute_component_change':float(np.nanmean(abs(vv-full[name])))})
    pd.DataFrame(stress).to_csv(TABLE/'outlier_and_missingness_sensitivity.csv',index=False)
    summary={'original_v1_0_reproduced':True,'development_games':int(devmask.sum()),'evaluation_games':int(testmask.sum()),
      'evaluation_games_in_WE_fitting_calibration_selection':0,'tau2_DI_units':tau,'EB_mean_reliability':float(np.mean(reliability)),
      'EB_median_reliability':float(np.median(reliability)),'EB_bootstrap_tau2_zero_fraction':float((taus==0).mean()),
      'EB_point_tau2_outside_bootstrap_range':bool(tau<np.quantile(taus,.025) or tau>np.quantile(taus,.975)),
      'EB_bootstrap_tau2_quantiles':np.quantile(taus,[.025,.5,.975]).tolist(),'D_APF_gate_passed':use_d,
      'dev_APF_half_repeatability':repeat,'paired_evaluation_players':int(valid.sum()),'framework_count':len(rows),
      'complete_games_stress_games':int(complete.game_pk.nunique()),'network_baseball_requests':0,'WE_LI_retrained':False,
      'interpretation':'Candidates only; no automatic adoption, calibrated true-rank CI, predictive superiority or new blind confirmatory sample'}
    (OUT/'results.json').write_text(json.dumps(summary,indent=2)+'\n')
    plots(profiles,pd.DataFrame(rows),pd.DataFrame(risk),pd.DataFrame(candidates))
    print(json.dumps(summary,indent=2));print(pd.DataFrame(risk).to_string(index=False));print(pd.DataFrame(rows).to_string(index=False))


def plots(profile,frameworks,risk,candidates):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    plt.rcParams.update({'font.size':10,'axes.spines.top':False,'axes.spines.right':False})
    fig,ax=plt.subplots(figsize=(10,5));q=candidates.loc[candidates.role.eq('evaluation')].set_index('metric').loc[['CR','GPR','SD_WPA','APF']]
    for i,col in enumerate(['OBP_spearman','EOWS_spearman','DI_spearman']):ax.bar(np.arange(4)+(i-1)*.24,q[col],width=.24,label=col.replace('_spearman',''))
    ax.set_xticks(range(4),q.index);ax.set(ylim=(-1,1),ylabel='Spearman correlation',title='Held-out role: lower redundancy alone does not validate a candidate');ax.axhline(0,c='gray');ax.legend()
    fig.tight_layout();fig.savefig(CHART/'candidate_redundancy.png',dpi=200);plt.close(fig)
    fig,axes=plt.subplots(1,2,figsize=(12,4.5));axes[0].scatter(profile.High_PA,profile.DI,alpha=.6,s=16,label='Raw DI');axes[0].scatter(profile.High_PA,profile.EB_DI,alpha=.6,s=16,label='EB DI')
    axes[0].set(xlabel='Valid High PA',ylabel='DI (wins per100 High PA)',title='Working-model shrinkage is a distinct estimate');axes[0].legend()
    axes[1].barh(risk.predictor,risk.MSE*10000,color='#147b88');axes[1].set(xlabel='Held-out mean prediction MSE ×10,000',title='Compare raw, EB and league-only predictions')
    fig.tight_layout();fig.savefig(CHART/'high_leverage_shrinkage_validation.png',dpi=200);plt.close(fig)
    fig,ax=plt.subplots(figsize=(9,4.5));ax.barh(frameworks.framework,frameworks.median_rank_range_width,color='#785497')
    ax.set(xlabel='Median 2.5–97.5% rank resampling width',title='Stability is descriptive, not a selection criterion')
    fig.supxlabel('B is a profile with no overall rank; D is omitted if its development screen fails.',fontsize=9)
    fig.tight_layout();fig.savefig(CHART/'framework_rank_stability.png',dpi=200);plt.close(fig)
    fig,axes=plt.subplots(1,2,figsize=(12,4.5))
    for name,color in [('Power','#d78b32'),('On-Base','#147b88'),('Hybrid','#785497'),('Other','#9aabb3')]:
        q=profile.loc[profile.Group.eq(name)];axes[0].scatter(q.EOWS_100,q.EB_DI,c=color,s=17,label=name);axes[1].scatter(q.OBP,q.GPR,c=color,s=17)
    axes[0].set(xlabel='Unchanged EOWS/100',ylabel='Candidate EB DI',title='Proposed profile: contribution and High impact');axes[0].legend()
    axes[1].set(xlabel='Official OBP',ylabel='Positive complete player-games (%)',title='Game positivity remains opportunity-dependent')
    fig.tight_layout();fig.savefig(CHART/'proposed_multidimensional_profile.png',dpi=200);plt.close(fig)


if __name__=='__main__':run()
