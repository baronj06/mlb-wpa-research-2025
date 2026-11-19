"""Offline interpretation diagnostics; original model and v1.0 scores stay frozen."""
from pathlib import Path
import json
import numpy as np
import pandas as pd
from scipy.stats import rankdata, spearmanr
from .eows_metrics import original_components, composite, percentile_scores, leverage_codes
from .sample import ROOT

REPORT=ROOT/'reports/eows_interpretation_v10'
TABLE=ROOT/'outputs/tables/eows_interpretation_v10'
CHART=ROOT/'outputs/charts/eows_interpretation_v10'
METRICS=['OBP','HR_per_PA','Estimated_WPA','EOWS_100','DI','CR','EOWS_C']
LEADERS=['Shohei Ohtani','Juan Soto','Iván Herrera','Will Smith','Freddie Freeman']
ONBASE={'single','double','triple','home_run','walk','intent_walk','hit_by_pitch'}


def fixed_percentile_mapping(values,reference):
    """Fixed point-sample midrank map with linear interpolation; diagnostic only."""
    reference=np.asarray(reference,float)
    u,inverse=np.unique(reference,return_inverse=True)
    scores=percentile_scores(reference)
    mid=np.bincount(inverse,weights=scores)/np.bincount(inverse)
    if len(u)==1:return np.full_like(values,50.,dtype=float)
    return np.interp(values,u,mid,left=0.,right=100.)


def game_tensors(data,ids):
    """Aggregate whole-game clusters, preserving all players in each game."""
    games=pd.Index(sorted(data.game_pk.unique()));ids=pd.Index(ids)
    e=data.loc[data.metric_eligible];g=games.get_indexer(e.game_pk);p=ids.get_indexer(e.batter_id)
    c=leverage_codes(e.estimated_pre_pa_li);w=e.estimated_batter_wpa.to_numpy()
    lc=np.zeros((len(games),3));ls=lc.copy();np.add.at(lc,(g,c),1);np.add.at(ls,(g,c),w)
    pc=np.zeros((len(games),len(ids),3));ps=pc.copy();pp=pc.copy();keep=p>=0
    for a,v in [(pc,np.ones(keep.sum())),(ps,w[keep]),(pp,(w[keep]>0).astype(float))]:
        np.add.at(a,(g[keep],p[keep],c[keep]),v)
    return games,lc,ls,pc,ps,pp


def component_draws(weights,lc,ls,pc,ps,pp):
    total=lambda a:(weights@a.reshape(len(weights.T),-1)).reshape((len(weights),)+a.shape[1:])
    return total(lc),total(ls),total(pc),total(ps),total(pp)


def residualize(y,obp):
    """Descriptive common linear OBP adjustment, refitted for every row/draw."""
    x=np.column_stack([np.ones(len(obp)),obp]);hat=np.linalg.pinv(x)
    return y-(y@hat.T)@x.T


def resampling_row(name,values,point):
    score,_=composite(values);ranks=rankdata(-score,method='min',axis=1)
    width=np.quantile(ranks,.975,axis=0)-np.quantile(ranks,.025,axis=0)
    return {'scenario':name,'median_rank_range_width':float(np.median(width)),
            'mean_rank_range_width':float(width.mean()),'players_width_over50':int((width>50).sum()),
            'median_score_bias':float(np.median(score.mean(axis=0)-point)),
            'mean_absolute_score_bias':float(np.mean(abs(score.mean(axis=0)-point)))},score,ranks,width


def run():
    for p in [REPORT,TABLE,CHART]:p.mkdir(parents=True,exist_ok=True)
    data=pd.read_parquet(ROOT/'data/processed/season_2025_independent_markov_v03.parquet')
    players=pd.read_csv(ROOT/'outputs/tables/eows_2025_exploratory/preliminary_eows_c_rankings.csv').sort_values('player_id').reset_index(drop=True)
    official=pd.read_csv(ROOT/'data/processed/season_2025_official_player_stats.csv').set_index('player_id').loc[players.player_id]
    ids=players.player_id.to_numpy();P=len(ids)
    games,lc,ls,pc,ps,pp=game_tensors(data,ids)
    weights=np.random.default_rng(2025111801).multinomial(len(games),np.full(len(games),1/len(games)),size=500).astype(float)
    LC,LS,PC,PS,PP=component_draws(weights,lc,ls,pc,ps,pp)
    assert (PC>0).all()
    pc0,ps0,pp0=pc.sum(0),ps.sum(0),pp.sum(0)
    base=original_components(PC,PS,PP,LC,LS)
    point=[players[c].to_numpy() for c in ['EOWS_100','DI','CR']]
    rows=[]
    row,scores,ranks,width=resampling_row('baseline',base,players.EOWS_C.to_numpy());rows.append(row)
    assert np.allclose(np.quantile(ranks,.025,axis=0),players.Rank_ci_low,atol=1e-9)
    assert np.allclose(np.quantile(ranks,.975,axis=0),players.Rank_ci_high,atol=1e-9)
    assert np.allclose(scores.mean(0),players.bootstrap_mean_EOWS_C,atol=1e-9)
    frozen_league=original_components(PC,PS,PP,np.broadcast_to(lc.sum(0),LC.shape),np.broadcast_to(ls.sum(0),LS.shape))
    row,_,_,_=resampling_row('fixed_league_distribution_and_benchmark',frozen_league,players.EOWS_C.to_numpy());rows.append(row)
    no_DI=[base[0],np.broadcast_to(point[1],base[1].shape),base[2]]
    row,_,_,_=resampling_row('fixed_player_DI',no_DI,players.EOWS_C.to_numpy());rows.append(row)
    frozen_ps=PS.copy();frozen_pp=PP.copy()
    frozen_ps[:,:,2]=PC[:,:,2]*(ps0[:,2]/pc0[:,2])[None,:]
    frozen_pp[:,:,2]=PC[:,:,2]*(pp0[:,2]/pc0[:,2])[None,:]
    fixed_high=original_components(PC,frozen_ps,frozen_pp,LC,LS)
    row,_,_,_=resampling_row('fixed_high_category_WPA_mean_and_positive_rate',fixed_high,players.EOWS_C.to_numpy());rows.append(row)
    fixed_scores=sum(w*fixed_percentile_mapping(a,p) for w,a,p in zip([.4,.3,.3],base,point))
    fixed_ranks=rankdata(-fixed_scores,method='min',axis=1)
    fw=np.quantile(fixed_ranks,.975,axis=0)-np.quantile(fixed_ranks,.025,axis=0)
    rows.append({'scenario':'fixed_point_component_percentile_mapping','median_rank_range_width':float(np.median(fw)),
      'mean_rank_range_width':float(fw.mean()),'players_width_over50':int((fw>50).sum()),
      'median_score_bias':float(np.median(fixed_scores.mean(0)-players.EOWS_C)),
      'mean_absolute_score_bias':float(np.mean(abs(fixed_scores.mean(0)-players.EOWS_C)))})
    erank=rankdata(-base[0],axis=1,method='min');ew=np.quantile(erank,.975,axis=0)-np.quantile(erank,.025,axis=0)
    rows.append({'scenario':'EOWS_component_rank_only','median_rank_range_width':float(np.median(ew)),
      'mean_rank_range_width':float(ew.mean()),'players_width_over50':int((ew>50).sum()),'median_score_bias':None,'mean_absolute_score_bias':None})
    pd.DataFrame(rows).to_csv(TABLE/'bootstrap_ablation.csv',index=False)
    diag=players[['player_id','player_name','Group','High_PA','Rank','EOWS_C']].copy()
    diag['rank_range_width']=width
    for name,a,p in zip(['EOWS_100','DI','CR'],base,point):
        diag[name+'_bootstrap_sd']=a.std(0,ddof=1);diag[name+'_bootstrap_bias']=a.mean(0)-p
    diag['score_bootstrap_bias']=scores.mean(0)-players.EOWS_C
    diag.to_csv(TABLE/'player_variability.csv',index=False)
    # All seven metrics on the identical fixed cohort; WPA means independent estimate.
    for method in ['pearson','spearman']:players[METRICS].corr(method=method).to_csv(TABLE/f'correlations_{method}.csv')
    # Event-accounting audit: complete observed PA totals, not model-valid-only totals.
    game_index=games.get_indexer(data.game_pk);actor=pd.Index(ids).get_indexer(data.batter_id);keep=actor>=0
    result=data.result
    raw=np.column_stack([np.ones(len(data)),result.isin({'single','double','triple','home_run'}),result.eq('home_run'),
       result.isin({'walk','intent_walk'}),result.eq('hit_by_pitch'),result.isin({'sac_fly','sac_fly_double_play'}),
       ~result.isin({'walk','intent_walk','hit_by_pitch','sac_fly','sac_fly_double_play','sac_bunt','catcher_interf'})]).astype(float)
    batting=np.zeros((len(games),P,7));np.add.at(batting,(game_index[keep],actor[keep]),raw[keep])
    statnames=['PA','H','HR','BB','HBP','SF','AB']
    event_batting=batting.copy();batting=np.zeros_like(batting)
    for gi,game_pk in enumerate(games):
        source=ROOT/f'data/processed/season_2025/games/{game_pk}_official_stats.json'
        for row in json.loads(source.read_text()):
            pi=pd.Index(ids).get_indexer([row['player_id']])[0]
            if pi>=0:batting[gi,pi]+=np.array([row[name] for name in statnames])
    mismatches=[]
    for gi,pi,si in np.argwhere(event_batting!=batting):
        mismatches.append({'game_pk':int(games[gi]),'player_id':int(ids[pi]),'field':statnames[si],
          'event_label_derived':float(event_batting[gi,pi,si]),'official_game_box':float(batting[gi,pi,si])})
    pd.DataFrame(mismatches).to_csv(TABLE/'event_label_vs_official_batting_discrepancies.csv',index=False)
    assert np.array_equal(batting.sum(0),official[statnames].to_numpy()),'Cached game box totals differ from official season totals'
    bt=(weights@batting.reshape(len(games),-1)).reshape(500,P,7)
    def rates(x):
        pa,h,hr,bb,hbp,sf,ab=np.moveaxis(x,-1,0)
        numerator=h-hr+bb+hbp
        return {'HR_per_PA':hr/pa,'OBP':(h+bb+hbp)/(ab+bb+hbp+sf),'BB_per_PA':bb/pa,
          'nonHR_onbase_per_PA':numerator/pa,'nonHR_onbase_per_nonHR_PA':numerator/(pa-hr),
          'nonHR_OBP_conditional':numerator/(ab-hr+bb+hbp+sf)}
    pr=rates(batting.sum(0));br=rates(bt)
    for k,v in pr.items():players[k]=v
    for k,v in zip(['EOWS_100','DI','CR'],base):br[k]=v
    group_rows=[]
    for group in ['Power','On-Base','Hybrid','Other']:
        mask=players.Group.eq(group).to_numpy()
        for k in ['EOWS_100','DI','CR',*pr]:
            a=players[k].to_numpy()[mask];b=br[k][:,mask].mean(1)
            group_rows.append({'Group':group,'metric':k,'n':int(mask.sum()),'mean':float(a.mean()),'median':float(np.median(a)),
                'resampling_low':float(np.quantile(b,.025)),'resampling_high':float(np.quantile(b,.975))})
    pd.DataFrame(group_rows).to_csv(TABLE/'group_comparison.csv',index=False)
    A=players.Group.eq('Power').to_numpy();B=players.Group.eq('On-Base').to_numpy()
    residual=residualize(players.CR.to_numpy()[None,:],players.OBP.to_numpy())[0]
    res_draw=residualize(base[2],players.OBP.to_numpy())
    # This conditional OBP adjustment holds observed season OBP fixed, explicitly.
    diff=[]
    for name,y,b in [('raw_CR',players.CR.to_numpy(),base[2]),('CR_residual_after_linear_OBP',residual,res_draw)]:
        d=b[:,A].mean(1)-b[:,B].mean(1)
        diff.append({'comparison':name,'Power_minus_OnBase':float(y[A].mean()-y[B].mean()),
         'resampling_low':float(np.quantile(d,.025)),'resampling_high':float(np.quantile(d,.975))})
    pd.DataFrame(diff).to_csv(TABLE/'CR_OBP_adjusted_group_difference.csv',index=False)
    players['CR_OBP_residual_pp']=residual
    players.to_csv(TABLE/'player_onbase_diagnostics.csv',index=False)
    ev=data.loc[data.metric_eligible].copy();ev['positive_WPA']=ev.estimated_batter_wpa>0;ev['onbase_event']=ev.result.isin(ONBASE)
    pd.crosstab(ev.onbase_event,ev.positive_WPA).to_csv(TABLE/'positive_WPA_onbase_contingency.csv')
    ev.groupby('result').agg(PA=('batter_id','size'),positive=('positive_WPA','sum'),positive_rate=('positive_WPA','mean')).to_csv(TABLE/'positive_WPA_by_event.csv')
    qualified_events=ev.loc[ev.batter_id.isin(ids)]
    pd.crosstab(qualified_events.result,qualified_events.positive_WPA).to_csv(TABLE/'qualified_positive_WPA_by_event.csv')
    pd.crosstab(qualified_events.onbase_event,qualified_events.positive_WPA).to_csv(TABLE/'qualified_positive_WPA_onbase_contingency.csv')
    # No fitting of proposed metrics: diagnostic weight choices are separate scenarios.
    wt=[];point_rank=players.Rank.to_numpy()
    for name,w in [('baseline',(.4,.3,.3)),('50_25_25',(.5,.25,.25)),('34_33_33',(.34,.33,.33)),('omit_CR_70_30_0',(.7,.3,0)),('omit_DI_70_0_30',(.7,0,.3))]:
        sc,_=composite(point,w);rr=rankdata(-sc,method='min')
        wt.append({'scenario':name,'rank_spearman':spearmanr(rr,point_rank).statistic,
          'mean_absolute_rank_shift':float(np.mean(abs(rr-point_rank))),
          'max_rank_shift':float(max(abs(rr-point_rank))),'top10_overlap':int(np.sum((rr<=10)&(point_rank<=10)))})
    pd.DataFrame(wt).to_csv(TABLE/'alternative_weight_diagnostics.csv',index=False)
    # Transparent individual-player profiles and extreme-event selection.
    selected=players.loc[players.player_name.isin(LEADERS)].copy()
    selected['weighted_E']=.4*selected.E;selected['weighted_D']=.3*selected.D;selected['weighted_C']=.3*selected.C
    profiles=[];game_examples=[];pa_examples=[]
    for x in selected.itertuples():
        e=ev.loc[ev.batter_id.eq(x.player_id)].copy();high=e.loc[e.estimated_pre_pa_li>=2]
        game=e.groupby(['game_pk','date']).estimated_batter_wpa.agg(['sum','size']).reset_index()
        pos=e.loc[e.estimated_batter_wpa>0].estimated_batter_wpa
        profiles.append({'player_id':x.player_id,'player_name':x.player_name,'high_PA':len(high),
           'high_WPA':high.estimated_batter_wpa.sum(),'high_mean_WPA':high.estimated_batter_wpa.mean(),
           'high_positive_rate':(high.estimated_batter_wpa>0).mean(),'positive_PA':len(pos),
           'positive_games_fraction':float((game['sum']>0).mean()),
           'largest5_positive_PA_share_of_positive_WPA':float(pos.nlargest(5).sum()/pos.sum())})
        for label,frame in [('largest_positive',game.nlargest(3,'sum')),('largest_negative',game.nsmallest(2,'sum'))]:
            for row in frame.to_dict('records'):game_examples.append({'player_name':x.player_name,'selection':label,**row})
        for label,frame in [('largest_positive',e.nlargest(5,'estimated_batter_wpa')),('largest_negative',e.nsmallest(3,'estimated_batter_wpa'))]:
            for row in frame.to_dict('records'):pa_examples.append({'player_name':x.player_name,'selection':label,**row})
    selected.merge(pd.DataFrame(profiles),on=['player_id','player_name'],validate='one_to_one').to_csv(TABLE/'leading_player_contributions.csv',index=False)
    pd.DataFrame(game_examples).to_csv(TABLE/'leading_player_important_games.csv',index=False)
    pd.DataFrame(pa_examples).to_csv(TABLE/'leading_player_important_PAs.csv',index=False)
    # Audit duplicated cluster weights against literal repetitions of observed rows.
    checkgames=games[:2];sample=ev.loc[ev.game_pk.isin(checkgames)]
    repeated=pd.concat([sample.loc[sample.game_pk.eq(checkgames[0])]]*2+[sample.loc[sample.game_pk.eq(checkgames[1])]])
    direct=repeated.groupby('batter_id').estimated_batter_wpa.sum().reindex(ids,fill_value=0)
    assert np.allclose(direct,2*ps[0].sum(1)+ps[1].sum(1))
    x=players.OBP.to_numpy();y=players.CR.to_numpy();coef=np.linalg.lstsq(np.column_stack([np.ones(P),x]),y,rcond=None)[0]
    r2=1-np.sum(residual**2)/np.sum((y-y.mean())**2)
    summary={'replicates':500,'empty_category_replicas':0,'baseline_reproduced':True,'duplicated_game_audit_passed':True,
       'all_official_batting_totals_reconciled':True,'cohort':P,'game_count':len(games),
       'CR_OBP_pearson':float(np.corrcoef(x,y)[0,1]),'CR_OBP_spearman':float(spearmanr(x,y).statistic),
       'linear_CR_on_OBP_R2':float(r2),'CR_OBP_residual_sd_pp':float(residual.std(ddof=1)),
       'CR_OBP_intercept':float(coef[0]),'CR_OBP_slope':float(coef[1]),
       'residual_CR_spearman_with_DI':float(spearmanr(residual,players.DI).statistic),
       'rank_width_spearman_with_High_PA':float(spearmanr(width,players.High_PA).statistic),
       'median_EOWS_bootstrap_SD':float(diag.EOWS_100_bootstrap_sd.median()),
       'median_DI_bootstrap_SD':float(diag.DI_bootstrap_sd.median()),
       'median_adjacent_EOWS_gap':float(np.median(np.diff(np.sort(players.EOWS_100)))),
       'median_adjacent_composite_gap':float(np.median(np.diff(np.sort(players.EOWS_C)))),
       'DI_bootstrap_SD_spearman_with_High_PA':float(spearmanr(diag.DI_bootstrap_sd,players.High_PA).statistic),
       'linear_adjustment':'CR on observed fixed season OBP; descriptive only; no causal or predictive claim',
       'scope':'No model retraining, downloads, revised original metric or calibrated confidence interval'}
    (REPORT/'results.json').write_text(json.dumps(summary,indent=2)+'\n')
    plots(players,pd.DataFrame(rows),pd.DataFrame(group_rows),selected)
    print(json.dumps(summary,indent=2));print(pd.DataFrame(rows).to_string(index=False));print(pd.DataFrame(diff).to_string(index=False))


def plots(players,ablations,groups,leaders):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    plt.rcParams.update({'figure.dpi':130,'font.size':10,'axes.spines.top':False,'axes.spines.right':False})
    corr=players[METRICS].corr(method='spearman')
    fig,ax=plt.subplots(figsize=(9,7));im=ax.imshow(corr,vmin=-1,vmax=1,cmap='RdBu_r')
    ax.set_xticks(range(7),METRICS,rotation=40,ha='right');ax.set_yticks(range(7),METRICS)
    for i in range(7):
        for j in range(7):ax.text(j,i,f'{corr.iloc[i,j]:.2f}',ha='center',va='center',fontsize=9)
    fig.colorbar(im,ax=ax,label='Spearman correlation');ax.set_title('Same 215 hitters: independent estimates, not new validation')
    fig.tight_layout();fig.savefig(CHART/'component_correlations.png',dpi=200);plt.close(fig)
    colors={'Power':'#d78b32','On-Base':'#147b88','Hybrid':'#785497','Other':'#9aabb3'}
    fig,axes=plt.subplots(1,2,figsize=(12,4.5))
    for group,c in colors.items():
        q=players.loc[players.Group.eq(group)]
        axes[0].scatter(q.OBP,q.CR,c=c,s=16,label=group,alpha=.75)
        axes[1].scatter(q.OBP,q.CR_OBP_residual_pp,c=c,s=16,alpha=.75)
    axes[0].legend();axes[0].set(xlabel='Official OBP',ylabel='CR (%)',title='Positive-WPA frequency largely tracks on-base frequency')
    axes[1].axhline(0,color='gray',ls=':');axes[1].set(xlabel='Official OBP',ylabel='CR minus descriptive linear OBP fit (pp)',title='Residual differences are small; no causal interpretation')
    fig.tight_layout();fig.savefig(CHART/'CR_OBP_redundancy.png',dpi=200);plt.close(fig)
    fig,ax=plt.subplots(figsize=(11,5));names=ablations.scenario.str.replace('_',' ')
    ax.barh(names,ablations.median_rank_range_width,color='#147b88');ax.invert_yaxis()
    ax.set(xlabel='Median width of 2.5–97.5% rank resampling range',title='Mechanism ablations: same 500 whole-game draws')
    fig.supxlabel('Diagnostic interventions are not additive variance partitions or validated alternative metrics.',fontsize=9)
    fig.tight_layout();fig.savefig(CHART/'bootstrap_instability_mechanisms.png',dpi=200);plt.close(fig)
    leaders=leaders.set_index('player_name').loc[LEADERS]
    fig,ax=plt.subplots(figsize=(10,4.5));left=np.zeros(5)
    for component,w,c in [('E',.4,'#147b88'),('D',.3,'#d78b32'),('C',.3,'#785497')]:
        val=leaders[component]*w;ax.barh(LEADERS,val,left=left,label=f'{component}: {w:.0%}',color=c);left+=val
    ax.invert_yaxis();ax.set(xlim=(0,100),xlabel='Weighted percentile points',title='Original EOWS-C v1.0: score decomposition, not a reliable exact order')
    ax.legend(loc='lower right');fig.tight_layout();fig.savefig(CHART/'leading_player_score_contributions.png',dpi=200);plt.close(fig)
    metrics=['EOWS_100','DI','CR','HR_per_PA','OBP','BB_per_PA','nonHR_onbase_per_PA','nonHR_OBP_conditional']
    fig,axes=plt.subplots(2,4,figsize=(15,8))
    for ax,k in zip(axes.flat,metrics):
        q=groups.loc[groups.metric.eq(k)&groups.Group.ne('Other')].set_index('Group').loc[['Power','On-Base','Hybrid']]
        for i,(name,row) in enumerate(q.iterrows()):
            ax.plot([i,i],[row.resampling_low,row.resampling_high],c=colors[name]);ax.scatter(i,row['mean'],color=colors[name],s=24)
        ax.set_xticks(range(3),['Power','On-Base','Hybrid'],rotation=25);ax.set_title(k)
    fig.suptitle('Fixed groups: player-mean descriptors and conditional game-resampling ranges')
    fig.supxlabel('Original model fixed; percentile ranges are not verified true-parameter confidence intervals.',fontsize=10)
    fig.tight_layout();fig.savefig(CHART/'group_comparison_extended.png',dpi=200);plt.close(fig)


if __name__=='__main__':run()
