"""Publication-style exploratory figures from saved real-data EOWS tables."""

import json
import os

import numpy as np
import pandas as pd

from .sample import ROOT
from .eows_research import TABLES

OUT=ROOT/'outputs/charts/eows_2025_exploratory'
COLORS={'Power':'#b95e30','On-Base':'#166d82','Hybrid':'#72539a','Other':'#929b9e'}


def run():
    os.environ.setdefault('MPLCONFIGDIR',str(ROOT/'.cache/matplotlib'))
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    plt.rcParams.update({'axes.spines.top':False,'axes.spines.right':False,'font.size':10})
    OUT.mkdir(parents=True,exist_ok=True)
    q=pd.read_csv(TABLES/'preliminary_eows_c_rankings.csv');top=q.head(20).iloc[::-1]
    fig,axes=plt.subplots(1,2,figsize=(15,8),layout='constrained');y=np.arange(len(top))
    axes[0].hlines(y,top.EOWS_C_ci_low,top.EOWS_C_ci_high,color='#aab5bb',linewidth=2)
    axes[0].scatter(top.EOWS_C,y,c=top.Group.map(COLORS),zorder=3,s=30)
    axes[0].set(xlim=(0,100),xlabel='EOWS-C (0–100)',yticks=y,
        yticklabels=[f'{int(r.Rank)}. {r.player_name}' for r in top.itertuples()],title='Point scores and 95% bootstrap resampling ranges')
    axes[1].hlines(y,top.Rank_ci_low,top.Rank_ci_high,color='#aab5bb',linewidth=2)
    axes[1].scatter(top.Rank,y,c=top.Group.map(COLORS),zorder=3,s=30)
    axes[1].set(xlim=(1,len(q)),xlabel='Rank within the fixed 215-player cohort',yticks=y,yticklabels=[],
        title='Exact ordering is uncertain even near the top')
    fig.suptitle('2025 exploratory EOWS-C — top 20 point estimates, not reliable exact ranks',fontsize=14)
    fig.supxlabel('Independent frozen Markov estimates; 500 game-cluster bootstrap replicates. Model/training uncertainty is not included.',fontsize=9)
    fig.savefig(OUT/'top20_scores_and_rank_uncertainty.png',dpi=220);plt.close(fig)
    fig,axes=plt.subplots(2,2,figsize=(12,9),layout='constrained')
    for ax,(x,xlabel,target,ylabel) in zip(axes.flat,[
        ('HR_per_PA','HR / official PA','Estimated_WPA','Estimated raw batter WPA (wins)'),
        ('OBP','Official OBP','Estimated_WPA','Estimated raw batter WPA (wins)'),
        ('HR_per_PA','HR / official PA','EOWS_100','EOWS_100 (WPA per standardized 100 PA)'),
        ('OBP','Official OBP','EOWS_100','EOWS_100 (WPA per standardized 100 PA)')]):
        for group,part in q.groupby('Group'):ax.scatter(part[x],part[target],s=24,color=COLORS[group],alpha=.7,label=group)
        for r in q.head(3).itertuples():ax.annotate(r.player_name,(getattr(r,x),getattr(r,target)),fontsize=7,xytext=(3,4),textcoords='offset points')
        ax.set(xlabel=xlabel,ylabel=ylabel)
    axes[0,0].legend(fontsize=8);fig.suptitle('Power and on-base output versus estimated winning contributions')
    fig.supxlabel('Observed 2025 descriptives, PA >=400; associations are not causal and OBP includes home runs.',fontsize=9)
    fig.savefig(OUT/'hr_obp_wpa_eows_scatterplots.png',dpi=220);plt.close(fig)
    groups=pd.read_csv(TABLES/'group_descriptive_comparison.csv');order=['Power','On-Base','Hybrid','Other']
    fig,axes=plt.subplots(2,2,figsize=(11,8),layout='constrained')
    labels={'EOWS_100':'EOWS_100 (WPA / standardized 100 PA)','DI':'DI (WPA / 100 High PA above league)',
            'CR':'CR (standardized positive-WPA %)','EOWS_C':'EOWS-C percentile composite'}
    for ax,metric in zip(axes.flat,['EOWS_100','DI','CR','EOWS_C']):
        part=groups.loc[groups.metric.eq(metric)].set_index('Group').reindex(order)
        x=np.arange(4);ax.vlines(x,part.ci_low,part.ci_high,color='#54636a',linewidth=2)
        ax.scatter(x,part['mean'],c=[COLORS[g] for g in order],s=50,zorder=3)
        ax.set(xticks=x,xticklabels=[f'{g}\n(n={int(part.loc[g,"n_players"])})' for g in order],ylabel=labels[metric])
        if metric in ('EOWS_100','DI'):ax.axhline(0,color='gray',linestyle='--',linewidth=.7)
    fig.suptitle('Group means and 95% game-bootstrap resampling ranges')
    fig.supxlabel('Fixed groups/model; composite percentile resampling can be biased. Ranges are not calibrated confidence bounds or causal effects.',fontsize=9)
    fig.savefig(OUT/'group_comparisons_with_uncertainty.png',dpi=220);plt.close(fig)
    sensitivity=pd.read_csv(TABLES/'sensitivity_summary.csv')
    fig,axes=plt.subplots(1,2,figsize=(13,5.5),layout='constrained')
    x=np.arange(len(sensitivity));axes[0].barh(x,sensitivity.mean_absolute_rank_shift,color='#166d82')
    axes[0].set(yticks=x,yticklabels=sensitivity.scenario,xlabel='Mean absolute rank shift',title='Scenario sensitivity')
    axes[1].barh(x,sensitivity.top10_overlap,color='#72539a')
    axes[1].set(yticks=x,yticklabels=[],xlim=(0,10),xlabel='Shared players with primary top 10',title='Top-10 overlap (out of 10)')
    fig.supxlabel('Complete-game stress retains 111,456 / 181,408 eligible PAs; its severe sample loss is not a correction for the 0.83% primary exclusions.',fontsize=8)
    fig.savefig(OUT/'ranking_sensitivity.png',dpi=220);plt.close(fig)
    top10=q.head(10).iloc[::-1];fig,ax=plt.subplots(figsize=(10,5),layout='constrained');left=np.zeros(len(top10))
    for key,weight,color in [('E',.4,'#166d82'),('D',.3,'#d18c36'),('C',.3,'#72539a')]:
        value=weight*top10[key];ax.barh(top10.player_name,value,left=left,label=f'{key}: {weight:.0%}',color=color);left+=value
    ax.set(xlim=(0,100),xlabel='Weighted component percentile points',title='EOWS-C components: top 10 point estimates')
    ax.legend(loc='lower right');fig.supxlabel('EOWS, DI and CR remain separate metrics. Shared EOWS-C score scale does not imply a probability of winning.',fontsize=8)
    fig.savefig(OUT/'top10_component_breakdown.png',dpi=220);plt.close(fig)
    raw_rank=q.Estimated_WPA.rank(ascending=False,method='min')
    fig,ax=plt.subplots(figsize=(7,6),layout='constrained')
    ax.scatter(raw_rank,q.Rank,c=q.Group.map(COLORS),s=25,alpha=.75)
    ax.plot([1,len(q)],[1,len(q)],'--',color='gray',linewidth=1)
    for idx in (raw_rank-q.Rank).abs().nlargest(4).index:
        ax.annotate(q.loc[idx,'player_name'],(raw_rank[idx],q.loc[idx,'Rank']),fontsize=8,xytext=(3,4),textcoords='offset points')
    ax.set(xlabel='Estimated raw WPA rank',ylabel='EOWS-C point rank',xlim=(len(q)+2,0),ylim=(len(q)+2,0),
           title='Different descriptions can change player ordering')
    fig.supxlabel('Both WPA and EOWS use independent model estimates. Rank differences are not evidence that EOWS-C is superior.',fontsize=8)
    fig.savefig(OUT/'estimated_wpa_vs_eows_c_rank.png',dpi=220);plt.close(fig)
    print('Saved',len(list(OUT.glob('*.png'))),'exploratory figures')


if __name__=='__main__':run()
