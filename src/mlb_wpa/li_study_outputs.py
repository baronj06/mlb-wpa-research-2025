"""Report-only study outputs. Does not fit, select or tune any model."""

import hashlib
import json
import os

import numpy as np
import pandas as pd

from .collection import atomic_json, utc_now
from .li_accuracy_study import OUT, categories


def run():
    os.environ.setdefault('MPLCONFIGDIR',str(OUT.parents[1]/'.cache/matplotlib'))
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    report=json.loads((OUT/'results.json').read_text());plan=json.loads((OUT/'design.json').read_text())
    models=['baseline','markov_global','selected_context'];titles=['Frozen baseline','Markov WE + global LI','Selected WE + hierarchical LI']
    fig,axes=plt.subplots(1,3,figsize=(13,4.2),layout='constrained')
    for ax,name,title in zip(axes,models,titles):
        matrix=pd.read_csv(OUT/f'{name}_confusion.csv',index_col=0)
        row_percent=matrix.iloc[:,:3].div(matrix.sum(axis=1),axis=0).to_numpy()*100
        ax.imshow(row_percent,cmap='Blues',vmin=0,vmax=100)
        for i in range(3):
            for j in range(3):
                ax.text(j,i,f'{matrix.iloc[i,j]:,}\n{row_percent[i,j]:.1f}%',ha='center',va='center',
                        color='white' if row_percent[i,j]>55 else 'black',fontsize=10)
        ax.set(xticks=range(3),xticklabels=['Low','Medium','High'],yticks=range(3),
               yticklabels=['Low','Medium','High'],xlabel='Predicted category',ylabel='Reference category',title=title)
    fig.suptitle('Same 65 held-out reference games — counts and row percentages')
    fig.supxlabel('Fixed LI cutoffs: 0.85 and 2.0. All 4,765 reference-valid PAs; 35 reference-unresolved PAs reported separately.',fontsize=8)
    fig.savefig(OUT/'confusion_matrices.png',dpi=220);plt.close(fig)
    fig,axes=plt.subplots(1,2,figsize=(11,4.5),layout='constrained')
    for name,title,color in [('baseline','Baseline','#8e8e8e'),('markov','Markov','#146c83')]:
        bins=pd.read_csv(OUT/f'{name}_test_calibration.csv')
        axes[0].plot(bins.mean_prediction,bins.home_win_fraction,'o-',label=title,color=color)
    axes[0].plot([0,1],[0,1],'--',color='black',linewidth=.8)
    axes[0].set(xlabel='Mean predicted home WE',ylabel='Observed home-win frequency',xlim=(0,1),ylim=(0,1),title='Reference-test calibration (65 games)')
    axes[0].legend()
    x=np.arange(3);r=[report['test_results'][n] for n in models]
    axes[1].bar(x-.18,[100*t['high_precision'] for t in r],width=.36,label='High precision',color='#d38b35')
    axes[1].bar(x+.18,[100*t['high_recall'] for t in r],width=.36,label='High recall',color='#146c83')
    axes[1].set(xticks=x,xticklabels=['Baseline','Markov/global','Markov/context'],ylabel='Percent',ylim=(0,100),title='Recall improves without a precision collapse')
    axes[1].legend(loc='lower right');fig.savefig(OUT/'calibration_and_high_li.png',dpi=220);plt.close(fig)
    breakdown=pd.read_csv(OUT/'error_breakdowns.csv')
    fig,axes=plt.subplots(1,3,figsize=(13,4.2),layout='constrained')
    for ax,dim,levels,title in zip(axes,['inning_band','score_band','high_li_band'],
        [['1-6','7-9','10+'],['Tied','1 run','2 runs','3+ runs'],['2.0-2.2','2.2-3.0','3.0+']],
        ['Inning','Absolute score margin','Reference high-LI magnitude']):
        for name,color,offset in [('baseline','#8e8e8e',-.18),('markov_global','#146c83',.18)]:
            q=breakdown.loc[breakdown.model.eq(name)&breakdown.dimension.eq(dim)].set_index('level').reindex(levels)
            ax.bar(np.arange(len(levels))+offset,100*q.high_miss_rate,width=.36,color=color,label=name)
        ax.set(xticks=range(len(levels)),xticklabels=levels,ylabel='Missed reference High PAs (%)',ylim=(0,100),title=title)
    axes[0].legend();fig.suptitle('Where the baseline missed High-LI opportunities')
    fig.supxlabel('Rates use reference High PAs within each stratum; small extra-inning counts limit conclusions.',fontsize=8)
    fig.savefig(OUT/'high_li_error_breakdown.png',dpi=220);plt.close(fig)
    diagnostic={};baseline_fn=None
    for name in models:
        q=pd.read_csv(OUT/f'{name}_test_predictions.csv');q=q.loc[q.reference_validated]
        wrong=categories(q.predicted_li)!=categories(q.reference_li)
        high=q.reference_li.ge(2);miss=high&~q.predicted_li.ge(2)
        diagnostic[name]={'all_category_errors':int(wrong.sum()),'high_false_positives':int((~high&q.predicted_li.ge(2)).sum()),
            'high_false_positive_rate':float((~high&q.predicted_li.ge(2)).sum()/(~high).sum()),
            'category_errors_near_085':int((wrong&(abs(q.reference_li-.85)<=.1)).sum()),
            'category_errors_near_2':int((wrong&(abs(q.reference_li-2)<=.2)).sum()),
            'extra_inning_reference_games':int(q.loc[q.inning.ge(10)].game_pk.nunique()),
            'extra_inning_reference_pa':int(q.inning.ge(10).sum()),
            'high_misses':int(miss.sum())}
    goals={}
    for name,r in report['test_results'].items():
        goals[name]={'li_mae_below_020':r['li']['mae']<.2,'accuracy_above_085':r['accuracy']>.85,
                     'high_miss_below_015':r['high_false_negative_rate']<.15,
                     'wpa_maintained_or_improved':r['wpa']['mae']<=plan['target_goals']['wpa_mae_at_most']}
    atomic_json(OUT/'goal_assessment.json',{'generated_at_utc':utc_now(),'user_goals':goals,
        'candidate_for_followup_eows_sensitivity':all(goals['markov_global'].values()),
        'prior_we_ece_005_criterion_met':report['test_we_calibration']['markov']['ece']<=.05,
        'ranking_release_allowed':False,'ranking_sensitivity_evaluated':False,
        'limitations':['Test references previously inspected, not a new blinded confirmation',
            'WE test ECE remains above the prior 5% operational criterion',
            '35 reference attribution/state cases unresolved','No player-rank or cross-season sensitivity assessed'],
        'error_diagnostics':diagnostic})
    pd.DataFrame(report['test_we_calibration']).T.to_csv(OUT/'we_calibration_comparison.csv')
    p=json.loads((OUT/'provenance.json').read_text())
    p['report_outputs_sha256']={str(x.relative_to(OUT)):hashlib.sha256(x.read_bytes()).hexdigest()
        for x in OUT.iterdir() if x.suffix in ('.png','.csv')}
    p['report_script_sha256']=hashlib.sha256(__import__('pathlib').Path(__file__).read_bytes()).hexdigest()
    atomic_json(OUT/'provenance.json',p)
    print(json.dumps({'goals':goals,'diagnostics':diagnostic},indent=2))


if __name__=='__main__':run()
