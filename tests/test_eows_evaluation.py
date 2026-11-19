"""Original formula oracles, exclusions, frozen application and real season checks."""

import json

import numpy as np
import pandas as pd
import pytest
from scipy.stats import rankdata

from mlb_wpa.eows_metrics import (original_components, percentile_scores, composite,
    summarize_events, scoring_table, classify_hitters, leverage_codes)
from mlb_wpa.eows_bootstrap import bootstrap
from mlb_wpa.sample import ROOT

OUT=ROOT/'reports/eows_2025_exploratory'
TABLES=ROOT/'outputs/tables/eows_2025_exploratory'
DATA=ROOT/'data/processed/season_2025_independent_markov_v03.parquet'


def test_original_hypothetical_eows_example_and_di_cr_units():
    # Original hypothetical example, never a real MLB observation.
    e,d,c=original_components([[60,30,10]],[[.12,.12,.10]],[[30,15,5]],
                              [600,300,100],[1,1,.5])
    assert e[0]==pytest.approx(.34)
    assert d[0]==pytest.approx(.5)
    assert c[0]==pytest.approx(50)


def test_missing_category_means_are_undefined_not_zero():
    e,d,c=original_components([[2,1,0]],[[.02,.01,0]],[[1,1,0]],
                              [60,30,10],[0,0,0])
    assert np.isnan(e[0]) and np.isnan(d[0]) and np.isnan(c[0])


def test_strict_positive_cr_and_all_league_not_only_qualified_benchmarks():
    # Synthetic events: player999 is outside the comparison player population.
    ev=pd.DataFrame({'batter_id':[1,1,1,2,2,2,999],
        'estimated_pre_pa_li':[.2,1,2,.2,1,2,2],
        'estimated_batter_wpa':[0,.1,-.1,.2,-.2,.3,.7]})
    table,league=scoring_table(ev,[1,2]);p=table.set_index('player_id')
    assert league['league_category_pa']==[2,2,3]
    assert league['high_league_mean_wpa']==pytest.approx(.3)
    assert p.loc[1,'DI']==pytest.approx(-40)
    assert p.loc[1,'CR']==pytest.approx(100*2/7)  # WPA=0 is not positive.
    assert p.loc[2,'CR']==pytest.approx(100*5/7)


def test_original_midrank_percentiles_and_composite_population():
    assert np.array_equal(percentile_scores([1,1,3]),[25,25,100])
    assert np.array_equal(percentile_scores([3,3,3]),[50,50,50])
    with pytest.raises(ValueError):percentile_scores([1])
    with pytest.raises(ValueError):percentile_scores([1,np.nan])
    score,p=composite([[1,2,3],[3,1,2],[2,3,1]])
    assert np.allclose(score,.4*p[0]+.3*p[1]+.3*p[2])
    with pytest.raises(ValueError):composite([[1,2],[1,2],[1,2]],weights=[.4,.3,.4])


def test_exact_composite_ties_do_not_split_due_to_floating_arithmetic():
    q=pd.read_csv(TABLES/'preliminary_eows_c_rankings.csv');n=len(q)
    r=[rankdata(q[c],method='average') for c in ['EOWS_100','DI','CR']]
    exact=(40*(r[0]-1)+30*(r[1]-1)+30*(r[2]-1))/(n-1)
    assert np.allclose(q.EOWS_C,exact,atol=1e-12)
    assert np.array_equal(q.Rank,rankdata(-exact,method='min'))


def test_li_cutoffs_are_exact_and_invalid_li_is_rejected():
    assert np.array_equal(leverage_codes([0,.849999,.85,1.99999,2]),[0,0,1,1,2])
    with pytest.raises(ValueError):leverage_codes([np.nan])
    with pytest.raises(ValueError):leverage_codes([-.1])


def test_classification_is_from_official_pa400_pool_before_metric_filtering():
    stats=pd.read_csv(ROOT/'data/processed/season_2025_official_player_stats.csv')
    q,definition=classify_hitters(stats)
    assert len(q)==215 and q.PA.ge(400).all()
    h=q.HR_per_PA>=definition['HR_per_PA_q75'];o=q.OBP>=definition['OBP_q75']
    assert q.loc[h&o,'Group'].eq('Hybrid').all()
    assert q.loc[h&~o,'Group'].eq('Power').all()
    assert q.loc[~h&o,'Group'].eq('On-Base').all()
    assert q.loc[~h&~o,'Group'].eq('Other').all()


def test_frozen_models_load_without_calibrating_or_reestimating_normalizer(monkeypatch):
    from mlb_wpa.eows_apply import frozen_models
    from mlb_wpa.li_candidates import MarkovWE, PotentialLI
    def denied(*args,**kwargs):raise AssertionError('No fitting allowed')
    monkeypatch.setattr(MarkovWE,'calibrate',denied)
    monkeypatch.setattr(PotentialLI,'__init__',denied)
    model,primary,alternative=frozen_models()
    original=json.loads((ROOT/'reports/li_accuracy_study/results.json').read_text())
    assert primary.normalizer==original['li_normalizers_training_only']['markov_global']
    assert alternative.normalizer==original['li_normalizers_training_only']['selected_context']


def test_real_fullseason_coverage_ownership_and_quarantine():
    d=pd.read_parquet(DATA);report=json.loads((OUT/'application_quality.json').read_text())
    assert len(d)==182926 and d.game_pk.nunique()==2430
    assert not d.duplicated(['game_pk','at_bat_index']).any()
    assert int(d.metric_eligible.sum())==181408==report['metric_eligible_pa']
    bad=d.loc[~d.metric_eligible]
    assert len(bad)==1518 and bad[['estimated_batter_wpa','estimated_pre_pa_li']].isna().all().all()
    assert d.known_reference_unresolved.sum()==35
    assert not d.loc[d.known_reference_unresolved,'metric_eligible'].any()
    assert report['heldout_li_max_change']==report['heldout_wpa_max_change']==0
    assert not report['model_refitted'] and report['network_requests']==0


def test_real_component_recomputation_for_one_player_and_common_population():
    q=pd.read_csv(TABLES/'preliminary_eows_c_rankings.csv');d=pd.read_parquet(DATA)
    valid=d.loc[d.metric_eligible];point=q.iloc[0];player=valid.loc[valid.batter_id.eq(point.player_id)]
    c=leverage_codes(player.estimated_pre_pa_li);lc=leverage_codes(valid.estimated_pre_pa_li)
    p=np.bincount(lc,minlength=3)/len(valid)
    means=np.array([player.loc[c==k,'estimated_batter_wpa'].mean() for k in range(3)])
    assert point.EOWS_100==pytest.approx(100*(p@means))
    benchmark=valid.loc[lc==2,'estimated_batter_wpa'].mean()
    assert point.DI==pytest.approx(100*(means[2]-benchmark))
    positive=np.array([player.loc[c==k,'estimated_batter_wpa'].gt(0).mean() for k in range(3)])
    assert point.CR==pytest.approx(100*(p@positive))
    assert q.valid_pa_fraction.ge(.95).all() and q[['Low_PA','Medium_PA','High_PA']].min().min()>=20
    assert q.EOWS_C.between(0,100).all() and q.CR.between(0,100).all()
    for metric,percent in [('EOWS_100','E'),('DI','D'),('CR','C')]:
        assert np.allclose(q[percent],percentile_scores(q[metric]))


def test_sensitivity_keeps_wpa_fixed_when_only_li_or_weights_change():
    primary=pd.read_csv(TABLES/'preliminary_eows_c_rankings.csv').set_index('player_id')
    long=pd.read_csv(TABLES/'sensitivity_rankings_long.csv')
    for name,q in long.loc[long.scenario.ne('complete_games_only')].groupby('scenario'):
        assert len(q)==len(primary)
        assert np.allclose(q.Estimated_WPA,primary.loc[q.player_id,'Estimated_WPA'])
    summary=pd.read_csv(TABLES/'sensitivity_summary.csv')
    assert len(summary)==8 and summary.rank_spearman.between(-1,1).all()


def test_game_cluster_bootstrap_uses_fixed_cohort_without_missing_fills():
    # Synthetic toy season: every game contains both players and all categories.
    rows=[]
    for game in range(4):
        for player in [1,2]:
            for li in [.2,1.,2.]:
                rows.append({'game_pk':game,'batter_id':player,'metric_eligible':True,
                    'estimated_pre_pa_li':li,'estimated_batter_wpa':(game-1.5)*(.01 if player==1 else -.01)})
    players=pd.DataFrame({'player_id':[1,2],'Group':['Power','On-Base']})
    intervals,groups,report=bootstrap(pd.DataFrame(rows),players,reps=20,seed=1)
    assert report['requested_reps']==report['usable_reps']==20
    assert report['games_per_replica']==4
    assert intervals.Rank_ci_low.ge(1).all() and intervals.Rank_ci_high.le(2).all()
    assert set(groups.comparison)=={'Power','On-Base','Power minus On-Base'}


def test_real_uncertainty_is_reported_instead_of_certifying_exact_order():
    q=pd.read_csv(TABLES/'preliminary_eows_c_rankings.csv');r=json.loads((OUT/'research_results.json').read_text())
    assert r['bootstrap']['requested_reps']==500
    assert r['bootstrap']['usable_reps']+r['bootstrap']['replicas_rejected_for_any_empty_player_category']==500
    assert q.Rank_ci_low.le(q.Rank_ci_high).all()
    assert q.bootstrap_top10_probability.between(0,1).all()
    assert q.wide_rank_uncertainty.any()
    assert q.Ranking_status.str.contains('Exploratory').all()


def test_nonlinear_composite_bootstrap_bias_is_exposed_not_hidden():
    q=pd.read_csv(TABLES/'preliminary_eows_c_rankings.csv')
    assert np.allclose(q.bootstrap_EOWS_C_bias,q.bootstrap_mean_EOWS_C-q.EOWS_C)
    g=pd.read_csv(TABLES/'group_descriptive_comparison.csv')
    expected=g['mean'].lt(g.ci_low)|g['mean'].gt(g.ci_high)
    assert g.resampling_range_excludes_point.eq(expected).all()
    report=json.loads((OUT/'bootstrap_design_and_completion.json').read_text())
    assert 'not calibrated confidence' in report['interval_interpretation']
