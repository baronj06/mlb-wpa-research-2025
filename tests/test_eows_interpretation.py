"""Synthetic cluster/mapping invariants and saved real-output interpretation checks."""
import json
import numpy as np
import pandas as pd
from mlb_wpa.eows_interpretation import (fixed_percentile_mapping,residualize,
    game_tensors,component_draws,TABLE,REPORT)
from mlb_wpa.eows_metrics import percentile_scores


def test_fixed_percentile_mapping_preserves_ties_and_endpoint_policy():
    reference=np.array([1.,1.,3.])
    assert np.allclose(fixed_percentile_mapping(reference,reference),[25,25,100])
    assert np.allclose(fixed_percentile_mapping(np.array([-1.,4.]),reference),[0,100])
    assert np.allclose(fixed_percentile_mapping(np.array([0.,2.]),[1.,1.]),50)


def test_residual_adjustment_is_orthogonal_for_every_draw():
    x=np.array([.25,.3,.34,.4]);y=np.array([[1.,2.,5.,4.],[7.,4.,2.,9.]])
    r=residualize(y,x)
    assert np.allclose(r.sum(1),0,atol=1e-12)
    assert np.allclose(r@x,0,atol=1e-12)


def test_repeated_whole_game_weights_preserve_shared_player_observations():
    # Explicitly synthetic: two players share game1; invalid event is not promoted.
    d=pd.DataFrame({'game_pk':[1,1,2,2],'batter_id':[10,20,10,20],
      'metric_eligible':[True,True,True,False],
      'estimated_pre_pa_li':[.5,2.,1.,np.nan],
      'estimated_batter_wpa':[.1,-.2,.3,np.nan]})
    games,*features=game_tensors(d,[20,10])
    out=component_draws(np.array([[2.,1.]]),*features)
    _,_,counts,sums,positive=out
    assert counts.sum()==5
    assert np.allclose(sums.sum(2),[[-.4,.5]])
    assert positive.sum()==3
    assert list(games)==[1,2]


def test_league_DI_benchmark_cannot_change_component_percentile_order():
    means=np.array([-.02,.01,.01,.05])
    assert np.array_equal(percentile_scores(100*(means-.004)),percentile_scores(100*(means+.007)))


def test_frozen_bootstrap_reproduction_and_no_empty_categories():
    r=json.loads((REPORT/'results.json').read_text())
    assert r['baseline_reproduced'] and r['duplicated_game_audit_passed']
    assert r['replicates']==500 and r['empty_category_replicas']==0
    assert r['all_official_batting_totals_reconciled']


def test_official_box_stat_exception_is_visible_not_silently_imputed():
    d=pd.read_csv(TABLE/'event_label_vs_official_batting_discrepancies.csv')
    assert len(d)==1
    x=d.iloc[0]
    assert (x.game_pk,x.player_id,x.field,x.event_label_derived,x.official_game_box)==(776474,608070,'AB',4,3)


def test_leader_decomposition_preserves_original_weighted_score():
    d=pd.read_csv(TABLE/'leading_player_contributions.csv')
    assert len(d)==5
    assert np.allclose(d.weighted_E+d.weighted_D+d.weighted_C,d.EOWS_C)
    assert (d.high_PA>=20).all()
