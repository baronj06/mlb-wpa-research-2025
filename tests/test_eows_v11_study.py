"""Synthetic variance/shrinkage invariants and real frozen-study safeguards."""
import json
import numpy as np
import pandas as pd
import pytest
from mlb_wpa.eows_v11_study import (cluster_mean_variance,fit_tau2,shrink_di,
    fit_adjustment,adjusted_frequency,gate_apf,framework_scores,OUT,TABLE)
from mlb_wpa.eows_metrics import composite
from mlb_wpa.sample import ROOT


def test_cluster_ratio_variance_matches_literal_duplicated_games():
    # Synthetic: game1 twice, game2 once; each game keeps every player.
    n=np.array([[2.,1.],[1.,2.]])
    s=np.array([[.2,.3],[-.1,.2]])
    m,v,N,G=cluster_mean_variance(np.array([[2.,1.]]),n,s)
    nn=np.repeat(n,[2,1],axis=0);ss=np.repeat(s,[2,1],axis=0)
    mean=ss.sum(0)/nn.sum(0)
    expected=3/2*np.sum((ss-mean*nn)**2,axis=0)/nn.sum(0)**2
    assert np.allclose(m,mean) and np.allclose(v,expected)
    assert np.allclose(N,nn.sum(0)) and np.allclose(G,3)


def test_missing_or_single_cluster_variance_remains_unknown():
    m,v,N,G=cluster_mean_variance(np.ones((1,2)),np.array([[0.,1.],[0.,0.]]),np.array([[0.,.2],[0.,0.]]))
    assert np.isnan(m[0,0]) and np.isnan(v).all()
    assert m[0,1]==.2 and G[0,1]==1


def test_working_normal_marginal_ML_has_expected_limits():
    valid=np.ones(40,dtype=bool)
    assert fit_tau2(np.zeros(40),np.ones(40),valid)==0
    assert fit_tau2(np.tile([-3.,3.],20),np.ones(40),valid)==pytest.approx(8,abs=1e-4)
    with pytest.raises(ValueError):fit_tau2(np.ones(4),np.ones(4),np.ones(4,dtype=bool))


def test_shrinkage_formula_missing_policy_and_vectorized_tau():
    shrunk,r=shrink_di(np.array([20.,5.,np.nan]),np.array([9.,1.,1.]),1.)
    assert np.allclose(shrunk[:2],[2.,2.5]) and np.allclose(r[:2],[.1,.5])
    assert np.isnan(shrunk[2]) and np.isnan(r[2])
    values,rel=shrink_di(np.ones((2,3)),np.ones((2,3)),np.array([[0.],[3.]]))
    assert np.allclose(values,[[0,0,0],[.75,.75,.75]])
    with pytest.raises(ValueError):shrink_di(np.ones(2),np.ones(2),-1)
    with pytest.raises(ValueError):shrink_di(np.ones(2),np.ones(2),np.nan)


def test_opportunity_adjustment_uses_saved_coefficients_without_eval_fit():
    obp=np.array([.2,.3,.4,.25,.35]);opps=np.array([2.,3.,4.,4.,2.])
    y=2+90*obp+.5*opps
    beta=fit_adjustment(y,obp,opps,np.array([100,200,50,150,80]))
    assert np.allclose(beta,[2,90,.5])
    assert np.allclose(adjusted_frequency(y,obp,opps,beta),0)
    assert np.allclose(adjusted_frequency(y+1,obp,opps,beta),1)


def test_consistency_gate_cannot_be_passed_by_low_redundancy_alone():
    assert not gate_apf(.069,214,[0,.1,.2])
    assert not gate_apf(.4,50,[0,.1,.2])
    assert not gate_apf(.4,200,[.95,.1,.2])
    assert gate_apf(.4,200,[.2,.1,.2])


def test_failed_gate_produces_no_unjustified_D_and_keeps_original_A():
    z={'EOWS_100':np.array([1.,2.,3.]),'DI':np.array([2.,1.,3.]),'CR':np.array([2.,3.,1.])}
    out=framework_scores(z,z['DI'],np.ones(3),False)
    assert 'D_APF_EB_DI' not in out
    assert np.array_equal(out['A_v1_0'],composite([z[k] for k in ['EOWS_100','DI','CR']])[0])


def test_development_evaluation_and_WE_roles_do_not_overlap():
    design=json.loads((OUT/'design.json').read_text());lock=json.loads((OUT/'selection_lock.json').read_text())
    a=set(design['development_games']);b=set(design['evaluation_games'])
    assert len(a)==1565 and len(b)==865 and not a&b
    assert set(lock['fit_games'])==a and not lock['evaluation_games_used_for_fit']
    q=pd.read_parquet(ROOT/'data/processed/season_2025_independent_markov_v03.parquet')
    assert set(q.loc[q.model_role.eq('other_application'),'game_pk'])==b


def test_candidate_profiles_keep_missingness_and_original_components_separate():
    original=pd.read_csv(ROOT/'outputs/tables/eows_2025_exploratory/preliminary_eows_c_rankings.csv').set_index('player_id')
    new=pd.read_csv(TABLE/'proposed_v11_player_profiles.csv').set_index('player_id').loc[original.index]
    assert np.allclose(new.EOWS_100,original.EOWS_100)
    assert np.allclose(new.DI,original.DI)
    assert np.allclose(new.CR,original.CR)
    assert np.allclose(new.EB_DI,new.DI*new.EB_reliability)
    assert new.EB_reliability.between(0,1).all()
