"""Markov mechanics, prospective potential LI and frozen evaluation checks."""

from collections import Counter
from dataclasses import replace
import json

import numpy as np
import pandas as pd
import pytest

from mlb_wpa.attribution import State
from mlb_wpa.independent_we import counterfactual_state
from mlb_wpa.li_candidates import MarkovWE, PotentialLI, KEYS
from mlb_wpa.li_accuracy_study import OUT, evaluate


def synthetic_state(key=(0,'___'),inning=1,half='top'):
    return State(inning,half,key[0],tuple(-(i+1) if b!='_' else None for i,b in enumerate(key[1])),0,0)


@pytest.fixture(scope='module')
def synthetic_chain():
    # Artificial out/HR process for manually verifiable mechanics only.
    counts={};rows=[]
    for key in KEYS:
        runs=sum(b!='_' for b in key[1])+1
        out=(0,1,'___' if key[0]==2 else key[1]);hr=(runs,0,'___')
        counts[key]=Counter({out:75,hr:25})
        for outcome,n in [(out,75),(hr,25)]:
            for _ in range(n):
                s=synthetic_state(key,1 if outcome==out else 7,'top' if outcome==out else 'bottom')
                rows.append({'game_pk':1,'supported':True,'start_state':s,'batter_before':s,
                    'after_state':counterfactual_state(s,outcome),'outcome':outcome})
    return MarkovWE(counts,{1}),counts,rows


def test_half_inning_pmf_matches_known_geometric_distribution(synthetic_chain):
    model,_,_=synthetic_chain
    pmf=model.run_pmf[(2,'___')]
    assert np.allclose(pmf[:5],[.75*.25**r for r in range(5)])
    assert sum(pmf)==pytest.approx(1,abs=1e-12)
    assert model.max_truncated_tail<1e-8


def test_markov_extra_and_walkoff_probabilities_follow_game_rules(synthetic_chain):
    m,_,_=synthetic_chain
    assert m.raw_probability(synthetic_state())==pytest.approx(.5,abs=1e-12)
    assert m.raw_probability(synthetic_state((0,'_2_'),10))==pytest.approx(.5,abs=1e-12)
    s=synthetic_state((0,'___'),9,'bottom')
    assert m.raw_probability(s)==pytest.approx(1-.5*m.run_pmf[(0,'___')][0])
    assert m.predict([replace(s,home_score=1)])[0]==1
    assert m.predict([replace(s,outs=3,away_score=1)])[0]==0


def test_raw_markov_we_is_coherent_with_outgoing_not_incoming_transitions(synthetic_chain):
    m,counts,_=synthetic_chain
    for key in [(0,'___'),(1,'1__'),(2,'123')]:
        for inning,half in [(1,'top'),(9,'bottom'),(10,'top')]:
            s=synthetic_state(key,inning,half)
            after=[counterfactual_state(s,o) for o in counts[key]]
            probability=np.array(list(counts[key].values()))/100
            assert probability@m.predict(after,False)==pytest.approx(m.predict([s],False)[0],abs=1e-10)


def test_hierarchical_probabilities_are_normalized_and_partially_pooled(synthetic_chain):
    m,parents,rows=synthetic_chain;li=PotentialLI(m,parents,rows,True,200)
    outcomes,p,info=li.probabilities(synthetic_state())
    assert p.sum()==pytest.approx(1) and np.all(p>=0)
    assert info['context_pa']==75
    assert p[outcomes.index((1,0,'___'))]==pytest.approx(50/275)
    outcomes,p,info=li.probabilities(synthetic_state(inning=10,half='bottom'))
    assert info['context_pa']==0 and info['hierarchical_weight']==0
    assert p[outcomes.index((1,0,'___'))]==pytest.approx(.25)
    # No evaluated PA result is an input to LI.
    assert li.predict(synthetic_state())[0]==li.predict(synthetic_state())[0]


def test_model_components_reject_leaked_donor_or_calibration_games(synthetic_chain):
    m,parents,rows=synthetic_chain
    leaked=[{**rows[0],'game_pk':999}]+rows[1:]
    with pytest.raises(ValueError,match='leakage'):PotentialLI(m,parents,leaked)
    with pytest.raises(ValueError,match='leakage'):m.calibrate(rows[:2])


def test_missing_prediction_counts_as_high_miss_not_silent_exclusion():
    # Synthetic adverse fixture; the last reference-invalid row is not gold.
    q=pd.DataFrame({'game_pk':[1,1,1,1],'reference_validated':[True,True,True,False],
        'reference_li':[.2,1.,2.5,3.],'predicted_li':[.3,1.1,np.nan,3.],
        'reference_wpa':[.01,.02,.03,np.nan],'predicted_wpa':[.02,.02,np.nan,.1]})
    r,c=evaluate(q)
    assert r['reference_pa']==3 and r['li_paired_pa']==2 and r['li_missing_pa']==1
    assert r['high_false_negative_rate']==1 and c.loc['High','Missing']==1
    assert r['accuracy']==pytest.approx(2/3)


def test_real_design_and_selection_keep_reference_games_out():
    p=json.loads((OUT/'design.json').read_text())
    groups=[set(p[k]) for k in ['training_games','calibration_games','selection_validation_games','reference_test_games']]
    assert [len(g) for g in groups]==[1000,200,300,65]
    assert not any(groups[i]&groups[j] for i in range(4) for j in range(i))
    lock=json.loads((OUT/'selection_lock.json').read_text())
    assert not lock['test_reference_values_used'] and not lock['hyperparameter_search']
    best=min(lock['validation_metrics'],key=lambda k:lock['validation_metrics'][k]['brier_game_balanced'])
    assert lock['selected_we']==best=='markov'


def test_real_comparison_uses_identical_reference_population_and_reports_uncertainty():
    r=json.loads((OUT/'results.json').read_text())
    assert r['reference_total_pa']==4800 and r['reference_unresolved_pa']==35
    for name,q in r['test_results'].items():
        assert q['reference_pa']==q['li_paired_pa']==4765
        assert q['wpa']['n']==4758 and q['high_reference_pa']==534
        c=pd.read_csv(OUT/f'{name}_confusion.csv',index_col=0)
        assert c.to_numpy().sum()==4765 and c['Missing'].sum()==0
        assert q['li']['mae_cluster_95ci'][0]<=q['li']['mae']<=q['li']['mae_cluster_95ci'][1]
    assert r['test_results']['baseline']['high_false_negatives']==191
    assert r['test_results']['markov_global']['high_false_negatives']==17
    assert r['test_results']['markov_global']['li']['mae']<.2
    assert not r['rankings_generated']


def test_research_goals_do_not_automatically_release_eows_rankings():
    p=json.loads((OUT/'goal_assessment.json').read_text())
    assert all(p['user_goals']['markov_global'].values())
    assert p['candidate_for_followup_eows_sensitivity']
    assert not p['ranking_release_allowed'] and not p['ranking_sensitivity_evaluated']
    assert not p['prior_we_ece_005_criterion_met']
