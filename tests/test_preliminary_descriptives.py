"""Single-expansion and supported descriptive-analysis safeguards."""

import hashlib
import json

import numpy as np
import pandas as pd
import pytest

from mlb_wpa.preliminary_descriptives import assess_model, batting_rates, attach_wpa_sources
from mlb_wpa.sample import ROOT
from mlb_wpa.we_poc import validate_split

OUT=ROOT/'reports/expanded_we_once'


def test_single_change_uses_more_cached_games_with_identical_estimator():
    old=json.loads((ROOT/'reports/independent_we_poc/design.json').read_text())
    new=json.loads((OUT/'design.json').read_text());sets=validate_split(new)
    assert [len(s) for s in sets]==[1000,200,65]
    assert set(old['test_reference_games'])==sets[2]
    assert set(old['calibration_games'])<=sets[1]
    for key in ['ridge_penalty','min_kernel_pa','min_kernel_games','bootstrap_replicates']:
        assert new[key]==old[key]
    original=(ROOT/'src/mlb_wpa/we_poc.py').read_text()
    expected=original.replace('Run the frozen 215-game offline independent WE/LI proof of concept.',
        'Frozen evaluation driver for the single cached training-size expansion.').replace(
        "OUT=ROOT/'reports/independent_we_poc'","OUT=ROOT/'reports/expanded_we_once'").replace(
        "'selected_games':215","'selected_games':plan['selected_games']")
    assert (ROOT/'src/mlb_wpa/we_expanded.py').read_text()==expected


def test_actual_gate_rejects_category_and_high_leverage_errors():
    r=json.loads((OUT/'results.json').read_text());p=json.loads((OUT/'acceptance_policy.json').read_text())
    confusion=pd.read_csv(OUT/'li_category_confusion.csv',index_col=0)
    gate=assess_model(r,p,confusion)
    assert not gate['metrics_and_rankings_allowed']
    assert {'li_mae_max','li_category_agreement_min','high_li_precision_min','high_li_recall_min'}<=set(gate['failed_checks'])
    assert gate['values']['high_li_precision']==pytest.approx(343/535)
    assert gate['values']['high_li_recall']==pytest.approx(343/534)
    assert gate['checks']['model_li_reference_coverage_min']


def test_obp_uses_exact_research_denominator_and_zero_is_missing():
    q=pd.DataFrame({'PA':[13,0],'AB':[8,0],'H':[3,0],'HR':[1,0],'BB':[2,0],'HBP':[1,0],'SF':[1,0]})
    rates=batting_rates(q)
    assert rates.OBP.iloc[0]==pytest.approx(6/12)
    assert rates.HR_per_PA.iloc[0]==pytest.approx(1/13)
    assert np.isnan(rates.OBP.iloc[1]) and np.isnan(rates.HR_per_PA.iloc[1])


def test_partial_wpa_is_not_extrapolated_or_replaced_by_api_diagnostic():
    # Explicit synthetic source/coverage fixture, not real MLB observations.
    stats=pd.DataFrame({'player_id':[1,2],'PA':[100,100]})
    season=pd.DataFrame({'batter_id':[1,2],'api_supplied_offense_wpa_unverified':[.5,.7]})
    reference=pd.DataFrame({'batter_id':[1],'validated':[True],'batter_attributed_wpa':[.1]})
    out=attach_wpa_sources(stats,season,reference).set_index('player_id')
    assert out.loc[1,'reference_sample_batter_wpa_sum']==pytest.approx(.1)
    assert out.full_season_reference_batter_wpa.isna().all()
    assert out.loc[2,'reference_sample_verified_pa']==0
    assert np.isnan(out.loc[2,'reference_sample_batter_wpa_sum'])


def test_real_official_descriptives_reconcile_to_existing_season():
    q=pd.read_csv(ROOT/'outputs/tables/preliminary_2025/players_pa400_hr_obp_wpa_sources.csv')
    full=pd.read_csv(ROOT/'outputs/tables/preliminary_2025/all_player_batting_and_wpa_sources.csv')
    original=pd.read_csv(ROOT/'data/processed/season_2025_official_player_stats.csv')
    assert len(q)==215 and q.player_id.is_unique and q.PA.ge(400).all()
    assert full.PA.sum()==original.PA.sum()==182926
    assert full.HR.sum()==original.HR.sum()
    expected=batting_rates(original).set_index('player_id').OBP
    assert np.allclose(q.set_index('player_id').OBP,expected.reindex(q.player_id),equal_nan=True)
    assert q.full_season_reference_batter_wpa.isna().all()
    report=json.loads((OUT/'descriptive_results.json').read_text())
    assert not report['eows_rankings_generated'] and not report['hitter_group_winning_comparisons_generated']


def test_previous_code_data_checkpoint_and_poc_artifacts_preserved():
    manifest=json.loads((OUT/'preserved_inputs.json').read_text())
    for name,expected in manifest.items():
        assert hashlib.sha256((ROOT/name).read_bytes()).hexdigest()==expected,name
