"""Offline integration checks against the collected real season, when present."""

import hashlib
import json

import pandas as pd
import pytest

from mlb_wpa.phase3_finalize import validate_primary_table
from mlb_wpa.phase3_quality import require_metrics_permission
from mlb_wpa.sample import ROOT


@pytest.fixture(scope='module')
def snapshot():
    report=ROOT/'reports/phase3_data_quality.json'
    if not report.exists():pytest.skip('Full-season snapshot has not been generated')
    return (json.loads(report.read_text()),
            pd.read_parquet(ROOT/'data/processed/season_2025_cleaned_pas.parquet'))


def test_real_season_pa_ownership_and_exclusions(snapshot):
    report,data=snapshot
    validate_primary_table(data)
    assert len(data)==report['official_season_pa']==report['represented_pa']
    assert data.game_pk.nunique()==report['expected_games']
    official=pd.read_csv(ROOT/'data/processed/season_2025_official_player_stats.csv').set_index('player_id')
    actual=data.groupby('batter_id').size()
    assert actual.reindex(official.index,fill_value=0).eq(official.PA).all()
    assert actual.index.isin(official.index).all()
    assert report['official_player_stat_mismatches']==0
    assert report['player_pa_identity_count_mismatches']==0
    assert int(data.validated.sum())==report['validated_pa']
    assert data.loc[data.validated,'state_derivation_error'].isna().all()
    if not report['quality_gate']['metrics_and_rankings_allowed']:
        with pytest.raises(RuntimeError):require_metrics_permission(report['quality_gate'])


def test_season_formula_and_output_provenance(snapshot):
    provenance=json.loads((ROOT/'reports/phase3_provenance.json').read_text())
    original=json.loads((ROOT/'reports/attribution_provenance.json').read_text())
    formula=hashlib.sha256((ROOT/'config/research_v1.json').read_bytes()).hexdigest()
    assert formula==original['formula_config_sha256']==provenance['research_formula_sha256']
    for name,expected in provenance['outputs_sha256'].items():
        assert hashlib.sha256((ROOT/name).read_bytes()).hexdigest()==expected
