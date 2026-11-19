import copy
import hashlib
import json

import pytest

from mlb_wpa import verify_sources
from mlb_wpa.sample import ROOT, normalize


def synthetic_play():
    """Synthetic fixture for validation logic; not an MLB observation."""
    return {
        "about": {"atBatIndex": 0, "halfInning": "top", "inning": 1, "isComplete": True},
        "atBatIndex": 0, "credits": [{"credit": "b_pa"}],
        "matchup": {"batter": {"id": 1}, "pitcher": {"id": 2}},
        "homeTeamWinProbability": 40, "awayTeamWinProbability": 60,
        "homeTeamWinProbabilityAdded": -10,
        "result": {"eventType": "synthetic_single"}, "runners": [],
    }


def test_synthetic_sign_units_and_missing_li():
    play = synthetic_play()
    row = normalize([play])[0]
    assert row["batting_team_event_wpa"] == pytest.approx(0.1)
    assert row["li"] is None
    assert row["batter_attribution_status"] == "unverified_do_not_use_for_metrics"
    play["about"]["halfInning"] = "bottom"
    assert normalize([play])[0]["batting_team_event_wpa"] == pytest.approx(-0.1)


def test_real_2024_response_hash():
    provenance = json.loads((ROOT / "data/raw/745444_provenance.json").read_text())
    assert hashlib.sha256((ROOT / "data/raw/745444_win_probability.json").read_bytes()).hexdigest() == provenance["sha256"]


def test_reviewed_2025_sample_and_runner_separation():
    rows, crosswalk, runner, report = verify_sources.audit()
    assert len(rows) == 74
    assert len(crosswalk) == 19
    assert len({r["pa_id"] for r in rows}) == 74
    assert runner[0]["gameSeq"] == "64"
    assert runner[0]["playerId"] == "12037"  # Runner Jon Berti, not batter Ian Happ.
    happ = next(r for r in rows if r["at_bat_index"] == 63)
    assert happ["batter_id"] == 664023
    assert happ["batter_attributed_wpa"] == pytest.approx(-0.06471)
    assert happ["home_wp_after_pa"] - happ["home_wp_before_pa"] == pytest.approx(-0.04491)
    assert float(runner[0]["wpa"]) == pytest.approx(0.0198)
    assert happ["li_before_pa"] == pytest.approx(2.304)
    assert happ["li_at_batter_event"] == pytest.approx(2.356)
    assert happ["li_source_game_seq"] == 64
    assert report["max_fangraphs_event_vs_box_wpa_abs_error"] < 1e-4
    assert report["season_scale_ready"] is False


def test_corrupted_event_identity_fails_closed(monkeypatch):
    original = verify_sources.fg_query

    def corrupted(filename, key):
        data = copy.deepcopy(original(filename, key))
        if key == "scores/play-log":
            data[0]["playerId"] = "19470"  # Synthetic corruption of cached evidence.
        return data

    monkeypatch.setattr(verify_sources, "fg_query", corrupted)
    with pytest.raises(ValueError):
        verify_sources.audit()


def test_wrong_pre_pa_state_fails_closed(monkeypatch):
    original = verify_sources.pre_pa_state

    def incorrect(plays, position):
        return (0, "___", 99, 99) if position == 63 else original(plays, position)

    monkeypatch.setattr(verify_sources, "pre_pa_state", incorrect)
    with pytest.raises(ValueError, match="Pre-PA"):
        verify_sources.audit()


def test_formula_contract_preserved():
    config = json.loads((ROOT / "config/research_v1.json").read_text())
    assert config["formulas"]["EOWS_100"] == "100 × Σ(p_k × w̄_k)"
    assert config["formulas"]["DI"] == "100 × (w̄_H,player - w̄_H,MLB)"
    assert config["formulas"]["CR"] == "100 × Σ[p_k × (N_k,+ / N_k)]"
    assert config["formulas"]["EOWS-C"] == "0.40E + 0.30D + 0.30C"
    assert config["baseline_weights"] == {"E": 0.4, "D": 0.3, "C": 0.3}
    assert config["leverage_thresholds"] == {"medium_min_inclusive": 0.85, "high_min_inclusive": 2.0}
    assert config["positive_wpa_rule"] == "WPA > 0"
    assert config["full_season_collection_enabled"] is False
