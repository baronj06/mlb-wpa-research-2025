"""New real-sample regression tests plus explicitly synthetic edge fixtures."""

from collections import Counter
from dataclasses import replace
import hashlib
import json

import pytest

from mlb_wpa.attribution import State, Unit, build_ledger
from mlb_wpa.collection import CachedClient
from mlb_wpa.final_validation import match_units_by_state, validate_one, opportunity_state
from mlb_wpa.sample import ROOT
from mlb_wpa.investigate_missing import investigate


@pytest.fixture(scope="module")
def real_results():
    manifest = json.loads((ROOT / "config/validation_2025_games.json").read_text())
    checkpoint = json.loads((ROOT / "data/checkpoints/validation_2025.json").read_text())
    client = CachedClient(offline=True)
    results = [validate_one(g, checkpoint["games"][str(g["game_pk"])], client) for g in manifest["games"]]
    assert client.network_requests == 0
    return results


@pytest.mark.parametrize("position,official,validated", [(0,64,64),(1,83,83),(2,64,64),(3,77,77),(4,81,79),
                                                        (5,66,66),(6,69,67),(7,60,60),(8,73,73),(9,78,78)])
def test_ten_real_games_have_conservative_official_denominators(real_results, position, official, validated):
    rows, events, report = real_results[position]
    assert len(rows) == report["official_pa"] == official
    assert sum(r["validated"] for r in rows) == report["validated_pa"] == validated
    assert len({r["at_bat_index"] for r in rows}) == official
    assert report["reference_box_accounting"]["max_abs_difference"] < 1e-4
    assert not report["unexplained_box_actor_ids"]


def test_real_aggregate_coverage_and_unresolved_values(real_results):
    rows = [r for result in real_results for r in result[0]]
    assert len(rows) == 715 and sum(r["validated"] for r in rows) == 711
    unresolved = [r for r in rows if not r["validated"]]
    assert len(unresolved) == 4
    assert all(r["batter_attributed_wpa"] is None and r["pre_pa_li"] is None for r in unresolved)
    assert Counter(r["reason"] for r in unresolved) == {
        "multi_runner_or_missing_credit_requires_review": 1,
        "runner_action_pa_boundary_unverified": 1,
        "simultaneous_terminal_runner_and_batter_event": 2}


def test_real_automatic_runner_and_runner_only_outs(real_results):
    assert real_results[1][2]["automatic_runner_pa_count"] == 2
    assert real_results[7][2]["runner_only_plays"][0]["event"] == "caught_stealing_2b"
    assert real_results[7][2]["runner_only_plays"][0]["matched"] is True
    assert real_results[9][2]["runner_only_plays"][0]["event"] == "pickoff_1b"
    assert real_results[9][2]["runner_only_plays"][0]["matched"] is True


def test_real_combined_steal_error_state_and_balk_before_first_pitch(real_results):
    first_rows, first_events, _ = real_results[0]
    pa = next(r for r in first_rows if r["at_bat_index"] == 44)
    events = [e for e in first_events if e["at_bat_index"] == 44]
    assert pa["validated"] and pa["independent_event_count"] == 1
    assert [e["bases"] for e in events] == ["1__", "__3"]
    balk_pa = next(r for r in real_results[1][0] if r["at_bat_index"] == 38)
    assert balk_pa["validated"] and balk_pa["independent_event_count"] == 2


def synthetic_unit_and_reference():
    state = State(1, "top", 0, (None, None, None), 0, 0)
    unit = Unit(1, 0, 0, "batter", "double", 1, state)
    event = {"gameSeq": "55", "playerId": "100", "inning": "1", "ih": "0", "outs": "0",
             "base": "___", "score": "0-0", "playDesc": "Synthetic batter hit a ground rule double.",
             "wpa": ".05", "li": "1.1", "we": ".45"}
    return unit, event


def test_synthetic_unique_semantic_match_not_positional_renumbering():
    unit, event = synthetic_unit_and_reference()
    matched = match_units_by_state([unit], [event], {100: 1})
    assert matched[0][1]["gameSeq"] == "55"


def test_synthetic_ambiguous_semantic_match_is_quarantined():
    unit, event = synthetic_unit_and_reference()
    with pytest.raises(ValueError, match="ambiguous"):
        match_units_by_state([unit], [event, {**event, "gameSeq": "56"}], {100: 1})


@pytest.mark.parametrize("key,value", [("base","1__"),("outs","1"),("score","1-0"),("ih","1"),("playerId","101")])
def test_synthetic_wrong_state_or_actor_never_fuzzy_matches(key, value):
    unit, event = synthetic_unit_and_reference()
    with pytest.raises(ValueError, match="missing_reference"):
        match_units_by_state([unit], [{**event, key: value}], {100: 1, 101: 2})


def test_research_formula_contract_hash_unchanged():
    previous = json.loads((ROOT / "reports/attribution_provenance.json").read_text())["formula_config_sha256"]
    assert hashlib.sha256((ROOT / "config/research_v1.json").read_bytes()).hexdigest() == previous


def test_real_missing_records_are_automatic_ball_walks_not_imputed():
    report = investigate()
    assert report["missing_sequences"] == [1, 56]
    assert report["direct_fangraphs_api_rows"] == 80
    assert all(c["automatic_ball_walk"] and not c["replacement_performed"] for c in report["cases"])
    assert all(c["same_provider_wpa"] is None and c["same_provider_pre_pa_li"] is None for c in report["cases"])
    assert not report["same_model_complete_alternative_verified"]
    assert not report["pitchfx_numeric_wpa_or_li_fields"]
