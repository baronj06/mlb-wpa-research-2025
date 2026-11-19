"""Synthetic fixtures test logic only; real regressions are labeled separately."""

from copy import deepcopy

import pytest

from mlb_wpa.attribution import build_ledger, match_reference, pa_records
from mlb_wpa.validate_attribution import SAMPLES, audit_game, audit_savant, check_in_pa_boundaries


def movement(actor, event, index, start=None, end=None, out=False):
    return {"details": {"runner": {"id": actor}, "eventType": event, "playIndex": index, "isScoringEvent": end == "score"},
            "movement": {"start": start, "end": end, "isOut": out}}


def synthetic_play(index=0, actor=99, result="strikeout", outs=1, pa=True, no_pitch=False):
    return {"about": {"atBatIndex": index, "inning": 1, "halfInning": "top", "isComplete": True},
            "matchup": {"batter": {"id": actor}}, "count": {"outs": outs},
            "result": {"eventType": result, "homeScore": 0, "awayScore": 0},
            "credits": [{"credit": "b_pa", "player": {"id": actor}}] if pa else [],
            "playEvents": [{"index": 0, "isPitch": not no_pitch,
                            "details": {"eventType": result} if no_pitch else {}}],
            "runners": []}


def synthetic_sb_fixture():
    walk = synthetic_play(actor=1, result="walk", outs=0)
    walk["matchup"]["postOnFirst"] = {"id": 1}
    strikeout = synthetic_play(index=1)
    strikeout["playEvents"] = [{"index": 0, "isPitch": False, "details": {"eventType": "stolen_base_2b"}},
                              {"index": 1, "isPitch": True, "details": {}}]
    strikeout["runners"] = [movement(1, "stolen_base_2b", 0, "1B", "2B")]
    return [walk, strikeout]


def synthetic_reference(units):
    data = []
    for u, wpa, we, li in zip(units, [.05, .02, -.03], [.45, .43, .46], [.9, 1.2, 1.4], strict=True):
        s = u.before
        data.append({"gameSeq": str(u.sequence), "playerId": str(u.actor_id + 1000),
                     "inning": str(s.inning), "ih": "0" if s.half == "top" else "1",
                     "outs": str(s.outs), "base": s.mask, "score": f"{s.home_score}-{s.away_score}",
                     "wpa": str(wpa), "we": str(we), "li": str(li),
                     "playDesc": "Synthetic stolen base" if u.kind == "runner" else "Synthetic batter result"})
    return data


def test_synthetic_state_and_contribution_separation():
    units, blockers = build_ledger(synthetic_sb_fixture())
    assert not blockers
    assert [u.kind for u in units] == ["batter", "runner", "batter"]
    assert units[1].before.mask == "1__"
    assert units[2].before.mask == "_2_"
    matched = match_reference(units, synthetic_reference(units), {1001: 1, 1099: 99})
    row = pa_records(7, units, matched)[1]
    assert row["batter_attributed_wpa"] == -.03
    assert row["runner_wpa"] == .02
    assert row["whole_pa_wpa"] == pytest.approx(-.01)
    assert row["pre_pa_li"] == 1.2
    assert row["li_at_batter_event"] == 1.4
    assert row["home_wp_before_pa_inferred"] == pytest.approx(.45)


@pytest.mark.parametrize("code", ["wild_pitch", "passed_ball", "balk", "stolen_base_2b"])
def test_synthetic_independent_actions_are_not_batter_results(code):
    plays = synthetic_sb_fixture()
    plays[1]["playEvents"][0]["details"]["eventType"] = code
    plays[1]["runners"][0]["details"]["eventType"] = code
    units, blockers = build_ledger(plays)
    assert not blockers
    assert units[1].kind == "runner" and units[1].event_type == code
    assert units[1].actor_id != units[2].actor_id


@pytest.mark.parametrize("code", ["caught_stealing_2b", "pickoff_1b"])
def test_synthetic_third_out_runner_event_has_no_completed_pa(code):
    first = synthetic_play(0, outs=1)
    second = synthetic_play(1, outs=2)
    walk = synthetic_play(2, actor=1, result="walk", outs=2)
    walk["matchup"]["postOnFirst"] = {"id": 1}
    runner_only = synthetic_play(3, result=code, outs=3, pa=False, no_pitch=True)
    runner_only["runners"] = [movement(1, code, 0, "1B", None, True)]
    units, blockers = build_ledger([first, second, walk, runner_only])
    assert not blockers
    assert sum(u.kind == "batter" for u in units) == 3
    assert units[-1].kind == "runner" and units[-1].before.outs == 2


def test_synthetic_no_pitch_intentional_walk_is_pa():
    units, blockers = build_ledger([synthetic_play(result="intent_walk", no_pitch=True, outs=0)])
    assert not blockers
    assert units[0].kind == "batter"


def test_synthetic_unknown_action_is_blocked():
    p = synthetic_play()
    p["playEvents"][0]["index"] = 1
    p["playEvents"].insert(0, {"index": 0, "isPitch": False, "details": {"eventType": "unrecognized_state_action"}})
    _, blockers = build_ledger([p])
    assert any(b["reason"] == "unknown_action_code" for b in blockers)


def test_synthetic_unverified_between_pa_boundary_rejected():
    plays = synthetic_sb_fixture()
    units, _ = build_ledger(plays)
    with pytest.raises(ValueError, match="boundary"):
        check_in_pa_boundaries(plays, units)


def test_synthetic_simultaneous_strikeout_wild_pitch_is_blocked():
    p = synthetic_play()
    p["runners"] = [movement(99, "wild_pitch", 0, None, "1B")]
    _, blockers = build_ledger([p])
    assert any(b["reason"] == "simultaneous_terminal_runner_and_batter_event" for b in blockers)


def test_synthetic_postterminal_runner_action_requires_review():
    plays = synthetic_sb_fixture()
    plays[1]["playEvents"][0]["index"] = 2
    plays[1]["runners"][0]["details"]["playIndex"] = 2
    _, blockers = build_ledger(plays)
    assert any(b["reason"] == "postterminal_runner_action_requires_review" for b in blockers)


def test_synthetic_multiple_runner_credit_is_blocked():
    plays = synthetic_sb_fixture()
    plays[1]["runners"].append(movement(2, "stolen_base_2b", 0, "2B", "3B"))
    _, blockers = build_ledger(plays)
    assert any("multi_runner" in b["reason"] for b in blockers)


def test_synthetic_terminal_error_retains_provider_pa_convention():
    p = synthetic_play(result="field_error", outs=0)
    p["runners"] = [movement(99, "error", 0, None, "1B")]
    units, blockers = build_ledger([p])
    assert not blockers
    assert len(units) == 1 and units[0].kind == "batter"


def test_synthetic_preterminal_error_is_quarantined():
    plays = synthetic_sb_fixture()
    plays[1]["playEvents"][0]["details"]["eventType"] = "error"
    plays[1]["runners"][0]["details"]["eventType"] = "error"
    _, blockers = build_ledger(plays)
    assert any(b["reason"] == "unknown_runner_code" for b in blockers)


@pytest.mark.parametrize("changed", ["actor", "base", "outs", "score", "inning", "half"])
def test_synthetic_wrong_identity_or_li_state_fails(changed):
    units, _ = build_ledger(synthetic_sb_fixture())
    events = synthetic_reference(units)
    key, value = {"actor": ("playerId", "1099"), "base": ("base", "_2_"), "outs": ("outs", "1"),
                  "score": ("score", "1-0"), "inning": ("inning", "2"), "half": ("ih", "1")}[changed]
    events[1][key] = value
    with pytest.raises(ValueError, match="identity|state"):
        match_reference(units, events, {1001: 1, 1099: 99})


def test_synthetic_reference_gap_not_closed_or_imputed():
    units, _ = build_ledger(synthetic_sb_fixture())
    events = synthetic_reference(units)
    matched = match_reference(units, events[1:], {1001: 1, 1099: 99})
    row = pa_records(7, units, matched)[0]
    assert row["batter_attributed_wpa"] is None
    assert row["pre_pa_li"] is None
    assert row["metric_input_available"] is False
    assert matched[2]["wpa"] == .02  # Remaining source sequence was not shifted.


@pytest.mark.parametrize("value", [None, ""])
def test_synthetic_missing_li_not_replaced(value):
    units, _ = build_ledger(synthetic_sb_fixture())
    events = synthetic_reference(units)
    events[1]["li"] = value
    matched = match_reference(units, events, {1001: 1, 1099: 99})
    row = pa_records(7, units, matched)[1]
    assert row["pre_pa_li"] is None and not row["metric_input_available"]


@pytest.mark.parametrize("value", ["nan", "inf", "-0.1"])
def test_synthetic_invalid_li_rejected(value):
    units, _ = build_ledger(synthetic_sb_fixture())
    events = synthetic_reference(units)
    events[1]["li"] = value
    with pytest.raises(ValueError):
        match_reference(units, events, {1001: 1, 1099: 99})


def test_synthetic_duplicate_reference_sequence_rejected():
    units, _ = build_ledger(synthetic_sb_fixture())
    events = synthetic_reference(units)
    events.append(deepcopy(events[0]))
    with pytest.raises(ValueError, match="duplicate"):
        match_reference(units, events, {1001: 1, 1099: 99})


@pytest.mark.parametrize("sample,pa_count,available,gaps", [(SAMPLES[0], 79, 77, [1, 56]), (SAMPLES[1], 74, 74, [])])
def test_real_samples_and_reference_missingness(sample, pa_count, available, gaps):
    rows, events, comparisons, report = audit_game(sample)
    assert len(rows) == pa_count
    assert sum(r["metric_input_available"] for r in rows) == available
    assert report["missing_reference_sequences"] == gaps
    assert report["retrosheet"]["status"] == "type_inning_half_and_order_agree"
    assert report["complete_player_box_reconciliation"]["max_abs_difference"] < 1e-4
    assert sum(e["kind"] == "runner" for e in events) == (3 if pa_count == 79 else 1)
    if gaps:
        assert [r["at_bat_index"] for r in rows if r["batter_attributed_wpa"] is None] == [0, 52]
        assert all(not d["used_to_fill_missing_values"] for d in report["missing_event_box_residuals"])


def test_real_api_supplied_delta_is_not_whole_pa_delta():
    _, _, comparisons, report = audit_game(SAMPLES[1])
    row = next(r for r in comparisons if r["at_bat_index"] == 63)
    assert row["api_supplied_offense_wpa"] == pytest.approx(-.069)
    assert row["api_whole_pa_offense_wpa"] == pytest.approx(-.048)
    assert row["api_residual_independent_change"] == pytest.approx(.021)
    assert row["reference_batter_wpa"] == pytest.approx(-.06471)
    assert row["reference_runner_wpa"] == pytest.approx(.0198)
    assert report["api_whole_vs_supplied_differing_pa_indices"] == [63]


def test_real_savant_terminal_delta_cannot_be_used_as_pa_wpa():
    _, _, comparisons, _ = audit_game(SAMPLES[1])
    rows, report = audit_savant(comparisons)
    assert report["pitch_rows"] == 307 and report["completed_pas"] == 74
    assert report["terminal_delta_differs_from_summed_delta_pa_count"] == 63
    assert report["sum_delta_vs_api_whole_pa"]["max_abs_difference"] < 1e-10
    assert report["li_column_present"] is False
    first = rows[0]
    assert first["savant_terminal_offense_delta"] == pytest.approx(-.021)
    assert first["savant_sum_offense_deltas"] == pytest.approx(-.022)
