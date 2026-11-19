"""Reproducible offline attribution/LI audit of two fixed real games.

No EOWS calculations, season collection, fitted probability models or rankings.
"""

import csv
import hashlib
import json
import math
import re
from collections import Counter, defaultdict
from datetime import datetime, timezone

from bs4 import BeautifulSoup

from .attribution import build_ledger, match_reference, pa_records
from .sample import ROOT, number
from .verify_sources import FOLDER, audit_hashes, build_crosswalk, fg_query, load_json, write_csv

SAMPLES = (
    {"year": 2024, "game_pk": 745444, "wp": ROOT / "data/raw/745444_win_probability.json"},
    {"year": 2025, "game_pk": 778563, "wp": FOLDER / "2025_wp.json"},
)


def check_in_pa_boundaries(plays, units):
    """Require evidence that a donor runner action occurs inside a started PA.

    Between-batter events and missing clocks need review. A runner-only third
    out is not a donor for a completed PA and is handled separately.
    """
    by_index = {p["about"]["atBatIndex"]: p for p in plays}
    checks = []
    for u in units:
        if u.kind != "runner":
            continue
        p = by_index[u.play_index]
        if not any(c.get("credit") == "b_pa" for c in p.get("credits", [])):
            continue
        event = next(e for e in p["playEvents"] if e["index"] == u.action_index)
        start, when = p["about"].get("startTime"), event.get("startTime")
        prior_pitch = any(e.get("isPitch") and e["index"] < u.action_index for e in p["playEvents"])
        # A balk can precede the first pitch while the named batter is already
        # batting. Require explicit source context as well as the PA clock;
        # absence of a prior pitch alone is not evidence of a between-PA event.
        explicit_batter_context = event.get("details", {}).get("description", "").startswith(
            f"With {p['matchup']['batter'].get('fullName', '__unknown__')} batting,")
        try:
            after_start = bool(start and when) and datetime.fromisoformat(when.replace("Z", "+00:00")) >= datetime.fromisoformat(start.replace("Z", "+00:00"))
        except ValueError:
            after_start = False
        if not (prior_pitch or explicit_batter_context) or not after_start:
            raise ValueError("runner_action_pa_boundary_unverified")
        checks.append({"at_bat_index": u.play_index, "prior_pitch_observed": prior_pitch,
                       "explicit_batter_context": explicit_batter_context, "action_after_pa_start_timestamp": True})
    return checks


def retrospective_runner_check(year, metadata, units):
    """Corroborate action types/order against a separate Retrosheet narrative.

    This is an independent event-sequence check, NOT an independent WPA/LI
    calculation or a player-ID crosswalk based on names.
    """
    soup = BeautifulSoup((FOLDER / f"{year}_retrosheet.html").read_text(), "html.parser")
    pbp = [p.get_text() for p in soup.find_all("pre") if "Starting Lineups:" in p.get_text()]
    if len(pbp) != 1:
        raise ValueError("Retrosheet narrative format changed")
    teams = metadata["gameData"]["teams"]
    sides = {teams["away"]["teamName"].upper(): "top", teams["home"]["teamName"].upper(): "bottom"}
    header = re.compile(r"(" + "|".join(re.escape(k) for k in sides) + r") (\d+)(?:ST|ND|RD|TH):")
    headings = list(header.finditer(pbp[0]))
    observed = []
    for pos, h in enumerate(headings):
        text = pbp[0][h.end():headings[pos + 1].start() if pos + 1 < len(headings) else len(pbp[0])]
        text = re.sub(r"\s+", " ", text)
        for action in re.finditer(r"threw a wild pitch|stole second|stole third|stole home", text):
            code = {"threw a wild pitch": "wild_pitch", "stole second": "stolen_base_2b",
                    "stole third": "stolen_base_3b", "stole home": "stolen_base_home"}[action.group()]
            observed.append((int(h.group(2)), sides[h.group(1)], code))
    expected = [(u.before.inning, u.before.half, u.event_type) for u in units if u.kind == "runner"]
    if observed != expected:
        raise ValueError("Retrosheet/MLB runner event order mismatch")
    return {"status": "type_inning_half_and_order_agree", "runner_events": len(observed),
            "independent_numeric_wpa_or_li_validation": False}


def difference_summary(values):
    if not values:
        return {"n": 0, "max_abs_difference": None, "mean_abs_difference": None}
    return {"n": len(values), "max_abs_difference": max(abs(v) for v in values),
            "mean_abs_difference": sum(abs(v) for v in values) / len(values)}


def audit_game(sample):
    year, game_pk = sample["year"], sample["game_pk"]
    metadata = load_json(f"{year}_metadata.json")
    game = metadata["gameData"]["game"]
    if (game["pk"], int(game["season"]), game["type"]) != (game_pk, year, "R"):
        raise ValueError("sample game identity mismatch")
    plays = json.loads(sample["wp"].read_text())
    box = fg_query(f"{year}_fangraphs.html", "scores/box-score")
    crosswalk, ids = build_crosswalk(metadata, box)
    units, blockers = build_ledger(plays)
    if blockers:
        return [], [], [], {"game_pk": game_pk, "year": year, "status": "blocked", "blockers": blockers}
    boundary_checks = check_in_pa_boundaries(plays, units)
    events = fg_query(f"{year}_fangraphs_play_log.html", "scores/play-log")
    matches = match_reference(units, events, crosswalk)
    records = pa_records(game_pk, units, matches)
    counts = Counter(u.actor_id for u in units if u.kind == "batter")
    for side in ("home", "away"):
        for p in metadata["liveData"]["boxscore"]["teams"][side]["players"].values():
            if counts[p["person"]["id"]] != p.get("stats", {}).get("batting", {}).get("plateAppearances", 0):
                raise ValueError("official per-player PA reconciliation failed")
    event_rows = []
    for u in units:
        event_rows.append({"game_pk": game_pk, "sequence": u.sequence, "at_bat_index": u.play_index,
                           "mlb_action_index": u.action_index, "kind": u.kind, "event_type": u.event_type,
                           "actor_mlbam_id": u.actor_id, "inning": u.before.inning, "half": u.before.half,
                           "outs_before": u.before.outs, "bases_before": u.before.mask,
                           "home_score_before": u.before.home_score, "away_score_before": u.before.away_score,
                           "reference_status": matches[u.sequence]["status"],
                           "reference_wpa": matches[u.sequence]["wpa"], "reference_li": matches[u.sequence]["li"]})
    source_sums = defaultdict(float)
    missing_by_actor = Counter()
    for u in units:
        value = matches[u.sequence]["wpa"]
        if value is None:
            missing_by_actor[u.actor_id] += 1
        else:
            source_sums[u.actor_id] += value
    residuals, complete_residuals, null_box = [], [], []
    for side in ("home", "away"):
        for p in box[side]["bat"]:
            if p["RowType"] != 0:
                continue
            actor = crosswalk[int(p["playerid"])]
            if p["WPA"] is None:
                null_box.append({"mlbam_id": actor, "official_pa": counts[actor]})
                continue
            diff = p["WPA"] - source_sums[actor]
            if missing_by_actor[actor]:
                residuals.append({"mlbam_id": actor, "missing_event_count": missing_by_actor[actor],
                                  "box_minus_observed_event_sum": diff, "used_to_fill_missing_values": False})
            else:
                complete_residuals.append(diff)
    if any(abs(d) > 1e-4 for d in complete_residuals):
        raise ValueError("complete-player reference box/event WPA mismatch")
    continuity = []
    for first, second in zip(events, events[1:]):
        # A gap cannot be validated as an adjacent event transition.
        if int(second["gameSeq"]) == int(first["gameSeq"]) + 1:
            sign = 1 if second["ih"] == "1" else -1
            continuity.append(float(second["we"]) - float(first["we"]) - sign * float(second["wpa"]))
    if any(abs(d) > 2e-5 for d in continuity):
        raise ValueError("reference WP continuity failed")
    comparisons = []
    for row in records:
        p = next(p for p in plays if p["about"]["atBatIndex"] == row["at_bat_index"])
        sign = -1 if p["about"]["isTopInning"] else 1
        pos = plays.index(p)
        before = plays[pos - 1]["homeTeamWinProbability"] if pos else 50.0
        whole = sign * (p["homeTeamWinProbability"] - before) / 100
        supplied = sign * p["homeTeamWinProbabilityAdded"] / 100
        comparisons.append({"game_pk": game_pk, "at_bat_index": row["at_bat_index"],
                            "batter_id": row["batter_id"], "batting_half": p["about"]["halfInning"],
                            "api_supplied_offense_wpa": supplied, "api_whole_pa_offense_wpa": whole,
                            "api_residual_independent_change": whole - supplied,
                            "api_li_unverified_time": p.get("leverageIndex"),
                            "reference_batter_wpa": row["batter_attributed_wpa"],
                            "reference_whole_pa_wpa": row["whole_pa_wpa"],
                            "reference_runner_wpa": row["runner_wpa"],
                            "reference_pre_pa_li": row["pre_pa_li"],
                            "reference_batter_event_li": row["li_at_batter_event"]})
    report = {
        "game_pk": game_pk, "year": year, "status": "partial_reference" if residuals else "reviewed_sample_passed",
        "official_pa": len(records), "expected_scoring_events": len(units), "observed_fangraphs_events": len(events),
        "independent_events": dict(Counter(u.event_type for u in units if u.kind == "runner")),
        "missing_reference_sequences": [u.sequence for u in units if matches[u.sequence]["status"] == "missing_reference_event"],
        "missing_batter_wpa_pa_indices": [r["at_bat_index"] for r in records if r["batter_attributed_wpa"] is None],
        "missing_pre_pa_li_pa_indices": [r["at_bat_index"] for r in records if r["pre_pa_li"] is None],
        "available_batter_wpa_and_pre_pa_li": sum(r["metric_input_available"] for r in records),
        "api_missing_li_pa_indices": [p["about"]["atBatIndex"] for p in plays if p.get("leverageIndex") is None],
        "null_box_wpa": null_box,
        "missing_event_box_residuals": residuals,
        "complete_player_box_reconciliation": difference_summary(complete_residuals),
        "reference_adjacent_wp_continuity": difference_summary(continuity),
        "api_vs_reference_batter_wpa": difference_summary([r["api_supplied_offense_wpa"] - r["reference_batter_wpa"]
                                                         for r in comparisons if r["reference_batter_wpa"] is not None]),
        "api_li_vs_reference_pre_pa_li": difference_summary([r["api_li_unverified_time"] - r["reference_pre_pa_li"]
                                                            for r in comparisons if r["api_li_unverified_time"] is not None and r["reference_pre_pa_li"] is not None]),
        "api_li_vs_reference_batter_event_li": difference_summary([r["api_li_unverified_time"] - r["reference_batter_event_li"]
                                                                  for r in comparisons if r["api_li_unverified_time"] is not None and r["reference_batter_event_li"] is not None]),
        "api_whole_vs_supplied_differing_pa_indices": [r["at_bat_index"] for r in comparisons
                                                      if abs(r["api_residual_independent_change"]) > 1e-8],
        "retrosheet": retrospective_runner_check(year, metadata, units),
        "in_pa_boundary_checks": boundary_checks,
        "crosswalk_players": len(ids), "blockers": [],
    }
    return records, event_rows, comparisons, report


def audit_savant(comparisons):
    rows = list(csv.DictReader((FOLDER / "2025_savant_both_teams.csv").open()))
    if {r["game_pk"] for r in rows} != {"778563"}:
        raise ValueError("Savant sample exceeded authorized game")
    grouped = defaultdict(list)
    for row in rows:
        grouped[int(row["at_bat_number"]) - 1].append(row)
    source = {r["at_bat_index"]: r for r in comparisons if r["game_pk"] == 778563}
    result = []
    for index, pitches in sorted(grouped.items()):
        pitches.sort(key=lambda r: int(r["pitch_number"]))
        if len({p["pitch_number"] for p in pitches}) != len(pitches):
            raise ValueError("duplicate Savant pitch")
        terminal = [p for p in pitches if p["events"]]
        if len(terminal) != 1:
            raise ValueError("Savant terminal PA mismatch")
        values = [number(float(p["delta_home_win_exp"]), -1, 1, "Savant delta") if p["delta_home_win_exp"] else None for p in pitches]
        sign = -1 if terminal[0]["inning_topbot"] == "Top" else 1
        reference = source[index]
        if any(int(p["batter"]) != reference["batter_id"] for p in pitches) or (
            terminal[0]["inning_topbot"] == "Top"
        ) != (reference["batting_half"] == "top"):
            raise ValueError("Savant stable batter ID or inning half mismatch")
        result.append({"game_pk": 778563, "at_bat_index": index, "pitch_rows": len(pitches),
                       "savant_terminal_offense_delta": sign * float(terminal[0]["delta_home_win_exp"]),
                       "savant_sum_offense_deltas": sign * sum(values) if all(v is not None for v in values) else None,
                       "nonterminal_nonzero_deltas": sum(v is not None and v != 0 for v in values[:-1]),
                       "api_whole_pa_offense_wpa": reference["api_whole_pa_offense_wpa"],
                       "reference_batter_wpa": reference["reference_batter_wpa"],
                       "reference_whole_pa_wpa": reference["reference_whole_pa_wpa"]})
    return result, {
        "pitch_rows": len(rows), "completed_pas": len(result),
        "pa_with_nonzero_nonterminal_deltas": sum(r["nonterminal_nonzero_deltas"] > 0 for r in result),
        "terminal_delta_differs_from_summed_delta_pa_count": sum(abs(r["savant_terminal_offense_delta"] - r["savant_sum_offense_deltas"]) > 1e-8 for r in result),
        "sum_delta_vs_api_whole_pa": difference_summary([r["savant_sum_offense_deltas"] - r["api_whole_pa_offense_wpa"] for r in result]),
        "sum_delta_vs_fangraphs_whole_pa": difference_summary([r["savant_sum_offense_deltas"] - r["reference_whole_pa_wpa"] for r in result]),
        "missing_delta_rows": sum(not r["delta_home_win_exp"] for r in rows),
        "li_column_present": any("leverage" in k.lower() or k.lower() == "li" for k in rows[0]),
        "interpretation": "Observed pitch-level deltas; terminal delta alone is not PA WPA despite CSV documentation's PA wording",
    }


def main():
    checked = audit_hashes()
    all_pa, all_events, all_comparisons, reports, all_ids = [], [], [], [], []
    for sample in SAMPLES:
        pa, events, comparisons, report = audit_game(sample)
        all_pa.extend(pa); all_events.extend(events); all_comparisons.extend(comparisons); reports.append(report)
        _, id_rows = build_crosswalk(load_json(f"{sample['year']}_metadata.json"),
                                     fg_query(f"{sample['year']}_fangraphs.html", "scores/box-score"))
        all_ids.extend(id_rows)
    savant, savant_report = audit_savant(all_comparisons)
    report = {"method_version": "attribution-validation v0.2", "scope": "two fixed games only; no EOWS implementation",
              "games": reports, "savant_2025": savant_report, "response_hashes_checked": checked,
              "season_scale_ready": False,
              "remaining_blockers": ["2024 FanGraphs reference missing two scoring events; no values imputed",
                                     "caught stealing, pickoffs, passed balls, balks and simultaneous actions need real reference samples",
                                     "sequence and lineup conventions not yet validated on representative domestic/extra-inning/suspended games"]}
    write_csv(ROOT / "data/processed/attribution_validation_pas.csv", all_pa)
    write_csv(ROOT / "data/processed/attribution_validation_events.csv", all_events)
    write_csv(ROOT / "data/processed/source_wpa_comparisons.csv", all_comparisons)
    write_csv(ROOT / "data/processed/savant_field_validation.csv", savant)
    write_csv(ROOT / "data/processed/attribution_player_id_crosswalk.csv", all_ids)
    (ROOT / "reports/attribution_validation.json").write_text(json.dumps(report, indent=2) + "\n")
    inputs = []
    for path in sorted(FOLDER.glob("*_provenance.json")):
        manifest = json.loads(path.read_text())
        inputs.append({"manifest": str(path.relative_to(ROOT)), "url": manifest["url"], "sha256": manifest["sha256"],
                       "status_code": manifest["status_code"], "retrieved_at_utc": manifest["retrieved_at_utc"]})
    inputs.append({"manifest": "data/raw/745444_provenance.json", **json.loads((ROOT / "data/raw/745444_provenance.json").read_text())})
    provenance = {"generated_at_utc": datetime.now(timezone.utc).isoformat(), "method_version": report["method_version"],
                  "command": ".venv/bin/python -m mlb_wpa.validate_attribution", "inputs": inputs,
                  "formula_config_sha256": hashlib.sha256((ROOT / "config/research_v1.json").read_bytes()).hexdigest(),
                  "code_sha256": {str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest()
                                  for p in (ROOT / "src/mlb_wpa/attribution.py", ROOT / "src/mlb_wpa/validate_attribution.py",
                                            ROOT / "src/mlb_wpa/verify_sources.py", ROOT / "src/mlb_wpa/sample.py")},
                  "independent_numeric_reference": "FanGraphs event values; box agreement is same-provider accounting, not independent model validation",
                  "retrosheet_role": "independent recorded action type/order only, no WPA or LI provided",
                  "missing_policy": "null, with reasons; no box-residual, cross-provider, or average-LI imputation"}
    (ROOT / "reports/attribution_provenance.json").write_text(json.dumps(provenance, indent=2) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
