"""Offline validation on the ten predeclared additional 2025 games.

    State/ID matching isolates an unresolved PA without shifting later events.
    The strict legacy sequence matcher remains available for earlier snapshots.
"""

from collections import Counter, defaultdict
from dataclasses import replace
import hashlib
import json
import math
import re
from datetime import datetime

from bs4 import BeautifulSoup

from .attribution import build_ledger, start_state, state_matches, RUNNER_TYPES
from .collection import CachedClient, atomic_json, ensure_scope, utc_now, read_cached_body
from .sample import ROOT, number
from .validate_attribution import check_in_pa_boundaries, difference_summary
from .verify_sources import build_crosswalk, write_csv

VERSION = "attribution-validation v0.3"


def parse_query(content, key):
    if content.lstrip().startswith((b"{", b"[")):
        data = json.loads(content)
        if key == "scores/box-score" and isinstance(data, dict) and {"home", "away"} <= data.keys():
            return data
        if key == "scores/play-log" and isinstance(data, list):
            return data
        raise ValueError("unexpected_direct_reference_schema")
    soup = BeautifulSoup(content, "html.parser")
    script = soup.find("script", id="__NEXT_DATA__")
    if script is None:
        raise ValueError("missing_fangraphs_structured_page_data")
    queries = json.loads(script.string)["props"]["pageProps"]["dehydratedState"]["queries"]
    values = [q["state"]["data"] for q in queries if q["queryKey"][0] == key]
    if len(values) != 1:
        raise ValueError("missing_or_ambiguous_fangraphs_query")
    return values[0]


def content_for(receipt, client):
    return read_cached_body(client.root, receipt)


def opportunity_state(plays, position):
    """Observed automatic-runner placement is a start condition, not WPA."""
    state = start_state(plays, position)
    p = plays[position]
    placements = [e for e in p["playEvents"] if e.get("details", {}).get("eventType") == "runner_placed"]
    if not placements:
        return state, "previous_official_post_state"
    pitches = [e["index"] for e in p["playEvents"] if e.get("isPitch")]
    if (len(placements) != 1 or state.inning < 10 or state.outs != 0 or any(state.bases)
        or not pitches or placements[0]["index"] >= min(pitches)):
        raise ValueError("unverified_automatic_runner_placement")
    actor = placements[0].get("player", {}).get("id")
    corroborated = actor is not None and (
        p["matchup"].get("postOnSecond", {}).get("id") == actor or any(
            r["details"]["runner"]["id"] == actor and r["movement"].get("originBase") == "2B"
            for r in p.get("runners", [])))
    if not corroborated:
        raise ValueError("automatic_runner_identity_unverified")
    return replace(state, bases=(None, actor, None)), "explicit_automatic_runner_placement"


def event_label_matches(unit, event):
    """Description corroborates explicit MLB codes; never drives actor joins."""
    text = event["playDesc"].lower()
    code = unit.event_type
    if unit.kind == "runner":
        if code.startswith("stolen_base"):
            return "stolen base" in text or "steal" in text
        if code.startswith("pickoff_error"):
            return "error" in text
        if code.startswith("pickoff") or code.startswith("caught_stealing"):
            return bool(re.search(r"pick|caught stealing|out stealing", text))
        return {"wild_pitch": "wild pitch", "passed_ball": "passed ball", "balk": "balk",
                "defensive_indiff": "indifference", "other_out": "out"}.get(code, "__unsupported__") in text
    expressions = {
        "single": r"singled", "double": r"doubled|ground rule double", "triple": r"tripled", "home_run": r"homered|home run",
        "walk": r"walked", "intent_walk": r"walked", "hit_by_pitch": r"hit by.*pitch",
        "strikeout": r"struck out|strike|dropped third", "strikeout_double_play": r"strike|struck out",
        "field_error": r"error", "catcher_interf": r"interference",
        "sac_fly": r"sacrifice fly", "sac_bunt": r"sacrifice|bunt",
        "fielders_choice": r"fielder|choice|forced", "fielders_choice_out": r"fielder|choice|forced|out",
        "force_out": r"fielder|forced|force", "field_out": r"out|retired",
        "double_play": r"out|double play", "grounded_into_double_play": r"double play",
        "triple_play": r"triple play", "sac_fly_double_play": r"sacrifice|double play",
        "sac_bunt_double_play": r"sacrifice|double play|bunt",
    }
    return bool(re.search(expressions.get(code, r"(?!)"), text))


def match_units_by_state(units, events, crosswalk, *, label_matcher=None):
    """Unique actor/state/type matches; preserve original seq and source gaps."""
    selected = []
    for unit in units:
        candidates = [e for e in events if crosswalk.get(int(e["playerId"])) == unit.actor_id
                      and state_matches(unit, e) and (label_matcher or event_label_matches)(unit, e)]
        if len(candidates) != 1:
            raise ValueError("missing_reference_event" if not candidates else "ambiguous_reference_event")
        e = candidates[0]
        values = {k: None if e.get(k) in (None, "") else number(float(e[k]), low, high, f"reference {k}")
                  for k, low, high in (("wpa", -1, 1), ("li", 0, math.inf), ("we", 0, 1))}
        if any(v is None for v in values.values()):
            raise ValueError("missing_reference_numeric_value")
        selected.append((unit, e, values))
    seq = [int(e["gameSeq"]) for _, e, _ in selected]
    if seq != sorted(set(seq)) or any(b != a + 1 for a, b in zip(seq, seq[1:])):
        raise ValueError("unresolved_reference_events_inside_pa")
    for (_, before, first), (after_unit, after, second) in zip(selected, selected[1:]):
        sign = -1 if after_unit.before.half == "top" else 1
        if abs(second["we"] - first["we"] - sign * second["wpa"]) > 2e-5:
            raise ValueError("within_pa_reference_wp_continuity_failed")
    return selected


def validate_one(game, checkpoint_record, client, *, crosswalk_builder=None,
                 state_builder=None, ledger_builder=None, reference_matcher=None):
    if checkpoint_record.get("collection_status") != "raw_complete":
        raise ValueError("raw_collection_incomplete")
    receipts = checkpoint_record["sources"]
    m = json.loads(content_for(receipts["metadata"], client))
    plays = json.loads(content_for(receipts["wp"], client))
    identity = m["gameData"]
    if (identity["game"]["pk"], identity["game"]["season"], identity["game"]["type"], identity["datetime"]["officialDate"]) != (
        game["game_pk"], "2025", "R", game["date"]):
        raise ValueError("game_identity_mismatch")
    official_plays = m["liveData"]["plays"]["allPlays"]
    if len(plays) != len(official_plays) or any(
        p["about"]["atBatIndex"] != q["about"]["atBatIndex"] or p["matchup"]["batter"]["id"] != q["matchup"]["batter"]["id"]
        or p["result"]["eventType"] != q["result"]["eventType"] for p, q in zip(plays, official_plays, strict=True)):
        raise ValueError("mlb_wp_feed_identity_disagreement")
    box = parse_query(content_for(receipts["fangraphs_box"], client), "scores/box-score")
    events = parse_query(content_for(receipts["fangraphs_log"], client), "scores/play-log")
    if any(datetime.strptime(e["gameDate"], "%m/%d/%Y").date().isoformat() != game["date"] for e in events):
        raise ValueError("fangraphs_game_date_mismatch")
    crosswalk, ids = (crosswalk_builder or build_crosswalk)(m, box)
    seq = [int(e["gameSeq"]) for e in events]
    if len(seq) != len(set(seq)):
        raise ValueError("duplicate_reference_sequence")
    # All offensive events (including unresolved runner actions) should sum to
    # the provider's own box WPA. This checks accounting, not an independent WE model.
    totals = defaultdict(float)
    for e in events:
        totals[int(e["playerId"]) ] += number(float(e["wpa"]), -1, 1, "reference WPA")
    residuals = []
    bad_actors = set()
    for side in ("home", "away"):
        for player in box[side]["bat"]:
            if player["RowType"] != 0 or player["WPA"] is None:
                continue
            diff = player["WPA"] - totals[int(player["playerid"])]
            residuals.append(diff)
            if abs(diff) > 1e-4:
                bad_actors.add(crosswalk[int(player["playerid"])])
    rows, event_rows, non_pa, cases = [], [], [], []
    counts = Counter()
    claimed = set()
    action_counts = Counter()
    for position, p in enumerate(plays):
        index = p["about"]["atBatIndex"]
        credits = {c["player"]["id"] for c in p.get("credits", []) if c["credit"] == "b_pa"}
        is_pa = bool(credits)
        actor = next(iter(credits)) if len(credits) == 1 else p["matchup"]["batter"]["id"]
        if is_pa:
            counts[actor] += 1
        for e in p["playEvents"]:
            code = e.get("details", {}).get("eventType")
            if code in RUNNER_TYPES:
                action_counts[code] += 1
        row = {"game_pk": game["game_pk"], "date": game["date"], "at_bat_index": index,
               "batter_id": actor, "result": p["result"]["eventType"],
               "validated": False, "reason": "", "batter_attributed_wpa": None, "pre_pa_li": None,
               "li_at_batter_event": None, "pre_pa_li_reference_sequence": None,
               "batter_reference_sequence": None, "independent_event_count": None,
               "state_method": None, "reference": "FanGraphs observed event values"}
        sign = -1 if p["about"]["halfInning"] == "top" else 1
        row["api_supplied_batting_wpa"] = None if p.get("homeTeamWinProbabilityAdded") is None else sign * number(
            p["homeTeamWinProbabilityAdded"], -100, 100, "API WPA") / 100
        row["api_li_timing_unverified"] = p.get("leverageIndex")
        try:
            state, method = (state_builder or opportunity_state)(plays, position)
            row["state_method"] = method
            units, blockers = (ledger_builder or build_ledger)([p], initial_states={index: state})
            if blockers:
                raise ValueError(";".join(sorted({b["reason"] for b in blockers})))
            check_in_pa_boundaries([p], units)
            matched = (reference_matcher or match_units_by_state)(units, events, crosswalk)
            sequences = {int(e["gameSeq"]) for _, e, _ in matched}
            if sequences & claimed:
                raise ValueError("reference_event_reused")
            if is_pa and actor in bad_actors:
                raise ValueError("reference_box_accounting_unresolved")
            claimed.update(sequences)
            if is_pa:
                batter = [x for x in matched if x[0].kind == "batter"]
                if len(batter) != 1:
                    raise ValueError("ambiguous_batter_terminal")
                _, e, value = batter[0]
                row.update({"validated": True, "batter_attributed_wpa": value["wpa"], "pre_pa_li": matched[0][2]["li"],
                            "li_at_batter_event": value["li"], "pre_pa_li_reference_sequence": int(matched[0][1]["gameSeq"]),
                            "batter_reference_sequence": int(e["gameSeq"]), "independent_event_count": len(matched) - 1})
            for u, e, value in matched:
                event_rows.append({"game_pk": game["game_pk"], "at_bat_index": index, "kind": u.kind,
                                   "event_type": u.event_type, "actor_mlbam_id": u.actor_id, "reference_sequence": int(e["gameSeq"]),
                                   "inning": u.before.inning, "half": u.before.half, "outs": u.before.outs, "bases": u.before.mask,
                                   "home_score": u.before.home_score, "away_score": u.before.away_score,
                                   "reference_wpa": value["wpa"], "reference_li": value["li"]})
        except (ValueError, KeyError, TypeError, StopIteration) as exc:
            row["reason"] = str(exc)
            cases.append({"at_bat_index": index, "is_completed_pa": is_pa, "reason": str(exc), "result": p["result"]["eventType"]})
        if is_pa:
            rows.append(row)
        else:
            non_pa.append({"at_bat_index": index, "event": p["result"]["eventType"], "matched": not bool(row["reason"]), "reason": row["reason"]})
    official_counts = {}
    for side in ("home", "away"):
        for player in m["liveData"]["boxscore"]["teams"][side]["players"].values():
            official_counts[player["person"]["id"]] = player.get("stats", {}).get("batting", {}).get("plateAppearances", 0)
    if any(counts[k] != v for k, v in official_counts.items()) or sum(official_counts.values()) != len(rows):
        raise ValueError("official_pa_denominator_mismatch")
    n = len(rows)
    passed = sum(r["validated"] for r in rows)
    report = {"game_pk": game["game_pk"], "date": game["date"], "away": game["away_team_name"], "home": game["home_team_name"],
              "official_pa": n, "validated_pa": passed, "validation_percentage": 100 * passed / n,
              "unresolved_pa": n - passed, "unresolved_cases": cases, "runner_only_plays": non_pa,
              "independent_action_codes": dict(action_counts), "reference_box_accounting": difference_summary(residuals),
              "unexplained_box_actor_ids": sorted(bad_actors), "fangraphs_rows": len(events),
              "unmatched_reference_sequences": sorted(set(seq) - claimed),
              "api_missing_li_play_indices": [p["about"]["atBatIndex"] for p in plays if p.get("leverageIndex") is None],
              "automatic_runner_pa_count": sum(r["state_method"] == "explicit_automatic_runner_placement" for r in rows),
              "api_vs_reference_batter_wpa": difference_summary([r["api_supplied_batting_wpa"] - r["batter_attributed_wpa"]
                                                                 for r in rows if r["validated"] and r["api_supplied_batting_wpa"] is not None]),
              "api_li_vs_reference_pre_pa_li": difference_summary([r["api_li_timing_unverified"] - r["pre_pa_li"]
                                                                  for r in rows if r["validated"] and r["api_li_timing_unverified"] is not None]),
              "id_crosswalk_entries": len(ids), "status": "passed" if passed == n else "partial"}
    return rows, event_rows, report


def main():
    manifest = json.loads((ROOT / "config/validation_2025_games.json").read_text())
    ensure_scope(manifest)
    if len(manifest["games"]) != 10 or {g["game_pk"] for g in manifest["games"]} & {745444, 778563}:
        raise ValueError("Exactly ten additional games are required")
    path = ROOT / "data/checkpoints/validation_2025.json"
    checkpoint = json.loads(path.read_text())
    client = CachedClient(offline=True)
    all_rows, all_events, reports = [], [], []
    for game in manifest["games"]:
        record = checkpoint["games"][str(game["game_pk"])]
        rows, events, report = validate_one(game, record, client)
        all_rows.extend(rows); all_events.extend(events); reports.append(report)
        record["validation_status"] = report["status"]
        record["validator_version"] = VERSION
        record["validated_at_utc"] = utc_now()
        record["validated_pa"] = report["validated_pa"]
        record["validation_source_hashes"] = {k: v["sha256"] for k, v in record["sources"].items()}
        record["validator_source_sha256"] = hashlib.sha256((ROOT / "src/mlb_wpa/final_validation.py").read_bytes()).hexdigest()
        atomic_json(path, checkpoint)
        print(game["game_pk"], report["validated_pa"], "/", report["official_pa"], report["unresolved_cases"], flush=True)
    denominator = sum(g["official_pa"] for g in reports)
    numerator = sum(g["validated_pa"] for g in reports)
    report = {"method_version": VERSION, "scope": "ten additional real 2025 games only", "games": reports,
              "additional_games": 10, "official_pa": denominator, "validated_pa": numerator,
              "validation_percentage": 100 * numerator / denominator, "unresolved_pa": denominator - numerator,
              "full_season_collection_authorized": False, "season_scale_ready": False,
              "missing_policy": "null and quarantined with reasons; no model mixing or residual imputation"}
    report["api_vs_reference_batter_wpa"] = difference_summary([r["api_supplied_batting_wpa"] - r["batter_attributed_wpa"]
                                                              for r in all_rows if r["validated"] and r["api_supplied_batting_wpa"] is not None])
    report["api_li_vs_reference_pre_pa_li"] = difference_summary([r["api_li_timing_unverified"] - r["pre_pa_li"]
                                                               for r in all_rows if r["validated"] and r["api_li_timing_unverified"] is not None])
    report["recorded_independent_action_codes"] = dict(sum((Counter(g["independent_action_codes"]) for g in reports), Counter()))
    report["runner_only_plays"] = [dict(game_pk=g["game_pk"], **p) for g in reports for p in g["runner_only_plays"]]
    report["automatic_runner_start_pas"] = sum(g["automatic_runner_pa_count"] for g in reports)
    report["automatic_ball_walks_in_additional_sample"] = 0
    for game in manifest["games"]:
        plays = json.loads(content_for(checkpoint["games"][str(game["game_pk"])]["sources"]["wp"], client))
        report["automatic_ball_walks_in_additional_sample"] += sum(
            p["result"]["eventType"] == "walk" and any(e.get("details", {}).get("code") == "VP" and e.get("count", {}).get("balls") == 4
                                                       for e in p["playEvents"]) for p in plays)
    atomic_json(ROOT / "reports/final_validation_2025.json", report)
    write_csv(ROOT / "data/processed/final_validation_2025_pas.csv", all_rows)
    write_csv(ROOT / "data/processed/final_validation_2025_events.csv", all_events)
    provenance = {"method_version": VERSION, "generated_at_utc": utc_now(),
                  "selection_manifest": "config/validation_2025_games.json",
                  "selection_manifest_sha256": hashlib.sha256((ROOT / "config/validation_2025_games.json").read_bytes()).hexdigest(),
                  "inputs": {g["game_pk"]: checkpoint["games"][str(g["game_pk"])]["sources"] for g in manifest["games"]},
                  "research_config_sha256": hashlib.sha256((ROOT / "config/research_v1.json").read_bytes()).hexdigest(),
                  "source_hashes": {str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest() for p in
                                    (ROOT / "src/mlb_wpa/attribution.py", ROOT / "src/mlb_wpa/final_validation.py", ROOT / "src/mlb_wpa/collection.py",
                                     ROOT / "src/mlb_wpa/validate_attribution.py", ROOT / "src/mlb_wpa/verify_sources.py")}}
    atomic_json(ROOT / "reports/final_validation_provenance.json", provenance)
    print(json.dumps({k: v for k, v in report.items() if k != "games"}, indent=2))


if __name__ == "__main__":
    main()
