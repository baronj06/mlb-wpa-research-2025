"""Offline, fail-closed audit of one reviewed 2025 game across three sources.

This verifies event accounting and state-aligned LI, not research metrics.
The single runner-only event was manually reviewed against MLB's play events.
Do not extend this game's allowlist into a general event classifier.
"""

import csv
import hashlib
import json
import math
from collections import Counter, defaultdict

from bs4 import BeautifulSoup

from .sample import ROOT, number

FOLDER = ROOT / "data/raw/source_probes"
RUNNER_ONLY_SEQUENCES = {64}  # Jon Berti SB during Ian Happ's PA; sample-specific.


def load_json(name):
    return json.loads((FOLDER / name).read_text())


def fg_query(filename, key):
    soup = BeautifulSoup((FOLDER / filename).read_text(), "html.parser")
    script = soup.find("script", id="__NEXT_DATA__")
    if script is None:
        raise ValueError("FanGraphs page schema changed: missing __NEXT_DATA__")
    queries = json.loads(script.string)["props"]["pageProps"]["dehydratedState"]["queries"]
    matches = [q["state"]["data"] for q in queries if q["queryKey"][0] == key]
    if len(matches) != 1:
        raise ValueError(f"Expected one query: {key}")
    return matches[0]


def audit_hashes():
    manifests = sorted(FOLDER.glob("*_provenance.json"))
    if not manifests:
        raise ValueError("No source provenance")
    for path in manifests:
        manifest = json.loads(path.read_text())
        actual = hashlib.sha256((ROOT / manifest["path"]).read_bytes()).hexdigest()
        if actual != manifest["sha256"] or manifest["status_code"] != 200:
            raise ValueError(f"Failed source integrity: {path.name}")
    return len(manifests)


def build_crosswalk(metadata, box):
    """Join this game's IDs by team side, lineup slot, and substitution number.

    Names are labels only. Sequence-level MLBAM ID agreement is checked later.
    This limited mapping is not a career-wide player register.
    """
    crosswalk, records = {}, []
    for side in ("home", "away"):
        mlb_players = metadata["liveData"]["boxscore"]["teams"][side]["players"].values()
        lineup = {}
        for p in mlb_players:
            if p.get("battingOrder"):
                order = int(p["battingOrder"])
                key = (order // 100, order % 100)
                if key in lineup:
                    raise ValueError("Ambiguous MLB lineup slot")
                lineup[key] = p
        for p in box[side]["bat"]:
            if p["RowType"] != 0:
                continue
            key = (p["BatOrder"], p["SubNum"])
            mlb = lineup.pop(key)
            fg_id = int(p["playerid"])
            if fg_id in crosswalk:
                raise ValueError("Duplicate FanGraphs player ID")
            crosswalk[fg_id] = mlb["person"]["id"]
            records.append({"game_pk": metadata["gameData"]["game"]["pk"], "side": side, "lineup_slot": key[0],
                            "substitution_number": key[1], "fangraphs_id": fg_id,
                            "mlbam_id": mlb["person"]["id"],
                            "mlb_name": mlb["person"]["fullName"],
                            "fangraphs_name": p["PlayerName"],
                            "official_pa": mlb.get("stats", {}).get("batting", {}).get("plateAppearances", 0),
                            "method": "game_lineup_slot_and_substitution; event_ids_checked_where_observed"})
        if lineup:
            raise ValueError("Unmatched MLB lineup entries")
    return crosswalk, records


def pre_pa_state(plays, position):
    play = plays[position]
    if position == 0 or any(
        plays[position - 1]["about"][k] != play["about"][k]
        for k in ("inning", "halfInning")
    ):
        outs, base = 0, "___"
    else:
        previous = plays[position - 1]
        outs = previous["count"]["outs"]
        base = "".join(str(i) if previous["matchup"].get(k) else "_" for i, k in
                       enumerate(("postOnFirst", "postOnSecond", "postOnThird"), 1))
    result = plays[position - 1]["result"] if position else {"homeScore": 0, "awayScore": 0}
    return outs, base, result["homeScore"], result["awayScore"]


def write_csv(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def audit():
    hash_count = audit_hashes()
    metadata = load_json("2025_metadata.json")
    game = metadata["gameData"]["game"]
    if game["pk"] != 778563 or game["season"] != "2025" or game["type"] != "R":
        raise ValueError("Unexpected sample game/season/type")
    plays = load_json("2025_wp.json")
    official = metadata["liveData"]["plays"]["allPlays"]
    if len(plays) != len(official) or any(
        p["about"]["atBatIndex"] != q["about"]["atBatIndex"]
        or p["matchup"]["batter"]["id"] != q["matchup"]["batter"]["id"]
        for p, q in zip(plays, official, strict=True)
    ):
        raise ValueError("MLB feed/WP play join mismatch")
    box = fg_query("2025_fangraphs.html", "scores/box-score")
    linescore = fg_query("2025_fangraphs.html", "scores/line-score")
    if linescore["totals"]["homeTeam"]["R"] != 1 or linescore["totals"]["awayTeam"]["R"] != 4:
        raise ValueError("FanGraphs game identity mismatch")
    crosswalk, ids = build_crosswalk(metadata, box)
    events = fg_query("2025_fangraphs_play_log.html", "scores/play-log")
    if [int(e["gameSeq"]) for e in events] != list(range(1, 76)):
        raise ValueError("Unexpected sequence: re-review event attribution")
    runner_events = [e for e in events if int(e["gameSeq"]) in RUNNER_ONLY_SEQUENCES]
    if len(runner_events) != 1 or "stolen base" not in runner_events[0]["playDesc"]:
        raise ValueError("Reviewed runner-only event changed")
    batter_events = [e for e in events if int(e["gameSeq"]) not in RUNNER_ONLY_SEQUENCES]
    if len(plays) != len(batter_events):
        raise ValueError("PA count mismatch")
    totals = Counter(p["matchup"]["batter"]["id"] for p in plays)
    for side in ("home", "away"):
        for p in metadata["liveData"]["boxscore"]["teams"][side]["players"].values():
            pa = p.get("stats", {}).get("batting", {}).get("plateAppearances", 0)
            if totals[p["person"]["id"]] != pa:
                raise ValueError("Official player PA total mismatch")
    # The provider's per-player box WPA includes baserunning. Reconcile all
    # events first; exclude the runner event only from the batter-PA table.
    sums = defaultdict(float)
    for e in events:
        sums[int(e["playerId"])] += float(e["wpa"])
    max_box_error = max(abs(sums[int(p["playerid"])] - p["WPA"])
                        for side in ("home", "away") for p in box[side]["bat"] if p["RowType"] == 0)
    if max_box_error > 1e-4:
        raise ValueError("FanGraphs event/box WPA accounting mismatch")
    # Independently verify that we is HOME WP AFTER each event, not batter WP.
    for before, after in zip(events, events[1:]):
        sign = 1 if after["ih"] == "1" else -1
        if not math.isclose(float(after["we"]) - float(before["we"]),
                            sign * float(after["wpa"]), abs_tol=2e-5):
            raise ValueError("FanGraphs home WP/sign continuity failed")
    rows = []
    for position, (p, e) in enumerate(zip(plays, batter_events, strict=True)):
        about = p["about"]
        if not about["isComplete"] or not any(c.get("credit") == "b_pa" for c in p["credits"]):
            raise ValueError("Unexpected non-PA: sample rules need review")
        if crosswalk[int(e["playerId"])] != p["matchup"]["batter"]["id"]:
            raise ValueError("Stable player-ID mismatch")
        if int(e["inning"]) != about["inning"] or (e["ih"] == "0") != about["isTopInning"]:
            raise ValueError("Inning/half mismatch")
        start_seq = 1 if position == 0 else int(batter_events[position - 1]["gameSeq"]) + 1
        first = events[start_seq - 1]
        if (first["inning"], first["ih"]) != (e["inning"], e["ih"]):
            raise ValueError("First PA event inning/half mismatch")
        outs, base, home_score, away_score = pre_pa_state(plays, position)
        # First event's state must match the actual PA start state, so a
        # post-SB LI cannot be silently mistaken for the PA opportunity LI.
        state = (int(first["outs"]), first["base"], first["score"])
        if state != (outs, base, f"{home_score}-{away_score}"):
            raise ValueError("Pre-PA base/out/score mismatch")
        li = number(float(first["li"]), 0, math.inf, "pre-PA LI")
        wpa = number(float(e["wpa"]), -1, 1, "batter WPA")
        sign = -1 if about["isTopInning"] else 1
        wp_after = number(float(e["we"]), 0, 1, "home WP after")
        pre_pa_wp = float(events[start_seq - 2]["we"]) if start_seq > 1 else wp_after - sign * wpa
        rows.append({
            "game_pk": 778563, "season": 2025, "game_type": "R",
            "game_date": metadata["gameData"]["datetime"]["officialDate"],
            "at_bat_index": about["atBatIndex"], "pa_id": f"778563:{about['atBatIndex']}",
            "batter_id": p["matchup"]["batter"]["id"], "fangraphs_batter_id": int(e["playerId"]),
            "batter_name": p["matchup"]["batter"]["fullName"],
            "batting_team_id": metadata["gameData"]["teams"]["away" if about["isTopInning"] else "home"]["id"],
            "inning": about["inning"], "half_inning": about["halfInning"],
            "home_score_before_pa": home_score, "away_score_before_pa": away_score,
            "batting_score_diff_before_pa": sign * (home_score - away_score),
            "outs_before_pa": outs, "bases_before_pa": base,
            "pa_result": p["result"]["eventType"],
            "home_wp_before_pa": pre_pa_wp, "home_wp_after_pa": wp_after,
            "home_wp_before_batter_event": wp_after - sign * wpa,
            "batter_attributed_wpa": wpa,
            "li_before_pa": li, "li_at_batter_event": number(float(e["li"]), 0, math.inf, "batter-event LI"),
            "li_method": "first_event_state_matches_pa_start" if first is not e else "batter_event_state_matches_pa_start",
            "li_source_game_seq": int(first["gameSeq"]),
            "batter_event_game_seq": int(e["gameSeq"]),
            "independent_runner_event_in_pa": first is not e,
            "attribution_source": "FanGraphs reviewed event separation",
            "verification_scope": "single reviewed game; not season-ready",
        })
    savant = list(csv.DictReader((FOLDER / "2025_savant.csv").open()))
    if {r["game_pk"] for r in savant} != {"778563"}:
        raise ValueError("Savant query exceeded fixed game")
    terminal = [r for r in savant if r["events"]]
    keys = [(r["game_pk"], r["at_bat_number"]) for r in terminal]
    if len(keys) != len(set(keys)):
        raise ValueError("Duplicate Savant terminal PA")
    for s in terminal:
        p = plays[int(s["at_bat_number"]) - 1]
        if int(s["batter"]) != p["matchup"]["batter"]["id"] or not p["about"]["isTopInning"]:
            raise ValueError("Savant MLBAM/index/team mismatch")
    report = {
        "scope": "Phase 2 only; diagnostic samples, no season metrics or rankings",
        "primary_sample": {"game_pk": 778563, "date": "2025-03-18", "season": 2025,
                           "away": "Los Angeles Dodgers", "home": "Chicago Cubs", "game_type": "R"},
        "source_response_hashes_checked": hash_count,
        "mlb_completed_pa": len(plays), "official_box_pa_reconciliation": "passed",
        "fangraphs_events": len(events), "separate_runner_events": len(runner_events),
        "verified_batter_pa": len(rows), "game_specific_id_crosswalk_players": len(ids),
        "mlb_missing_li_indices": [p["atBatIndex"] for p in plays if p.get("leverageIndex") is None],
        "fangraphs_missing_pre_pa_li": sum(r["li_before_pa"] is None for r in rows),
        "li_state_reconstruction_pa_indices": [r["at_bat_index"] for r in rows if r["independent_runner_event_in_pa"]],
        "max_fangraphs_event_vs_box_wpa_abs_error": max_box_error,
        "wp_continuity_tolerance": 2e-5,
        "savant_dodgers_pitch_rows": len(savant), "savant_dodgers_completed_pa": len(terminal),
        "savant_li_column_present": any("leverage" in k.lower() or k.lower() == "li" for k in savant[0]),
        "season_scale_ready": False,
        "remaining_gates": ["general event classifier and stable ID crosswalk",
                            "audit other baserunning/combined events and LI state alignment",
                            "representative validation games and season coverage"],
    }
    return rows, ids, runner_events, report


def main():
    rows, ids, runner_events, report = audit()
    write_csv(ROOT / "data/processed/778563_verified_sample.csv", rows)
    write_csv(ROOT / "data/processed/778563_player_id_crosswalk.csv", ids)
    write_csv(ROOT / "data/processed/778563_runner_events.csv", runner_events)
    (ROOT / "reports/source_validation.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
