"""Collect one fixed real game, preserving missing LI and source provenance."""

import argparse
import csv
import hashlib
import json
import math
from datetime import datetime, timezone
from pathlib import Path

import requests

GAME_PK = 745444
URL = f"https://statsapi.mlb.com/api/v1/game/{GAME_PK}/winProbability"
ROOT = Path(__file__).resolve().parents[2]


def number(value, low, high, label):
    if isinstance(value, bool) or not isinstance(value, (float, int)):
        raise ValueError(f"{label}: missing or nonnumeric")
    if not math.isfinite(value) or not low <= value <= high:
        raise ValueError(f"{label}: invalid range")
    return float(value)


def normalize(payload):
    if not isinstance(payload, list) or not payload:
        raise ValueError("Expected nonempty MLB play array")
    rows = []
    seen = set()
    for play in payload:
        about = play["about"]
        index = about["atBatIndex"]
        if type(index) is not int or index < 0 or index in seen:
            raise ValueError("Invalid or duplicate atBatIndex")
        seen.add(index)
        if play.get("atBatIndex") != index:
            raise ValueError("Conflicting atBatIndex")
        # Exclude unfinished plays and runner-only events from the PA table.
        if not about.get("isComplete") or not any(
            c.get("credit") == "b_pa" for c in play.get("credits", [])
        ):
            continue
        half = about["halfInning"]
        if half not in ("top", "bottom"):
            raise ValueError("Invalid halfInning")
        home = number(play["homeTeamWinProbability"], 0, 100, "home WP")
        away = number(play["awayTeamWinProbability"], 0, 100, "away WP")
        if not math.isclose(home + away, 100, abs_tol=1e-8):
            raise ValueError("WP complement mismatch")
        delta = number(play["homeTeamWinProbabilityAdded"], -100, 100, "WPA")
        li = play.get("leverageIndex")
        if li is not None:
            li = number(li, 0, math.inf, "LI")
        rows.append({
            "game_pk": GAME_PK, "at_bat_index": index,
            "inning": about["inning"], "half_inning": half,
            "batter_id": play["matchup"]["batter"]["id"],
            "pitcher_id": play["matchup"]["pitcher"]["id"],
            "event": play["result"]["eventType"],
            "home_wp_after": home / 100,
            "home_wpa": delta / 100,
            "batting_team_event_wpa": (-delta if half == "top" else delta) / 100,
            "batter_attribution_status": "unverified_do_not_use_for_metrics",
            "li": li, "li_status": "missing" if li is None else "observed",
            "contains_runner_event": any(
                r.get("details", {}).get("eventType") != play["result"]["eventType"]
                for r in play.get("runners", [])
            ),
        })
    if not rows:
        raise ValueError("No completed credited PAs")
    return rows


def run(offline=False):
    raw = ROOT / "data/raw/745444_win_probability.json"
    if offline:
        content = raw.read_bytes()
    else:
        response = requests.get(URL, timeout=(10, 40))
        response.raise_for_status()
        content = response.content
    payload = json.loads(content)
    rows = normalize(payload)
    # Validate before writing; missing LI remains null, never replaced by 1.
    for folder in ("data/raw", "data/processed", "reports"):
        (ROOT / folder).mkdir(parents=True, exist_ok=True)
    if not offline:
        raw.write_bytes(content)
        (raw.parent / "745444_provenance.json").write_text(json.dumps({
            "url": URL, "game_pk": GAME_PK,
            "retrieved_at_utc": datetime.now(timezone.utc).isoformat(),
            "sha256": hashlib.sha256(content).hexdigest(),
            "transport": "requests; TLS verification enabled",
        }, indent=2) + "\n")
    with (ROOT / "data/processed/745444_plate_appearances.csv").open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    report = {
        "game_pk": GAME_PK, "source_rows": len(payload), "completed_pa_rows": len(rows),
        "missing_li_indices": [r["at_bat_index"] for r in rows if r["li"] is None],
        "zero_li_indices": [r["at_bat_index"] for r in rows if r["li"] == 0],
        "runner_event_pa_indices": [r["at_bat_index"] for r in rows if r["contains_runner_event"]],
        "schema_range_unique_checks": "passed",
        "formula_validation": "v1.0 documented; implementation deferred to Phase 4",
        "li_timing_and_batter_attribution": "requires cross-source verification",
        "scope": "one fixed game only; no rankings",
    }
    (ROOT / "reports/sample_validation.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--offline", action="store_true", help="Validate cached sample without network")
    run(parser.parse_args().offline)
