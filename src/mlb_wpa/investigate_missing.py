"""Reproduce the missing-record investigation from cached single-game probes."""

import json
import hashlib

from .attribution import build_ledger
from .collection import CachedClient, atomic_json, utc_now
from .sample import ROOT


def investigate():
    client = CachedClient(offline=True)
    raw, api_receipt = client.get("https://www.fangraphs.com/api/play-log/data", {
        "gamedate": "2024-03-20", "teamid": 29, "dh": 0})
    pitchfx, pitch_receipt = client.get("https://www.fangraphs.com/api/scores/pitchfx/", {
        "gamedate": "2024-03-20", "teamid": 29, "dh": 0})
    events, pitches = json.loads(raw), json.loads(pitchfx)
    plays = json.loads((ROOT / "data/raw/745444_win_probability.json").read_text())
    units, blockers = build_ledger(plays)
    if blockers:
        raise ValueError("Original sample ledger no longer validates")
    present = {int(e["gameSeq"]) for e in events}
    missing = [u for u in units if u.sequence not in present]
    cases = []
    for u in missing:
        p = next(p for p in plays if p["about"]["atBatIndex"] == u.play_index)
        automatic = [e for e in p["playEvents"] if e.get("details", {}).get("code") == "VP" and e.get("count", {}).get("balls") == 4]
        cases.append({"game_pk": 745444, "reference_sequence": u.sequence, "at_bat_index": u.play_index,
                      "batter_id": u.actor_id, "batter_name": p["matchup"]["batter"]["fullName"],
                      "mlb_result": p["result"]["eventType"], "automatic_ball_walk": bool(automatic),
                      "terminal_description": automatic[-1]["details"]["description"] if automatic else None,
                      "api_batting_wpa_alternative_model": -p["homeTeamWinProbabilityAdded"] / 100,
                      "api_li_timing_unverified": p.get("leverageIndex"),
                      "same_provider_wpa": None, "same_provider_pre_pa_li": None,
                      "replacement_performed": False})
    relevant = sorted({p["PlayDesc"] for p in pitches if "walked from automatic ball" in (p.get("PlayDesc") or "").lower()})
    report = {"investigated_at_utc": utc_now(), "game_pk": 745444, "direct_fangraphs_api_rows": len(events),
              "missing_sequences": [u.sequence for u in missing], "cases": cases,
              "fangraphs_pitchfx_rows": len(pitches), "pitchfx_corresponding_descriptions": relevant,
              "pitchfx_numeric_wpa_or_li_fields": [k for k in pitches[0] if k.lower() in {"wpa", "li", "leverageindex", "winprobability"}],
              "association": "Both omissions are automatic-ball walks ending with pitcher timer violations; backend causation is not proven",
              "same_model_complete_alternative_verified": False,
              "baseball_reference_access": {"url": "https://www.baseball-reference.com/boxes/SDN/SDN202403200.shtml",
                                            "result": "web fetch returned HTTP 403; no data acquired"},
              "retrosheet": "Cached single-game narrative confirms walks but provides no numeric WPA/LI",
              "receipts": {"direct_api": api_receipt, "pitchfx": pitch_receipt},
              "policy": "Keep null. Do not fill from box residuals, pitch RE, or a different WP model."}
    report["method_source_sha256"] = hashlib.sha256((ROOT / "src/mlb_wpa/investigate_missing.py").read_bytes()).hexdigest()
    return report


def main():
    report = investigate()
    atomic_json(ROOT / "reports/missing_fangraphs_investigation.json", report)
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
