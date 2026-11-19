"""Predeclared, bounded selection of ten additional 2025 games, not a season pull."""

from datetime import date, timedelta
import json

from .collection import CachedClient, atomic_json
from .sample import ROOT

# Spread across dates/teams; selection is fixed before inspecting validation
# outcomes. Choose the earliest final non-doubleheader game in a 3-day window.
TARGETS = [("2025-04-01", 119), ("2025-04-15", 135), ("2025-05-01", 147),
           ("2025-05-15", 121), ("2025-06-01", 141), ("2025-06-15", 136),
           ("2025-07-01", 137), ("2025-08-01", 112), ("2025-09-01", 113),
           ("2025-09-15", 144)]


def main():
    client = CachedClient()
    games, receipts = [], []
    for start, team in TARGETS:
        end = (date.fromisoformat(start) + timedelta(days=2)).isoformat()
        content, receipt = client.get("https://statsapi.mlb.com/api/v1/schedule", {
            "sportId": 1, "teamId": team, "startDate": start, "endDate": end, "gameType": "R"})
        receipts.append(receipt)
        eligible = [g for d in json.loads(content)["dates"] for g in d["games"]
                    if g["gameType"] == "R" and g["status"]["abstractGameState"] == "Final" and g["doubleHeader"] == "N"]
        if not eligible:
            raise ValueError(f"No eligible game in predeclared window: {start}, {team}")
        g = min(eligible, key=lambda g: (g["officialDate"], g["gamePk"]))
        games.append({"game_pk": g["gamePk"], "season": 2025, "date": g["officialDate"],
                      "home_team_name": g["teams"]["home"]["team"]["name"],
                      "away_team_name": g["teams"]["away"]["team"]["name"], "dh": 0,
                      "selection_target": {"start": start, "team_id": team}})
        print(games[-1], flush=True)
    atomic_json(ROOT / "config/validation_2025_games.json", {
        "scope": "validation_sample", "selection": "predeclared team/date windows; earliest final non-DH game",
        "previous_samples_excluded": [745444, 778563], "games": games})
    atomic_json(ROOT / "reports/validation_game_selection_provenance.json", {"requests": receipts})


if __name__ == "__main__":
    main()
