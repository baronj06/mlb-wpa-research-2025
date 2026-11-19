"""Fixed, bounded source checks; never queries a full season."""

import argparse
import hashlib
import json
from datetime import datetime, timezone

import requests

from .sample import ROOT

# Every request is restricted to a named game or one team's single game date.
PROBES = {
    "2024_fangraphs": ("https://www.fangraphs.com/wins.aspx", {
        "date": "2024-03-20", "team": "Dodgers", "dh": "0", "season": "2024",
    }, ".html"),
    "2024_fangraphs_play_log": ("https://www.fangraphs.com/scores/play-log", {
        "date": "2024-03-20", "hometeamid": "29", "dh": "0",
    }, ".html"),
    "2025_savant_both_teams": ("https://baseballsavant.mlb.com/statcast_search/csv", {
        "all": "true", "type": "details", "player_type": "batter",
        "game_date_gt": "2025-03-18", "game_date_lt": "2025-03-18",
        "hfGT": "R|", "group_by": "name",
    }, ".csv"),
    "2024_retrosheet": ("https://www.retrosheet.org/boxesetc/2024/B03200SDN2024.htm", {}, ".html"),
    "2025_retrosheet": ("https://www.retrosheet.org/boxesetc/2025/B03180CHN2025.htm", {}, ".html"),
    "2025_fangraphs_play_log": ("https://www.fangraphs.com/scores/play-log", {
        "date": "2025-03-18", "hometeamid": "17", "dh": "0",
    }, ".html"),
    "2024_metadata": ("https://statsapi.mlb.com/api/v1.1/game/745444/feed/live", {}, ".json"),
    "2025_metadata": ("https://statsapi.mlb.com/api/v1.1/game/778563/feed/live", {}, ".json"),
    "2025_wp": ("https://statsapi.mlb.com/api/v1/game/778563/winProbability", {}, ".json"),
    "2025_savant": ("https://baseballsavant.mlb.com/statcast_search/csv", {
        "all": "true", "type": "details", "player_type": "batter",
        "game_date_gt": "2025-03-18", "game_date_lt": "2025-03-18",
        "team": "LAD", "hfGT": "R|", "group_by": "name",
    }, ".csv"),
    "2025_fangraphs": ("https://www.fangraphs.com/wins.aspx", {
        "date": "2025-03-18", "team": "Dodgers", "dh": "0", "season": "2025",
    }, ".html"),
}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source", choices=PROBES)
    args = parser.parse_args()
    url, params, extension = PROBES[args.source]
    response = requests.get(url, params=params, timeout=(10, 45))
    folder = ROOT / "data/raw/source_probes"
    folder.mkdir(parents=True, exist_ok=True)
    path = folder / (args.source + extension)
    path.write_bytes(response.content)
    manifest = {
        "url": response.url, "requested_url": url, "params": params,
        "retrieved_at_utc": datetime.now(timezone.utc).isoformat(),
        "status_code": response.status_code, "content_type": response.headers.get("Content-Type"),
        "sha256": hashlib.sha256(response.content).hexdigest(),
        "bytes": len(response.content), "path": str(path.relative_to(ROOT)),
        "transport": "requests; TLS verification enabled",
    }
    (folder / (args.source + "_provenance.json")).write_text(json.dumps(manifest, indent=2) + "\n")
    print(json.dumps(manifest, indent=2))
    response.raise_for_status()


if __name__ == "__main__":
    main()
