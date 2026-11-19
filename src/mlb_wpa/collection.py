"""Cached, resumable public-data collection; season execution is policy-gated."""

import hashlib
import gzip
import json
import os
from pathlib import Path
import tempfile
import time
from datetime import datetime, timezone, date
from urllib.parse import urlparse

import requests

from .sample import ROOT

ALLOWED_HOSTS = {"statsapi.mlb.com", "www.fangraphs.com", "cdn.fangraphs.com",
                 "www.retrosheet.org", "baseballsavant.mlb.com"}
RETRY_STATUS = {408, 429, 500, 502, 503, 504}


def utc_now():
    return datetime.now(timezone.utc).isoformat()


def atomic_json(path, value):
    atomic_bytes(path, (json.dumps(value, indent=2, allow_nan=False) + "\n").encode())


def atomic_bytes(path, content):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(dir=path.parent, delete=False) as handle:
        temporary = Path(handle.name)
        handle.write(content)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


def read_cached_body(root, receipt):
    path = Path(root) / receipt.get("storage_path", f"blobs/{receipt['sha256']}")
    stored = path.read_bytes()
    content = gzip.decompress(stored) if receipt.get("storage_encoding") == "gzip" else stored
    if hashlib.sha256(content).hexdigest() != receipt["sha256"]:
        raise ValueError("Cached response hash mismatch; explicit review required")
    return content


def authorize_public_source(url):
    """Keep existing cache usable; do not automate unsupported live FG access.

    Provider policy reviewed 2026-11-17: blogs.fangraphs.com/contact/.
    Permitted exports need a separate importer and documented provenance.
    """
    if (urlparse(url).hostname or '').endswith('fangraphs.com'):
        raise PermissionError('Live FanGraphs automation disabled by provider policy; use existing cache or a permitted export')


class CachedClient:
    """Content-addressed snapshots, verified cache, bounded retries and pacing.

    Test injection of session/sleep avoids real HTTP and delays in unit tests.
    Responses are never trusted solely because the HTTP status is 200.
    """

    def __init__(self, root=None, session=None, sleep=time.sleep, retries=2, interval=.5, offline=False, compress=False, limiter=None,
                 network_authorizer=authorize_public_source):
        self.root = Path(root or ROOT / "data/raw/final_validation")
        self.session = session or requests.Session()
        self.sleep, self.retries, self.interval, self.offline = sleep, retries, interval, offline
        self.network_requests = 0
        self.compress = compress
        self.limiter = limiter
        self.network_authorizer = network_authorizer

    @staticmethod
    def key(url, params):
        request = {"method": "GET", "url": url, "params": params or {}}
        return hashlib.sha256(json.dumps(request, sort_keys=True).encode()).hexdigest()

    def get(self, url, params=None, refresh=False):
        if urlparse(url).scheme != "https" or urlparse(url).hostname not in ALLOWED_HOSTS:
            raise ValueError("Only allowlisted HTTPS public sources may be collected")
        key = self.key(url, params)
        pointer = self.root / "requests" / f"{key}.json"
        if pointer.exists() and not refresh:
            manifest = json.loads(pointer.read_text())
            content = read_cached_body(self.root, manifest)
            return content, manifest
        if self.offline:
            raise FileNotFoundError(f"No verified cached response: {key}")
        attempts = []
        for attempt in range(self.retries + 1):
            self.sleep(self.interval)
            if self.limiter is not None:
                self.limiter(url)
            self.network_authorizer(url)
            self.network_requests += 1
            try:
                response = self.session.get(url, params=params, timeout=(10, 40))
                attempts.append({"attempt": attempt + 1, "status": response.status_code, "time": utc_now(),
                                 "retry_after": response.headers.get("Retry-After")})
                if response.status_code in RETRY_STATUS and attempt < self.retries:
                    retry_after = response.headers.get("Retry-After", "")
                    delay = float(retry_after) if retry_after.isdigit() else 2 ** attempt
                    self.sleep(min(delay, 30))
                    continue
                response.raise_for_status()
                if urlparse(response.url).hostname not in ALLOWED_HOSTS:
                    raise ValueError("Unexpected source redirect host")
                content = response.content
                if len(content) > 10_000_000:
                    raise ValueError("Single-response sample size limit exceeded")
                digest = hashlib.sha256(content).hexdigest()
                manifest = {"request_key": key, "requested_url": url, "params": params or {},
                            "url": response.url, "retrieved_at_utc": utc_now(),
                            "status_code": response.status_code, "content_type": response.headers.get("Content-Type"),
                            "sha256": digest, "bytes": len(content), "attempts": attempts,
                            "tls_verification": True}
                if self.compress:
                    manifest["storage_encoding"] = "gzip"
                    manifest["storage_path"] = f"blobs/{digest}.gz"
                    atomic_bytes(self.root / manifest["storage_path"], gzip.compress(content, mtime=0))
                else:
                    atomic_bytes(self.root / "blobs" / digest, content)
                # Immutable response manifest remains even when the request pointer is refreshed.
                snapshot = f"manifests/{key}_{digest}_{time.time_ns()}.json"
                manifest["snapshot_manifest"] = snapshot
                atomic_json(self.root / snapshot, manifest)
                atomic_json(pointer, manifest)
                return content, manifest
            except (requests.Timeout, requests.ConnectionError) as exc:
                attempts.append({"attempt": attempt + 1, "error": type(exc).__name__, "time": utc_now()})
                if attempt < self.retries:
                    self.sleep(2 ** attempt)
                    continue
                self.record_failure(key, url, params, attempts, exc)
                raise
            except (requests.HTTPError, ValueError) as exc:
                self.record_failure(key, url, params, attempts, exc)
                raise
        raise RuntimeError("Retry loop exhausted")

    def record_failure(self, key, url, params, attempts, exc):
        atomic_json(self.root / "failures" / f"{key}_{time.time_ns()}.json", {
            "requested_url": url, "params": params or {}, "attempts": attempts,
            "error_type": type(exc).__name__, "error": str(exc), "time": utc_now()})


def season_plan(season=2025):
    if season != 2025:
        raise ValueError("Only the proposed primary 2025 season is configured")
    return {"scope": "season", "season": season, "execution_authorized": False,
            "schedule_request": {"url": "https://statsapi.mlb.com/api/v1/schedule",
                                 "params": {"sportId": 1, "season": season, "gameType": "R"}},
            "games": [], "state": "plan_only_no_network",
            "requirements": ["explicit subsequent authorization", "validation readiness gate", "complete missingness policy"]}


def manifest_from_schedule(payload, season=2025):
    """Prepare future season jobs from an ALREADY supplied schedule, no HTTP.

    Doubleheaders are marked for route review, not silently mapped to a
    potentially wrong provider game number. Their raw collection stays gated.
    """
    plan = season_plan(season)
    seen = set()
    for day in payload["dates"]:
        for game in day["games"]:
            if game["gameType"] != "R":
                continue
            if str(game.get("season", season)) != str(season):
                raise ValueError("Schedule contains a different regular season")
            if game["gamePk"] in seen:
                continue  # rescheduled/suspended schedule listings can repeat
            seen.add(game["gamePk"])
            is_doubleheader = game["doubleHeader"] != "N"
            plan["games"].append({"game_pk": game["gamePk"], "season": season, "date": game["officialDate"],
                                  "home_team_name": game["teams"]["home"]["team"]["name"],
                                  "away_team_name": game["teams"]["away"]["team"]["name"],
                                  "dh": None if is_doubleheader else 0,
                                  "route_requires_review": is_doubleheader,
                                  "schedule_status": game["status"]["abstractGameState"]})
    return plan


def ensure_scope(manifest, allow_season=False):
    games = manifest.get("games", [])
    if any(type(g["game_pk"]) is not int or g["game_pk"] <= 0 for g in games):
        raise ValueError("Game IDs must be positive integers")
    if len({g["game_pk"] for g in games}) != len(games):
        raise ValueError("Duplicate game in collection manifest")
    if any(g.get("season", 2025) != 2025 for g in games):
        raise ValueError("Collection manifest must use the 2025 primary season")
    if any(date.fromisoformat(g["date"]).year != 2025 for g in games):
        raise ValueError("Game dates must be in the configured primary season")
    if manifest.get("scope") == "validation_sample" and len(games) <= 10:
        return
    if not (manifest.get("scope") == "season" and allow_season):
        raise PermissionError("Full-season execution is disabled; at most 10 validation games are authorized")


class GameCollector:
    """Checkpoint each source separately and resume without re-downloading it.

    Raw collection complete != scientifically validated. A validation failure
    is recorded as quarantine, never promoted to an eligible analysis dataset.
    """

    def __init__(self, client, checkpoint):
        self.client, self.path = client, Path(checkpoint)
        self.state = json.loads(self.path.read_text()) if self.path.exists() else {"version": 1, "games": {}}

    def collect(self, manifest, allow_season=False, refresh=False):
        ensure_scope(manifest, allow_season)
        self.state["manifest_sha256"] = hashlib.sha256(json.dumps(manifest, sort_keys=True).encode()).hexdigest()
        self.state["scope"] = manifest["scope"]
        for game in manifest["games"]:
            pk = game["game_pk"]
            record = self.state["games"].setdefault(str(pk), {"sources": {}, "validation_status": "not_run"})
            sources = {
                "metadata": (f"https://statsapi.mlb.com/api/v1.1/game/{pk}/feed/live", {}),
                "wp": (f"https://statsapi.mlb.com/api/v1/game/{pk}/winProbability", {}),
                "fangraphs_box": ("https://www.fangraphs.com/wins.aspx", {
                    "date": game["date"], "team": game["home_team_name"],
                    "dh": game.get("dh", 0), "season": 2025}),
            }
            try:
                if game.get("route_requires_review") or game.get("dh") is None:
                    raise ValueError("Provider doubleheader route requires explicit verification")
                for name, (url, params) in sources.items():
                    if name == "fangraphs_box":
                        metadata = json.loads(read_cached_body(self.client.root, record["sources"]["metadata"]))
                        identity = metadata["gameData"]
                        if (identity["game"]["pk"], identity["game"]["season"], identity["game"]["type"],
                            identity["datetime"]["officialDate"]) != (pk, "2025", "R", game["date"]):
                            raise ValueError("MLB game identity/season/type/date mismatch")
                        if identity["status"]["abstractGameState"] != "Final":
                            raise ValueError("Game is not final; snapshot remains in quarantine")
                        team_name = identity["teams"]["home"]["teamName"]
                        # MLB's display label differs from this legacy FG route.
                        params = {**params, "team": {"D-backs": "Diamondbacks"}.get(team_name, team_name)}
                    _, receipt = self.client.get(url, params, refresh=refresh)
                    if record["sources"].get(name, {}).get("sha256") != receipt["sha256"]:
                        record["validation_status"] = "not_run"
                    record["sources"][name] = receipt
                    self.save()
                resolved = urlparse(record["sources"]["fangraphs_box"]["url"])
                from urllib.parse import parse_qs
                fg_home = parse_qs(resolved.query).get("hometeamid", [None])[0]
                if fg_home is None:
                    raise ValueError("FanGraphs home team ID missing from resolved game URL")
                _, receipt = self.client.get("https://www.fangraphs.com/scores/play-log", {
                    "date": game["date"], "hometeamid": fg_home, "dh": game.get("dh", 0)}, refresh=refresh)
                if record["sources"].get("fangraphs_log", {}).get("sha256") != receipt["sha256"]:
                    record["validation_status"] = "not_run"
                record["sources"]["fangraphs_log"] = receipt
                record["collection_status"] = "raw_complete"
                record.pop("error", None)
            except (requests.RequestException, ValueError, KeyError, TypeError, OSError) as exc:
                record["collection_status"] = "failed"
                record["validation_status"] = "not_run"
                record["error"] = {"type": type(exc).__name__, "message": str(exc), "time": utc_now()}
            self.save()
        return self.state

    def save(self):
        self.state["updated_at_utc"] = utc_now()
        atomic_json(self.path, self.state)


def main():
    import argparse
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path)
    parser.add_argument("--offline", action="store_true")
    parser.add_argument("--execute-sample", action="store_true")
    parser.add_argument("--refresh-sources", action="store_true", help="Snapshot fresh responses and invalidate changed-source validation")
    args = parser.parse_args()
    if not args.execute_sample:
        print(json.dumps(season_plan(), indent=2))
        return
    if args.manifest is None:
        parser.error("--execute-sample requires an explicit --manifest")
    manifest = json.loads(args.manifest.read_text())
    ensure_scope(manifest)  # There is deliberately no CLI season-execution override.
    collector = GameCollector(CachedClient(offline=args.offline), ROOT / "data/checkpoints/validation_2025.json")
    state = collector.collect(manifest, refresh=args.refresh_sources)
    print(json.dumps({k: v.get("collection_status") for k, v in state["games"].items()}, indent=2))


if __name__ == "__main__":
    main()
