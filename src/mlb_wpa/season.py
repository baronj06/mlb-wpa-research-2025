"""Authorized Phase 3: full 2025 collection, quarantine and quality reporting."""

import argparse
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
import hashlib
import json
from pathlib import Path
import threading
import time
import fcntl
from urllib.parse import parse_qs, urlparse

import pandas as pd

from .collection import CachedClient, atomic_json, read_cached_body, utc_now, ensure_scope
from .final_validation import validate_one, parse_query
from .sample import ROOT

VERSION = "season-collection v1.0 / attribution-validation v0.3"
GAME_DIR = ROOT / "data/checkpoints/season_2025/games"
TABLE_DIR = ROOT / "data/processed/season_2025/games"
FORMULA_PATH = ROOT / "config/research_v1.json"
_local = threading.local()
_map_lock = threading.Lock()
_team_map = {}


class RateLimiter:
    def __init__(self, interval=.35, reference_interval=1.0):
        self.interval = interval
        self.reference_interval = reference_interval
        self.lock = threading.Lock()
        self.next_time = {}
        self.requests = 0
        self.reference_blocked_until = 0
        self.reference_pauses = 0

    def __call__(self, url):
        host = urlparse(url).hostname
        with self.lock:
            now = time.monotonic()
            if host == "www.fangraphs.com" and now < self.reference_blocked_until:
                raise RuntimeError("reference_deferred:FanGraphs cooldown after HTTP 429/403")
            slot = max(now, self.next_time.get(host, now))
            interval = self.reference_interval if host == "www.fangraphs.com" else self.interval
            self.next_time[host] = slot + interval
            self.requests += 1
        if slot > now:
            time.sleep(slot - now)

    def block_reference(self, seconds=300):
        with self.lock:
            self.reference_blocked_until = max(self.reference_blocked_until,time.monotonic()+seconds)
            self.reference_pauses += 1


def canonical_manifest(payload):
    chosen = {}
    for day in payload["dates"]:
        for game in day["games"]:
            if game["gameType"] != "R":
                continue
            pk = game["gamePk"]
            old = chosen.get(pk)
            if old is None or game["status"]["abstractGameState"] == "Final":
                chosen[pk] = game
    games = []
    for g in chosen.values():
        games.append({"game_pk": g["gamePk"], "season": 2025, "date": g["officialDate"],
                      "home_team_name": g["teams"]["home"]["team"]["name"],
                      "away_team_name": g["teams"]["away"]["team"]["name"],
                      "dh": 0 if g["doubleHeader"] == "N" else g["gameNumber"],
                      "schedule_status": g["status"]["abstractGameState"],
                      "schedule_home_id": g["teams"]["home"]["team"]["id"]})
    games.sort(key=lambda g: (g["date"], g["game_pk"]))
    return {"scope": "season", "season": 2025, "execution_authorized": True,
            "authorization": "User explicitly authorized Phase 3 full-season collection and conditional rankings.",
            "games": games}


def client_for_thread(limiter, offline):
    if not hasattr(_local, "client"):
        _local.client = CachedClient(compress=True, interval=0, limiter=limiter, offline=offline)
    return _local.client


def official_player_stats(metadata):
    records = []
    fields = {"PA": "plateAppearances", "AB": "atBats", "H": "hits", "HR": "homeRuns",
              "BB": "baseOnBalls", "HBP": "hitByPitch", "SF": "sacFlies"}
    for side in ("home", "away"):
        team = metadata["gameData"]["teams"][side]
        for p in metadata["liveData"]["boxscore"]["teams"][side]["players"].values():
            stats = p.get("stats", {}).get("batting", {})
            if not stats:
                continue
            records.append({"player_id": p["person"]["id"], "player_name": p["person"]["fullName"],
                            "team_id": team["id"], "team": team["name"],
                            **{output: stats.get(source, 0) for output, source in fields.items()}})
    return records


def fallback_rows(game, metadata, reason, wp_plays=None):
    rows = []
    for p in (wp_plays if wp_plays is not None else metadata["liveData"]["plays"]["allPlays"]):
        credits = {c["player"]["id"] for c in p.get("credits", []) if c.get("credit") == "b_pa"}
        if not credits:
            continue
        rows.append({"game_pk": game["game_pk"], "date": game["date"],
                     "at_bat_index": p["about"]["atBatIndex"],
                     "batter_id": next(iter(credits)) if len(credits) == 1 else None,
                     "result": p["result"]["eventType"], "validated": False,
                     "reason": reason, "batter_attributed_wpa": None, "pre_pa_li": None,
                     "reference": "FanGraphs; game-level validation unavailable"})
    return rows


def fingerprint(receipts):
    paths = [ROOT / f"src/mlb_wpa/{n}.py" for n in
             ("season", "collection", "final_validation", "attribution", "validate_attribution", "verify_sources")]
    return hashlib.sha256(json.dumps({"sources": {k: v["sha256"] for k,v in receipts.items()},
          "code": {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in paths}}, sort_keys=True).encode()).hexdigest()


def process_game(game, limiter, offline=False, official_only=False):
    GAME_DIR.mkdir(parents=True,exist_ok=True)
    with (GAME_DIR / f"{game['game_pk']}.lock").open("a") as lock:
        fcntl.flock(lock,fcntl.LOCK_EX)
        return _process_game(game,limiter,offline,official_only)


def _process_game(game, limiter, offline=False, official_only=False):
    pk = game["game_pk"]
    path = GAME_DIR / f"{pk}.json"
    record = json.loads(path.read_text()) if path.exists() else {"game_pk": pk, "sources": {}, "validation_status": "not_run"}
    client = client_for_thread(limiter, offline)
    metadata = None
    wp_plays = []
    def save():
        record["updated_at_utc"] = utc_now()
        atomic_json(path, record)
    def source(name, url, params=None):
        if name in record["sources"]:
            return read_cached_body(client.root, record["sources"][name])
        old_retries=client.retries
        if urlparse(url).hostname == "www.fangraphs.com":client.retries=0
        try:
            body, receipt = client.get(url, params)
        except Exception as exc:
            response=getattr(exc,"response",None)
            if response is not None and response.status_code in (429,403):limiter.block_reference()
            raise
        finally:client.retries=old_retries
        record["sources"][name] = receipt
        record["validation_status"] = "not_run"
        save()
        return body
    table_path = TABLE_DIR / f"{pk}_pas.parquet"
    try:
        metadata = json.loads(source("metadata", f"https://statsapi.mlb.com/api/v1.1/game/{pk}/feed/live"))
        identity = metadata["gameData"]
        if (identity["game"]["pk"], identity["game"]["season"], identity["game"]["type"], identity["datetime"]["officialDate"]) != (pk,"2025","R",game["date"]):
            raise ValueError("MLB game identity mismatch")
        if identity["status"]["abstractGameState"] != "Final":
            raise ValueError("MLB game is not final")
        wp_plays = json.loads(source("wp", f"https://statsapi.mlb.com/api/v1/game/{pk}/winProbability"))
        if official_only and len(record["sources"])<4:
            raise RuntimeError("reference_deferred:MLB sources collected; FanGraphs requests paused")
        # Actual feed metadata resolves doubleheader flags after postponements.
        dh = 0 if identity["game"]["doubleHeader"] == "N" else identity["game"]["gameNumber"]
        if "fangraphs_box" not in record["sources"]:
            home = identity["teams"]["home"]
            with _map_lock:
                fg_home = _team_map.get(home["id"])
                if fg_home is None:
                    name = {"D-backs": "Diamondbacks"}.get(home["teamName"],home["teamName"])
                    old_retries=client.retries;client.retries=0
                    try:
                        _, route = client.get("https://www.fangraphs.com/wins.aspx", {
                            "date": game["date"], "team": name, "dh": dh, "season": 2025})
                    except Exception as exc:
                        response=getattr(exc,"response",None)
                        if response is not None and response.status_code in (429,403):limiter.block_reference()
                        raise
                    finally:client.retries=old_retries
                    fg_home = int(parse_qs(urlparse(route["url"]).query)["hometeamid"][0])
                    _team_map[home["id"]] = fg_home
                    atomic_json(ROOT / "data/checkpoints/season_2025/team_routes.json", _team_map)
            source("fangraphs_box", "https://www.fangraphs.com/api/scores/box-score", {
                "gamedate": game["date"], "hometeamid": fg_home, "dh": dh})
        box_receipt = record["sources"]["fangraphs_box"]
        params = box_receipt.get("params", {})
        fg_home = params.get("hometeamid") or parse_qs(urlparse(box_receipt["url"]).query).get("hometeamid",[None])[0]
        if fg_home is None:
            raise ValueError("Missing FanGraphs team route")
        source("fangraphs_log", "https://www.fangraphs.com/api/play-log/data", {
            "gamedate": game["date"], "teamid": int(fg_home), "dh": dh})
        record["collection_status"] = "raw_complete"
        fp = fingerprint(record["sources"])
        if record.get("validation_fingerprint") == fp and table_path.exists():
            return record["quality"]
        rows, events, quality = validate_one(game, record, client)
        record["validation_status"] = quality["status"]
        record.pop("error", None)
    except Exception as exc:
        reason = f"game_error:{type(exc).__name__}:{exc}"
        record["error"] = reason
        if len(record["sources"]) < 4:
            record["collection_status"] = "failed"
        record["validation_status"] = "blocked"
        rows = fallback_rows(game, metadata, reason, wp_plays) if metadata else []
        events = []
        quality = {"game_pk":pk,"date":game["date"],"status":"blocked","error":reason,
                   "validated_pa":0,"unresolved_cases":[{"reason":reason}],"unmatched_reference_sequences":[]}
    stats = official_player_stats(metadata) if metadata else []
    official_pa = sum(p["PA"] for p in stats) if metadata else None
    names = {p["player_id"]:p["player_name"] for p in stats}
    for row in rows:
        row["season"] = 2025
        row["batter_name"] = names.get(row["batter_id"])
    quality.update({"game_pk":pk,"date":game["date"],"official_pa":official_pa,
                    "represented_pa":len(rows),"validated_pa":sum(bool(r["validated"]) for r in rows),
                    "unresolved_pa":None if official_pa is None else official_pa-sum(bool(r["validated"]) for r in rows),
                    "collection_status":record.get("collection_status","failed")})
    TABLE_DIR.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_parquet(table_path,index=False)
    pd.DataFrame(events).to_parquet(TABLE_DIR / f"{pk}_events.parquet",index=False)
    atomic_json(TABLE_DIR / f"{pk}_official_stats.json", stats)
    record["quality"] = quality
    record["validation_fingerprint"] = fingerprint(record["sources"])
    record["validated_at_utc"] = utc_now()
    record["method_version"] = VERSION
    record["official_sources_complete"] = metadata is not None and "wp" in record["sources"]
    save()
    return quality


def prepare(offline=False):
    client = CachedClient(compress=True,offline=offline)
    raw, receipt = client.get("https://statsapi.mlb.com/api/v1/schedule",{"sportId":1,"season":2025,"gameType":"R"})
    manifest = canonical_manifest(json.loads(raw))
    ensure_scope(manifest, allow_season=True)
    atomic_json(ROOT / "config/season_2025_games.json",manifest)
    atomic_json(ROOT / "reports/season_2025_schedule_provenance.json",receipt)
    GAME_DIR.mkdir(parents=True,exist_ok=True)
    old = ROOT / "data/checkpoints/validation_2025.json"
    if old.exists():
        for pk,record in json.loads(old.read_text())["games"].items():
            path=GAME_DIR/f"{pk}.json"
            if not path.exists():
                atomic_json(path,record)
            meta=json.loads(read_cached_body(client.root,record["sources"]["metadata"]))
            fg=parse_qs(urlparse(record["sources"]["fangraphs_box"]["url"]).query)["hometeamid"][0]
            _team_map[meta["gameData"]["teams"]["home"]["id"]]=int(fg)
    routes=ROOT / "data/checkpoints/season_2025/team_routes.json"
    if routes.exists():
        _team_map.update({int(k):v for k,v in json.loads(routes.read_text()).items()})
    return manifest


def summarize(manifest, qualities, complete):
    frames=[pd.read_parquet(TABLE_DIR/f"{g['game_pk']}_pas.parquet") for g in manifest["games"] if (TABLE_DIR/f"{g['game_pk']}_pas.parquet").exists()]
    frames=[f for f in frames if len(f)]
    data=pd.concat(frames,ignore_index=True) if frames else pd.DataFrame()
    data.to_parquet(ROOT / "data/processed/season_2025_plate_appearances.parquet",index=False)
    stats=[]
    for g in manifest["games"]:
        path=TABLE_DIR/f"{g['game_pk']}_official_stats.json"
        if path.exists():stats.extend(json.loads(path.read_text()))
    stat_frame=pd.DataFrame(stats)
    if len(stat_frame):
        numeric=["PA","AB","H","HR","BB","HBP","SF"]
        players=stat_frame.groupby("player_id")[numeric].sum()
        players["player_name"]=stat_frame.groupby("player_id")["player_name"].last()
        players["teams"]=stat_frame.groupby("player_id")["team"].agg(lambda s:"; ".join(sorted(set(s))))
        players.to_csv(ROOT / "data/processed/season_2025_official_player_stats.csv")
    game_df=pd.DataFrame(qualities)
    game_df.drop(columns=[c for c in game_df if game_df[c].map(lambda x:isinstance(x,(dict,list))).any()],errors="ignore").to_csv(ROOT/"reports/season_2025_quality_by_game.csv",index=False)
    known=sum(q["official_pa"] or 0 for q in qualities)
    valid=sum(q["validated_pa"] for q in qualities)
    unknown=sum(q["official_pa"] is None for q in qualities)
    excluded_rate=(known-valid)/known if known else None
    if len(data):
        data["excluded"]=~data["validated"].fillna(False).astype(bool)
        data.loc[data.excluded].to_csv(ROOT / "data/processed/season_2025_excluded_events.csv",index=False)
        bytype=data.groupby("result").agg(observed_pa=("validated","size"),excluded_pa=("excluded","sum"))
        bytype["missing_rate"]=bytype.excluded_pa/bytype.observed_pa
        bytype.to_csv(ROOT / "reports/season_2025_quality_by_event.csv")
        byreason=data.loc[data.excluded].groupby(["reason","result"],dropna=False).size().rename("excluded_pa")
        byreason.to_csv(ROOT / "reports/season_2025_exclusion_reasons.csv")
        if len(stat_frame):
            pvalid=data.loc[~data.excluded].groupby("batter_id").size()
            players["validated_pa"]=pvalid.reindex(players.index,fill_value=0)
            players["excluded_pa"]=players.PA-players.validated_pa
            players["missing_rate"]=players.excluded_pa/players.PA.replace(0,float("nan"))
            players.to_csv(ROOT/"reports/season_2025_quality_by_player.csv")
    # Fixed before season outcomes: one additional percentage point above the
    # pilot's 4/715 loss, or twice that loss, whichever is larger.
    limit=max(2*4/715,4/715+.01)
    collection_complete=complete and all(q["collection_status"]=="raw_complete" for q in qualities)
    material=excluded_rate is None or excluded_rate>limit or unknown>0 or not complete or not collection_complete
    report={"method_version":VERSION,"generated_at_utc":utc_now(),"expected_games":len(manifest["games"]),
            "processed_games":len(qualities),"collection_complete":collection_complete,
            "unknown_pa_denominator_games":unknown,"official_pa_known":known,"validated_pa":valid,
            "excluded_pa_known":known-valid,"excluded_rate":excluded_rate,"pilot_excluded_rate":4/715,
            "material_worsening_limit":limit,"material_worsening":material,
            "metrics_and_rankings_allowed":not material and complete,
            "formula_config_sha256":hashlib.sha256(FORMULA_PATH.read_bytes()).hexdigest(),
            "game_reports":qualities,"policy":"Null excluded inputs; no model mixing or imputation; official PA denominator."}
    atomic_json(ROOT / "reports/season_2025_data_quality.json",report)
    return report


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--execute",action="store_true")
    parser.add_argument("--offline",action="store_true")
    parser.add_argument("--workers",type=int,default=4)
    parser.add_argument("--official-only",action="store_true",help="Pause external reference requests while completing official MLB sources")
    args=parser.parse_args()
    if not args.execute:
        parser.error("Explicit --execute is required for the user-authorized Phase 3 run")
    manifest=prepare(offline=args.offline)
    limiter=RateLimiter()
    qualities=[]
    start=time.monotonic()
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        jobs={pool.submit(process_game,g,limiter,args.offline,args.official_only):g for g in manifest["games"]}
        for future in as_completed(jobs):
            game=jobs[future]
            try:quality=future.result()
            except Exception as exc:
                quality={"game_pk":game["game_pk"],"date":game["date"],"official_pa":None,"validated_pa":0,
                         "status":"worker_error","collection_status":"failed","error":str(exc)}
            qualities.append(quality)
            if len(qualities)%10==0 or len(qualities)==len(manifest["games"]):
                known=sum(q["official_pa"] or 0 for q in qualities);valid=sum(q["validated_pa"] for q in qualities)
                progress={"completed_games":len(qualities),"expected_games":len(manifest["games"]),
                          "official_pa_known":known,"validated_pa":valid,"network_requests":limiter.requests,
                          "elapsed_seconds":round(time.monotonic()-start),"updated_at_utc":utc_now()}
                atomic_json(ROOT / "reports/season_2025_progress.json",progress)
                print(json.dumps(progress),flush=True)
    qualities.sort(key=lambda q:(q["date"],q["game_pk"]))
    report=summarize(manifest,qualities,len(qualities)==len(manifest["games"]))
    print(json.dumps({k:v for k,v in report.items() if k!="game_reports"},indent=2),flush=True)


if __name__=="__main__":main()
