"""Respectful reference-source resume, serialized behind official collection."""

import json
import time
import argparse

from .collection import atomic_json, utc_now
from .sample import ROOT
from .season import prepare, process_game, RateLimiter, GAME_DIR, summarize


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--previous-provider-pauses',type=int,default=0)
    parser.add_argument('--initial-cooldown',type=int,default=0)
    args=parser.parse_args()
    manifest=prepare()
    limiter=RateLimiter(reference_interval=1.0)
    if args.initial_cooldown:limiter.block_reference(args.initial_cooldown)
    failures=0
    completed=0
    for game in manifest["games"]:
        path=GAME_DIR/f"{game['game_pk']}.json"
        while True:
            record=json.loads(path.read_text()) if path.exists() else {}
            if record.get("official_sources_complete"):
                break
            time.sleep(5)
        while True:
            remaining=limiter.reference_blocked_until-time.monotonic()
            if remaining>0:
                atomic_json(ROOT/'reports/season_2025_reference_progress.json',{
                    'completed_games':completed,'expected_games':len(manifest['games']),
                    'status':'provider_cooldown','cooldown_seconds_remaining':round(remaining),
                    'network_requests':limiter.requests,'updated_at_utc':utc_now()})
                time.sleep(min(20,remaining))
                continue
            quality=process_game(game,limiter)
            record=json.loads(path.read_text())
            error=record.get('error','')
            if '403' in error:
                print('Reference access blocked (403); stopping reference requests.',flush=True)
                atomic_json(ROOT/'reports/season_2025_reference_stop.json',{'game_pk':game['game_pk'],'error':error,'time':utc_now()})
                return
            if '429' in error or 'FanGraphs cooldown' in error:
                failures+=1
                if failures>=3 or limiter.reference_pauses+args.previous_provider_pauses>=3:
                    print('Three separated reference attempts failed; preserving checkpoint.',flush=True)
                    atomic_json(ROOT/'reports/season_2025_reference_stop.json',{'game_pk':game['game_pk'],'error':error,'time':utc_now()})
                    return
                continue
            failures=0
            break
        completed+=1
        if completed%10==0:
            progress={'completed_games':completed,'expected_games':len(manifest['games']),
                      'status':'collecting','network_requests':limiter.requests,'updated_at_utc':utc_now()}
            atomic_json(ROOT/'reports/season_2025_reference_progress.json',progress)
            print(json.dumps(progress),flush=True)
    qualities=[json.loads((GAME_DIR/f"{g['game_pk']}.json").read_text())['quality'] for g in manifest['games']]
    report=summarize(manifest,qualities,True)
    print(json.dumps({k:v for k,v in report.items() if k!='game_reports'},indent=2),flush=True)


if __name__=='__main__':main()
