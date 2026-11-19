"""Conservative event ledger for source validation, independent of EOWS.

MLB scoring codes identify independent actions; FanGraphs supplies observed
event WPA/LI. Never estimate a missing reference value or split a simultaneous
batting/runner outcome by arithmetic guesswork.
"""

from dataclasses import dataclass, replace
import math
import re

from .sample import number

PA_TYPES = {
    "single", "double", "triple", "home_run", "walk", "intent_walk",
    "hit_by_pitch", "strikeout", "strikeout_double_play", "field_out",
    "force_out", "grounded_into_double_play", "double_play", "triple_play",
    "sac_fly", "sac_bunt", "sac_fly_double_play", "sac_bunt_double_play",
    "fielders_choice", "fielders_choice_out", "field_error", "catcher_interf",
}
RUNNER_TYPES = {
    "stolen_base_2b", "stolen_base_3b", "stolen_base_home",
    "caught_stealing_2b", "caught_stealing_3b", "caught_stealing_home",
    "pickoff_1b", "pickoff_2b", "pickoff_3b", "pickoff_caught_stealing_2b",
    "pickoff_caught_stealing_3b", "pickoff_caught_stealing_home",
    "wild_pitch", "passed_ball", "balk", "defensive_indiff", "other_out",
    "pickoff_error_1b", "pickoff_error_2b", "pickoff_error_3b",
}
ADMIN_TYPES = {
    "game_advisory", "batter_timeout", "mound_visit", "pitching_substitution",
    "defensive_substitution", "offensive_substitution", "defensive_switch",
    "pitcher_step_off", "umpire_substitution", "ejection", "injury",
}


@dataclass(frozen=True)
class State:
    inning: int
    half: str
    outs: int
    bases: tuple  # runner MLBAM IDs on first, second, third
    home_score: int
    away_score: int

    @property
    def mask(self):
        return "".join(str(i) if runner is not None else "_" for i, runner in enumerate(self.bases, 1))


@dataclass(frozen=True)
class Unit:
    sequence: int
    play_index: int
    action_index: int
    kind: str
    event_type: str
    actor_id: int
    before: State


def start_state(plays, position):
    """Actual PA/runner-only sequence start, reset at half-inning boundaries."""
    p = plays[position]
    about = p["about"]
    previous = plays[position - 1] if position else None
    same_half = previous and all(previous["about"][k] == about[k] for k in ("inning", "halfInning"))
    bases = tuple(previous["matchup"].get(k, {}).get("id") for k in
                  ("postOnFirst", "postOnSecond", "postOnThird")) if same_half else (None,) * 3
    score = previous["result"] if previous else {"homeScore": 0, "awayScore": 0}
    return State(about["inning"], about["halfInning"], previous["count"]["outs"] if same_half else 0,
                 bases, score["homeScore"], score["awayScore"])


def move_runners(state, runners):
    """Apply simultaneous runner movements in two passes to avoid overwriting."""
    bases = list(state.bases)
    outs, home, away = state.outs, state.home_score, state.away_score
    for r in runners:
        move, details = r["movement"], r["details"]
        start = move.get("start")
        if start in ("1B", "2B", "3B"):
            index = int(start[0]) - 1
            if bases[index] != details["runner"]["id"]:
                raise ValueError("runner_start_identity_mismatch")
            bases[index] = None
    for r in runners:
        move, details = r["movement"], r["details"]
        if move.get("isOut"):
            outs += 1
        elif move.get("end") in ("1B", "2B", "3B"):
            index = int(move["end"][0]) - 1
            if bases[index] is not None:
                raise ValueError("runner_base_collision")
            bases[index] = details["runner"]["id"]
        elif move.get("end") == "score" or details.get("isScoringEvent"):
            if state.half == "top":
                away += 1
            else:
                home += 1
    if outs > 3:
        raise ValueError("more_than_three_outs")
    return replace(state, bases=tuple(bases), outs=outs, home_score=home, away_score=away)


def build_ledger(plays, initial_states=None, *, reviewed_rules=False):
    """Return ordered scoring units and blockers; no hardcoded game sequences.

    A blocker's presence quarantines the game from reference sequence alignment:
    an unsupported action could otherwise shift every later event's identity.
    """
    units, blockers = [], []
    seen = set()
    for pos, p in enumerate(plays):
        index = p["about"]["atBatIndex"]
        if index in seen:
            raise ValueError("duplicate_mlb_play_index")
        seen.add(index)
        if not p["about"].get("isComplete"):
            blockers.append({"play_index": index, "reason": "incomplete_play"})
            continue
        credited = {c["player"]["id"] for c in p.get("credits", []) if c.get("credit") == "b_pa"}
        is_pa = bool(credited)
        event_type = p["result"]["eventType"]
        if len(credited) > 1 or (is_pa and event_type not in PA_TYPES) or (not is_pa and event_type not in RUNNER_TYPES):
            blockers.append({"play_index": index, "reason": "unsupported_pa_or_credit", "event_type": event_type})
            continue
        if is_pa and next(iter(credited)) != p["matchup"]["batter"]["id"]:
            blockers.append({"play_index": index, "reason": "mid_pa_batter_credit_requires_review"})
            continue
        groups = {}
        for runner in p.get("runners", []):
            groups.setdefault(runner["details"]["playIndex"], []).append(runner)
        # For no-pitch PAs, the explicit result action supplies the ending index.
        endings = [e["index"] for e in p["playEvents"] if e.get("isPitch") or
                   e.get("details", {}).get("eventType") == event_type]
        if not endings and reviewed_rules:
            terminal_index = intentional_walk_terminal(p)
            if terminal_index is not None:
                endings = [terminal_index]
        if not endings:
            blockers.append({"play_index": index, "reason": "missing_terminal_event_index"})
            continue
        terminal = max(endings)
        state = initial_states[index] if initial_states is not None else start_state(plays, pos)
        for e in sorted(p["playEvents"], key=lambda x: x["index"]):
            if reviewed_rules and e.get('details',{}).get('eventType') == 'offensive_substitution' and e.get('position',{}).get('code') == '12':
                base=e.get('base');old=e.get('replacedPlayer',{}).get('id');new=e.get('player',{}).get('id')
                if base not in (1,2,3) or old is None or new is None or state.bases[base-1] != old:
                    blockers.append({'play_index':index,'reason':'pinch_runner_replacement_unverified'})
                else:
                    bases=list(state.bases);bases[base-1]=new
                    state=replace(state,bases=tuple(bases))
            action_index = e["index"]
            runners = groups.get(action_index, [])
            runner_codes = {r["details"].get("eventType") for r in runners}
            action_code = e.get("details", {}).get("eventType")
            independent = (runner_codes | {action_code}) & RUNNER_TYPES
            # Recorded outs/advances on the same batted-ball ending belong to
            # that provider's PA outcome, rather than an independent action.
            if is_pa and action_index == terminal and event_type not in {
                "walk", "intent_walk", "hit_by_pitch", "strikeout", "strikeout_double_play", "catcher_interf"
            }:
                independent -= {"other_out"}
            # A fielding error on the batted-ball ending is part of the
            # provider-attributed PA outcome, not a standalone runner action.
            # A preterminal error remains unresolved and quarantined.
            combined = {"error", "other_advance", "other_out"} if is_pa and action_index == terminal else set()
            if independent and len({r["details"]["runner"]["id"] for r in runners}) == 1:
                combined |= {"error"}  # reviewed SB + throwing-error chain
            unknown_runner = runner_codes - RUNNER_TYPES - {event_type} - combined
            if unknown_runner:
                blockers.append({"play_index": index, "reason": "unknown_runner_code", "codes": sorted(str(x) for x in unknown_runner)})
            foul_error = action_code == "error" and not runners and "dropped foul" in e.get("details", {}).get("description", "").lower()
            placed_runner = (action_code == "runner_placed" and state.inning >= 10 and state.outs == 0
                             and state.bases[1] == e.get("player", {}).get("id") and state.bases[1] is not None)
            if action_code and action_code not in PA_TYPES | RUNNER_TYPES | ADMIN_TYPES and not foul_error and not placed_runner:
                blockers.append({"play_index": index, "reason": "unknown_action_code", "event_type": action_code})
            if independent:
                if is_pa and action_index >= terminal:
                    reason = "simultaneous_terminal_runner_and_batter_event" if action_index == terminal else "postterminal_runner_action_requires_review"
                    blockers.append({"play_index": index, "reason": reason})
                    continue
                actors = {r["details"]["runner"]["id"] for r in runners}
                joint_advance = (reviewed_rules and len(independent)==1
                    and independent <= {'wild_pitch','passed_ball'}
                    and action_code in independent and runner_codes == independent
                    and e.get('player',{}).get('id') in actors)
                if len(independent) != 1 or (len(actors) != 1 and not joint_advance):
                    blockers.append({"play_index": index, "reason": "multi_runner_or_missing_credit_requires_review"})
                    continue
                actor=e['player']['id'] if joint_advance else next(iter(actors))
                units.append(Unit(len(units) + 1, index, action_index, "runner", next(iter(independent)), actor, state))
                try:
                    if len(runners) > 1 and len(actors) == 1:
                        # One runner may have linked SB/error legs at one
                        # action index. Preserve the complete observed move;
                        # never assign separate guessed WPA to the error leg.
                        for first, second in zip(runners, runners[1:]):
                            if first["movement"].get("end") != second["movement"].get("start"):
                                raise ValueError("unverified_runner_movement_chain")
                        collapsed = {**runners[-1], "movement": {**runners[-1]["movement"], "start": runners[0]["movement"].get("start")}}
                        state = move_runners(state, [collapsed])
                    else:
                        state = move_runners(state, runners)
                except ValueError as exc:
                    blockers.append({"play_index": index, "reason": str(exc)})
            elif runners and action_index < terminal:
                blockers.append({"play_index": index, "reason": "unclassified_preterminal_runner_movement"})
        if is_pa:
            units.append(Unit(len(units) + 1, index, terminal, "batter", event_type, next(iter(credited)), state))
    return units, blockers


def intentional_walk_terminal(play):
    """Locate four recorded automatic intentional balls; never invent an event."""
    if play.get('result',{}).get('eventType') != 'intent_walk':return None
    balls=[e for e in play.get('playEvents',[]) if e.get('details',{}).get('code') == 'VB'
           and e.get('details',{}).get('isBall') is True and not e.get('isPitch')]
    balls.sort(key=lambda e:e['index'])
    if len(balls)!=4 or [e.get('count',{}).get('balls') for e in balls] != [1,2,3,4]:return None
    index=balls[-1]['index']
    if not any(r['details'].get('eventType')=='intent_walk' and r['details'].get('playIndex')==index
               and r['details']['runner']['id']==play['matchup']['batter']['id'] for r in play.get('runners',[])):return None
    return index


def state_matches(unit, reference):
    s = unit.before
    return (int(reference["inning"]), reference["ih"], int(reference["outs"]), reference["base"], reference["score"]) == (
        s.inning, "0" if s.half == "top" else "1", s.outs, s.mask, f"{s.home_score}-{s.away_score}")


def match_reference(units, events, crosswalk):
    """Match exact scoring order, stable player IDs and complete before-state.

    FanGraphs gameSeq is checked, never renumbered to close source gaps.
    The numbering convention is empirical in the reviewed games; a state/ID
    disagreement is an explicit blocker, not an invitation for fuzzy matching.
    """
    by_sequence = {}
    for e in events:
        sequence = int(e["gameSeq"])
        if sequence in by_sequence:
            raise ValueError("duplicate_reference_sequence")
        by_sequence[sequence] = e
    expected = {u.sequence for u in units}
    if set(by_sequence) - expected:
        raise ValueError("unexpected_reference_sequence")
    result = {}
    runner_labels = {
        "wild_pitch": r"wild pitch", "passed_ball": r"passed ball", "balk": r"balk",
        "defensive_indiff": r"indifference", "other_out": r"out",
    }
    for unit in units:
        e = by_sequence.get(unit.sequence)
        if e is None:
            result[unit.sequence] = {"status": "missing_reference_event", "wpa": None, "li": None}
            continue
        if crosswalk.get(int(e["playerId"])) != unit.actor_id:
            raise ValueError("reference_actor_identity_mismatch")
        if not state_matches(unit, e):
            raise ValueError("reference_pre_event_state_mismatch")
        if unit.kind == "runner":
            pattern = runner_labels.get(unit.event_type)
            if pattern is None:
                pattern = r"stolen base" if unit.event_type.startswith("stolen_base") else r"caught stealing|picked off|pickoff"
            if not re.search(pattern, e["playDesc"], re.I):
                raise ValueError("runner_reference_event_type_mismatch")
        # The reference description is corroboration, never the sole classifier.
        wpa = None if e.get("wpa") in (None, "") else number(float(e["wpa"]), -1, 1, "reference WPA")
        li = None if e.get("li") in (None, "") else number(float(e["li"]), 0, math.inf, "reference LI")
        wp = None if e.get("we") in (None, "") else number(float(e["we"]), 0, 1, "reference WP")
        result[unit.sequence] = {"status": "matched" if wpa is not None and li is not None and wp is not None else "missing_reference_value",
                                 "wpa": wpa, "li": li, "home_wp_after": wp}
    return result


def pa_records(game_pk, units, matches):
    """Preserve independent-event WPA, batter WPA and opportunity LI separately."""
    groups = {}
    for u in units:
        groups.setdefault(u.play_index, []).append(u)
    records = []
    for index, group in groups.items():
        batters = [u for u in group if u.kind == "batter"]
        if not batters:
            continue  # Caught stealing ending an inning is not a completed PA.
        batter, first = batters[0], group[0]
        b, initial = matches[batter.sequence], matches[first.sequence]
        wpas = [matches[u.sequence]["wpa"] for u in group]
        reason = []
        if b["wpa"] is None:
            reason.append("missing_batter_wpa")
        if initial["li"] is None:
            reason.append("missing_pre_pa_li")
        if any(matches[u.sequence]["status"] != "matched" for u in group):
            reason.append("incomplete_reference_validation")
        sign = -1 if first.before.half == "top" else 1
        after = initial.get("home_wp_after")
        before_wp = after - sign * initial["wpa"] if after is not None and initial["wpa"] is not None else None
        records.append({
            "game_pk": game_pk, "at_bat_index": index, "batter_id": batter.actor_id,
            "batter_event_sequence": batter.sequence, "pre_pa_li_sequence": first.sequence,
            "batter_attributed_wpa": b["wpa"], "pre_pa_li": initial["li"],
            "li_at_batter_event": b["li"], "independent_event_count": len(group) - 1,
            "runner_wpa": sum(wpas[:-1]) if all(x is not None for x in wpas[:-1]) else None,
            "whole_pa_wpa": sum(wpas) if all(x is not None for x in wpas) else None,
            "home_wp_before_pa_inferred": before_wp,
            "status": ";".join(reason) if reason else "validated_in_reviewed_sample",
            "metric_input_available": not reason,
        })
    return records
