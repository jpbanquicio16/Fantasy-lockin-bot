"""Validated manual ledger writes shared by the CLI and Streamlit.

These record actions the user has taken in Sleeper; they never submit transactions.
"""
from __future__ import annotations

import math
from datetime import datetime, timedelta
from decimal import Decimal

from lockin_core.config import now_utc
from lockin_core.sim.problem import SLOT_ACCEPTS
from lockin_core.state.ledger import tip_batch_id
from lockin_core.state.models import ValidatedState
from lockin_core.state.store import Store


def score_units(score: float, resolution: int) -> int:
    value = Decimal(str(score)) * resolution
    if not value.is_finite() or value != value.to_integral_value():
        raise ValueError(f"Score must be finite and in increments of {1 / resolution:g}")
    return int(value)


def record_entry(store: Store, state: ValidatedState, entry_id: str, action: str,
                 slot_index: int | None = None, score: float | None = None, device: str = "cli") -> str:
    latest = store.latest_recommendation()
    entry = next((e for e in state.me.entries if e.entry_id == entry_id), None)
    if entry is None:
        raise ValueError("This score is no longer an open entry; refresh the state")
    now = now_utc()
    if action == "lock":
        if now >= entry.deadline_at:
            raise ValueError("The lock deadline has passed; use a history correction for a past lock")
        if entry.eligibility not in ("observed", "manual"):
            raise ValueError("Confirm the tip-off lineup before recording a lock")
        if slot_index is None:
            slot_index = state.me.slot_of(entry.player_id)
        if slot_index is None or not 0 <= slot_index < len(state.me.slots):
            raise ValueError("Choose a valid starting slot")
        if slot_index in state.me.locked or entry.player_id in state.me.used_players:
            raise ValueError("Player or slot is already locked")
        slot = state.me.slots[slot_index]
        if not SLOT_ACCEPTS[slot].intersection(state.players[entry.player_id].positions):
            raise ValueError("Player is not eligible for this slot")
        occupant = state.me.assignment.get(slot_index)
        if occupant in state.me.frozen or entry.player_id in state.me.frozen:
            raise ValueError("A player in a live game is frozen in his slot")
        units = entry.fp_half if score is None else score_units(score, state.units_per_point)
        mid = store.record_lock(state.league_id, state.week, state.me.roster_id, slot_index, slot,
                                entry.player_id, entry.game_id, units)
        store.resolve_entry(entry_id, "locked")
    elif action in ("pass", "not_eligible", "not_locked", "score"):
        if action == "score" and score is None:
            raise ValueError("Enter the score shown in Sleeper")
        value = {"fp_half": score_units(score, state.units_per_point)} if action == "score" else {"player": entry.player_id}
        mid = store.add_manual(action, entry_id, value, device=device)
        if action in ("pass", "not_eligible"):
            store.resolve_entry(entry_id, "passed" if action == "pass" else "not_eligible")
        else:
            store.invalidate(action)
    else:
        raise ValueError("Unknown entry action")
    if latest:
        best = latest["result"].get("best_action", {})
        matched = (action == "lock" and any(x.get("player_id") == entry.player_id and x.get("slot_index") == slot_index
                                            for x in best.get("locks", []))) or (
            action == "pass" and any(x.get("player_id") == entry.player_id for x in best.get("passes", [])))
        store.add_outcome(latest["rec_id"], {"action": action, "player_id": entry.player_id, "slot_index": slot_index},
                          matched, None, None)
    store.log("manual_action", {"id": mid, "entry_id": entry_id, "action": action, "device": device})
    return mid


def record_opponent(store: Store, state: ValidatedState, total: float, device: str = "cli") -> str:
    if state.opp is None:
        raise ValueError("No opponent this week")
    score_units(total, state.units_per_point)
    mid = store.add_manual("opp_locked", f"{state.week}:{state.opp.roster_id}", {"total": total}, device=device)
    store.invalidate("opponent total")
    return mid


def override_player(store: Store, player_id: str, p_play=None, minutes=None, out_until=None,
                    hours: float = 48, device: str = "cli") -> str:
    if not math.isfinite(hours) or hours <= 0:
        raise ValueError("Override duration must be positive")
    value = {"expires_at": (now_utc() + timedelta(hours=hours)).isoformat()}
    if p_play is not None:
        if not math.isfinite(p_play) or not 0 <= p_play <= 1:
            raise ValueError("P(play) must be between 0 and 1")
        value["p_play"] = p_play
    if minutes is not None:
        if not math.isfinite(minutes) or not 0 <= minutes <= 60:
            raise ValueError("Minutes must be between 0 and 60")
        value["minutes"] = minutes
    if out_until:
        from datetime import date
        date.fromisoformat(out_until)
        value["out_until"] = out_until
    if len(value) == 1:
        raise ValueError("Supply at least one override")
    mid = store.add_manual("override", player_id, value, device=device)
    store.invalidate("player override")
    return mid


def confirm_tip(store: Store, league_id: str, roster_id: int, tip: datetime, starters: list[str | None],
                device: str = "cli") -> None:
    if tip.tzinfo is None:
        raise ValueError("Tip time must include a timezone (for example +00:00)")
    if tip > now_utc():
        raise ValueError("Cannot confirm a tip that has not happened")
    players = [p for p in starters if p]
    if len(set(players)) != len(players):
        raise ValueError("A player cannot occupy two slots")
    if roster_id is None:
        raise ValueError("Specify a roster ID")
    store.save_tipoff(tip_batch_id(tip), league_id, roster_id, tip, None, None, starters,
                      "confirmed_manual", resolved_by="you")
    store.invalidate("tip correction")
    store.log("tip_corrected", {"tip_at": tip.isoformat(), "starters": starters, "device": device})


def record_completed_score(store: Store, league_id: str, week: int, roster_id: int, player_id: str,
                           game_id: str, score: float, resolution: int = 2, device: str = 'cli') -> None:
    """Explicit fallback when the user sees a final score before the statistics feed arrives."""
    key = f'{week}:{roster_id}:{player_id}:{game_id}'
    store.add_manual('score', key, {'fp_half': score_units(score, resolution)}, device=device)
    store.add_manual('completed_game', f'{league_id}:{game_id}',
                     {'game_id': game_id, 'completed_at': now_utc().isoformat()}, device=device)
    store.invalidate('manual completed score')


def record_historical_lock(store: Store, state: ValidatedState, calendar, player_id: str, game_id: str,
                           slot_index: int, score: float, locked_at: datetime, device='cli') -> str:
    from lockin_core.state.ledger import starters_at
    if locked_at.tzinfo is None or locked_at > now_utc():
        raise ValueError('Use the actual past lock time with a timezone')
    game = calendar.game(game_id)
    if game is None or game.status != 'final' or locked_at < game.end_at:
        raise ValueError('The game must have finished before the recorded lock time')
    if player_id not in state.players:
        raise ValueError('Unknown player')
    team = state.players[player_id].team
    deadline, _ = calendar.deadline(team, game_id)
    if locked_at >= deadline:
        raise ValueError('Recorded lock time is after the deadline')
    starters, status = starters_at(store, state.me.roster_id, game.tip_at)
    if status not in ('observed', 'confirmed_manual') or not starters or player_id not in starters:
        raise ValueError('Confirm that the player started at the game tip first')
    if not 0 <= slot_index < len(state.me.slots):
        raise ValueError('Invalid slot index')
    slot = state.me.slots[slot_index]
    if not SLOT_ACCEPTS[slot].intersection(state.players[player_id].positions):
        raise ValueError('Player is not eligible for that slot')
    lid = store.record_lock(state.league_id,state.week,state.me.roster_id,slot_index,slot,
                            player_id,game_id,score_units(score,state.units_per_point),locked_at=locked_at)
    store.resolve_entry(f'{state.week}:{state.me.roster_id}:{player_id}:{game_id}','locked')
    store.log('historical_lock',{'lock_id':lid,'device':device})
    return lid


def record_opponent_lock(store: Store, state: ValidatedState, player_id: str, slot_index: int,
                         score: float, game_id: str | None = None, device='cli') -> str:
    side = state.opp
    if side is None or player_id not in (*side.active, *side.reserve):
        raise ValueError('Player is not on the opponent roster')
    if not 0 <= slot_index < len(side.slots):
        raise ValueError('Invalid opponent slot index')
    slot = side.slots[slot_index]
    if not SLOT_ACCEPTS[slot].intersection(state.players[player_id].positions):
        raise ValueError('Opponent player is not eligible for this slot')
    lid = store.record_lock(state.league_id, state.week, side.roster_id, slot_index, slot, player_id,
                            game_id, score_units(score, state.units_per_point))
    store.invalidate('opponent slot lock')
    store.log('opponent_lock', {'lock_id':lid,'device':device})
    return lid
