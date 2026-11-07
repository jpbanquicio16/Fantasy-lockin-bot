"""Phone-friendly local/private-network UI. Decision logic lives in the shared core."""
from __future__ import annotations

from datetime import datetime, timedelta
import os

import streamlit as st

from lockin_core.config import Paths, load_league, load_model, load_sim, now_utc
from lockin_core.schedule.calendar import fmt_local
from lockin_core.state.manual import confirm_tip, record_entry, record_opponent, override_player
from lockin_core.state.store import Store
from lineup_lock.app.access import validate_bind_address
from lineup_lock.recommend.render import entry_headline


def main():
    st.set_page_config(page_title="Lock-In Assistant", page_icon="🏀", layout="centered")
    # Streamlit's XSRF protection and its session-backed forms stay enabled.
    address = st.get_option("server.address") or "127.0.0.1"
    try:
        validate_bind_address(address)
    except ValueError as error:
        st.error(str(error))
        st.stop()
    paths = Paths(root=os.environ["LOCKIN_ROOT"]) if os.environ.get("LOCKIN_ROOT") else Paths()
    paths.ensure()
    cfg, store = load_league(paths), Store(paths)
    st.title("Lock-In Assistant")
    st.caption("Make your lineup and lock changes in Sleeper, then record them here.")
    if not cfg.league_id:
        st.info("Set your league ID and username or roster ID in configs/league.yaml. Then run lockin ingest and lockin fit.")
        st.stop()
    objective = st.radio("Optimise for", ["Expected points", "Chance of winning"], horizontal=True)
    if st.button("Refresh recommendation", type="primary", use_container_width=True):
        try:
            from lineup_lock.recommend.run import recommend
            with st.spinner("Checking inputs and simulating the remaining week…"):
                recommend(cfg, load_model(paths), load_sim(paths), paths, store,
                          objective="win" if objective == "Chance of winning" else "points")
            st.rerun()
        except Exception as error:
            st.error(f"Recommendation unavailable: {error}")
    latest = store.latest_recommendation()
    now = now_utc()
    fresh = latest and datetime.fromisoformat(latest["expires_at"]) > now
    rec = latest["result"] if latest else None
    if latest and not fresh:
        st.warning("This recommendation has expired. Refresh before acting; stored deadline defaults are below.")
    if rec and fresh:
        store.mark_shown(latest["rec_id"])
        st.subheader(f"Week {rec['week']}")
        summary = rec["summary"]
        cols = st.columns(2)
        cols[0].metric("Expected week points", f"{summary['expected_points']:.1f}")
        cols[1].metric("Locked points", f"{summary['locked_points']:.1f}")
        if summary.get("p_win") is not None:
            st.metric("Simulated chance of winning", f"{summary['p_win']:.1%}")
        if summary.get("points_and_win_disagree"):
            st.info("Expected points and chance of winning favour different actions. See alternatives below.")
        entries = sorted(rec.get("entries", []), key=lambda c: c["deadline_at"])
        if entries:
            st.subheader(f"Next deadline · {entries[0]['deadline_local']}")
        for c in entries:
            with st.container(border=True):
                st.markdown(f"**{c['player']} · {c['score']:.1f}**")
                st.markdown(f"**{entry_headline(c)}** · {c.get('label', '')}")
                st.write(c["why"])
                st.caption(f"Deadline {c['deadline_local']} · Default: {c['default_if_unattended']}")
                if c.get("locked_unknown"):
                    st.warning("Lock status is unconfirmed. Record whether you already locked this score in Sleeper.")
                with st.expander("Values and uncertainty"):
                    st.json({k: c.get(k) for k in ("values", "voi_fp", "conditional_rule", "paired_se_fp",
                                                  "flip_score", "p_lock_better", "branch_scenarios")})
                    st.caption("Values are simulated week totals. Paired SE is simulation error; P(lock better), when available, compares model draws.")
                with st.form("entry_" + c["entry_id"]):
                    action = st.selectbox("Record what happened in Sleeper", ["lock", "pass", "not_eligible", "not_locked", "score"])
                    score = st.number_input("Score shown in Sleeper", value=float(c["score"]), step=0.5)
                    slot = st.number_input("Slot index (0 is the first starter; see lineup history)", min_value=0, value=int(c.get("lock_slot_index") or 0), step=1)
                    if st.form_submit_button("Save record"):
                        try:
                            state = current_state(cfg, paths, store)
                            record_entry(store, state, c["entry_id"], action, int(slot), score, "streamlit")
                            st.rerun()
                        except Exception as error:
                            st.error(str(error))
        lineup = rec.get("lineup_next_tip")
        if lineup:
            st.subheader(f"Next tip · {lineup['tip_local']}")
            st.dataframe(lineup.get("start", []), use_container_width=True, hide_index=True)
            if lineup.get("bench_for_this_game"):
                st.write("Bench for this game: " + ", ".join(p["player"] for p in lineup["bench_for_this_game"]))
            if lineup.get("locked_before_tip"):
                st.write("Lock first", lineup["locked_before_tip"])
        if rec.get("could_change"):
            st.subheader("What could change this")
            for reason in rec["could_change"]:
                st.write("• " + reason)
        with st.expander("Player contribution probabilities and alternatives"):
            st.dataframe(rec.get("players", []), use_container_width=True, hide_index=True)
            st.json(rec.get("alternatives", []))
        with st.expander("Input freshness and versions"):
            st.json({k: rec.get(k) for k in ("data_freshness", "flags", "versions", "n_scenarios", "runtime_s")})
    elif not latest:
        st.info("No current recommendation. Refresh after setup, or correct the input history below.")
    with st.expander("Deadline defaults", expanded=not fresh):
        for d in store.deadlines():
            da = d["default_action"]
            st.write(f"{da.get('player', 'Open score')} · {fmt_local(d['due_at'], cfg.tz)} · {da.get('default', 'check Sleeper')}")
    with st.expander("Confirm open scores before the first recommendation", expanded=not fresh):
        if st.button("Load current ledger"):
            try:
                st.session_state["ledger"] = current_state(cfg, paths, store)
            except Exception as error:
                st.error(str(error))
        ledger = st.session_state.get("ledger")
        if ledger is not None:
            for message in ledger.blocking + ledger.me.notes:
                st.warning(message)
            st.write({i: f"{slot}: {ledger.name(ledger.me.assignment.get(i, '-'))}"
                      for i, slot in enumerate(ledger.me.slots)})
            for entry in ledger.me.entries:
                with st.form("confirm_" + entry.entry_id):
                    st.write(f"{ledger.name(entry.player_id)} · {entry.fp_half / ledger.units_per_point:.1f}")
                    action = st.selectbox("Current state in Sleeper", ["not_locked", "lock", "pass", "not_eligible"])
                    slot = st.number_input("Slot index", min_value=0, value=ledger.me.slot_of(entry.player_id) or 0)
                    if st.form_submit_button("Confirm score state"):
                        try:
                            record_entry(store, current_state(cfg, paths, store), entry.entry_id, action,
                                         int(slot), device="streamlit")
                            st.session_state.pop("ledger", None)
                            st.rerun()
                        except Exception as error:
                            st.error(str(error))
        with st.form("manual_final"):
            st.caption("Use only after the game is final in Sleeper. Tip-off history still controls eligibility.")
            week = st.number_input("Matchup week", min_value=1, value=int(rec["week"] if rec else 1))
            player = st.text_input("Player ID for completed game")
            game = st.text_input("Game ID for completed game (Sleeper schedule ID)")
            score = st.number_input("Final score shown in Sleeper", value=0.0, step=0.5)
            if st.form_submit_button("Record completed score"):
                try:
                    from lockin_core.state.manual import record_completed_score
                    rid = cfg.my_roster_id or store.get("my_roster_id")
                    if rid is None or not player or not game:
                        raise ValueError("Set the roster ID and provide player and game IDs")
                    record_completed_score(store, cfg.league_id, int(week), rid, player, game, score,
                                           rec.get("units_per_point", 2) if rec else 2, "streamlit")
                    st.rerun()
                except Exception as error:
                    st.error(str(error))
    with st.expander("Record opponent total or your own information"):
        with st.form("opponent"):
            total = st.number_input("Opponent locked total", value=0.0, step=0.5)
            if st.form_submit_button("Record opponent total"):
                try:
                    record_opponent(store, current_state(cfg, paths, store), total, "streamlit")
                    st.rerun()
                except Exception as error:
                    st.error(str(error))
        with st.form("opponent_slot"):
            st.caption("For a nonzero opponent total, record its locked players/slots so they are not simulated twice.")
            opponent_player = st.text_input("Opponent locked player ID")
            opponent_slot = st.number_input("Opponent slot index", min_value=0, value=0)
            opponent_score = st.number_input("Opponent player's locked score", value=0.0, step=0.5)
            if st.form_submit_button("Record opponent slot lock"):
                try:
                    from lockin_core.state.manual import record_opponent_lock
                    record_opponent_lock(store, current_state(cfg, paths, store), opponent_player,
                                         int(opponent_slot), opponent_score, device="streamlit")
                    st.rerun()
                except Exception as error:
                    st.error(str(error))
        with st.form("override"):
            pid = st.text_input("Sleeper player ID")
            probability = st.number_input("P(play)", min_value=0.0, max_value=1.0, value=0.99)
            minutes = st.number_input("Expected minutes if playing", min_value=0.0, max_value=60.0, value=30.0)
            hours = st.number_input("Expires after hours", min_value=1.0, value=48.0)
            if st.form_submit_button("Save override"):
                try:
                    state = current_state(cfg, paths, store)
                    if pid not in state.players:
                        raise ValueError("Unknown rostered player ID")
                    override_player(store, pid, probability, minutes, hours=hours, device="streamlit")
                    st.rerun()
                except Exception as error:
                    st.error(str(error))
    with st.expander("Correct lineup and lock history"):
        rid = cfg.my_roster_id or store.get("my_roster_id")
        if rid is not None:
            snapshots = store.tipoffs(rid, now - timedelta(days=14), now)
            st.dataframe([{**r, "scheduled_tip_at": fmt_local(r["scheduled_tip_at"], cfg.tz)} for r in snapshots],
                         use_container_width=True, hide_index=True)
        with st.form("confirm_tip"):
            roster = st.number_input("Roster ID", min_value=1, value=int(rid or 1), step=1)
            tip = st.text_input("Tip time with timezone", placeholder="2026-11-06T00:30:00+00:00")
            starters = st.text_input("Player IDs in starting-slot order, comma separated; use - for an empty slot")
            if st.form_submit_button("Save corrected tip history"):
                try:
                    confirm_tip(store, cfg.league_id, int(roster), datetime.fromisoformat(tip),
                                [p.strip() if p.strip() not in ("", "-", "0") else None for p in starters.split(",")], "streamlit")
                    st.rerun()
                except Exception as error:
                    st.error(str(error))
        if rid is not None and rec:
            locks = store.locks(cfg.league_id, rec["week"], rid)
            st.dataframe(locks, use_container_width=True, hide_index=True)
            if locks:
                with st.form("retract_lock"):
                    lid = st.selectbox("Incorrect local lock record", [r["lock_id"] for r in locks])
                    st.caption("Corrects this assistant's ledger only. Sleeper locks cannot be undone.")
                    if st.form_submit_button("Retract incorrect record"):
                        store.retract_lock(lid)
                        st.rerun()
    with st.expander("Service status, manual audit and shadow report"):
        st.write("Last service tick:", store.get("last_tick", "not running"))
        st.dataframe(store.events(30), use_container_width=True, hide_index=True)
        st.dataframe(store.alerts(20), use_container_width=True, hide_index=True)
        if st.button("Build shadow report"):
            from lineup_lock.replay.shadow import main as shadow
            st.markdown(shadow(paths, store))


def current_state(cfg, paths, store):
    from lockin_core.state.build import build_state
    state, _ = build_state(cfg, paths, store=store, save=False, strict=False)
    return state


if __name__ == "__main__":
    main()
