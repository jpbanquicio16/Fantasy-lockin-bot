"""Plain-text rendering of a recommendation: the main screen (design section 26) for the
terminal, and the one-line push notification for each entry."""

from __future__ import annotations

from datetime import datetime

from lockin_core.schedule.calendar import fmt_delta, fmt_local

ADVICE = {"lock_now": "LOCK NOW", "wait": "WAIT", "pass": "PASS", "keep_fallback": "KEEP (counts at week end)"}


def entry_headline(c: dict) -> str:
    a = c["advice"]
    if a == "lock_now":
        return f"LOCK NOW in {c['lock_slot']}"
    if a == "wait":
        return "WAIT"
    return ADVICE.get(a, a.upper())


def notification(c: dict, rec: dict, link: str | None = None, prefix: str = "") -> tuple[str, str]:
    """(title, body) for one entry: action, slot, deadline in Adelaide time, default if you do nothing."""
    title = f"{prefix}{entry_headline(c).title().replace('Now', 'now')}: {c['player']} {c['score']:.1f}"
    lines = [f"{entry_headline(c)} - {c.get('label', '')}. Deadline {c['deadline_local']}."]
    if c["advice"] == "wait" and c.get("conditional_rule"):
        r = c["conditional_rule"]
        lines.append(f"{r['ruled_out'].title()} if he is ruled out, {r['available']} if cleared.")
    if c.get("move_note"):
        lines.append(c["move_note"].capitalize() + ".")
    lines.append(f"If you do nothing: {c['default_if_unattended']}.")
    if link:
        lines.append(link)
    return title, " ".join(lines)


def render(rec: dict, tz, now: datetime | None = None) -> str:
    out = []
    s = rec["summary"]
    obj = "Expected points" if rec["objective"] == "points" else "P(win)"
    fresh = rec.get("data_freshness", {})
    fr = " · ".join(f"{k} {fmt_local(datetime.fromisoformat(v), tz, with_day=False)}" for k, v in fresh.items()
                    if k in ("rosters", "injury_report"))
    out.append(f"WEEK {rec['week']} · {obj} (expected {s['expected_points']:.1f} FP"
               + (f", P(win) {s['p_win']:.2f}" if s.get("p_win") is not None else "") + f")    Data: {fr}")
    if s.get("points_and_win_disagree"):
        out.append("  Expected points and P(win) recommend different actions: both are listed under Alternatives.")
    entries = sorted(rec["entries"], key=lambda c: c["deadline_at"])
    if entries:
        nd = entries[0]
        dl = datetime.fromisoformat(nd["deadline_at"])
        when = fmt_delta(dl - (now or datetime.fromisoformat(rec["created_at"])))
        out.append("")
        out.append(f"NEXT DEADLINE  {nd['deadline_local']} (in {when})")
        for c in entries:
            out.append(f"  {c['player']:<22s} {c['score']:5.1f}  {entry_headline(c):<18s} {c.get('label', '')}"
                       f"   deadline {c['deadline_local']}")
            vals = " · ".join(f"{k.replace('_now', '').replace('_', ' ')} {v:.1f}" for k, v in
                              sorted(c["values"].items(), key=lambda kv: -kv[1]))
            out.append(f"      {vals}")
            out.append(f"      {c['why']}")
            if c.get("locked_unknown"):
                out.append("      Already locked it? Record it: lockin record lock "
                           f"\"{c['player']}\" (the API cannot tell)")
    lu = rec.get("lineup_next_tip")
    if lu:
        out.append("")
        out.append(f"NEXT TIP  {lu['tip_local']}")
        if lu["start"]:
            out.append("  Start   " + " · ".join(f"{x['slot']} {x['player']}" + ("" if x["p_start"] > 0.9 else
                                                                                   f" ({x['p_start']:.0%})")
                                              for x in lu["start"]))
        if lu.get("locked_before_tip"):
            out.append("  Lock first  " + " · ".join(f"{x['player']} in {x['slot']}" for x in lu["locked_before_tip"]))
        if lu.get("entries_due_at_this_tip"):
            out.append("  Decide by this tip  " + ", ".join(f"{x['player']} ({x['score']:.1f})"
                                                           for x in lu["entries_due_at_this_tip"]))
        if lu["bench_for_this_game"]:
            out.append("  Bench for this game  " + ", ".join(x["player"] for x in lu["bench_for_this_game"]))
    unlikely = [p for p in rec["players"] if p["label"] in ("unlikely to contribute this week", "out this week")]
    if unlikely:
        out.append("  Unlikely to contribute this week  " + ", ".join(
            f"{p['player']} ({p['p_contributes']:.0%})" for p in unlikely))
    if rec.get("could_change"):
        out.append("")
        out.append("COULD CHANGE THIS")
        out += [f"  · {x}" for x in rec["could_change"]]
    m = rec.get("matchup") or {}
    if m.get("p_win") is not None:
        out.append("")
        sens = " (sensitive to how the opponent locks)" if m.get("sensitive_to_opponent_model") else ""
        out.append(f"MATCHUP  You {m['my_locked']:.1f} locked · Opponent {m['opp_locked']:.1f} locked "
                   f"({m['opp_locked_source']}) · P(win) {m['p_win']:.2f}{sens}")
        out.append("  P(win) if he " + ", ".join(f"{k.replace('_', ' ')}: {v:.2f}"
                                                for k, v in m["p_win_by_opponent_policy"].items()))
    alts = rec.get("alternatives") or []
    if len(alts) > 1:
        out.append("")
        out.append("ALTERNATIVES (week total, vs chosen, paired SE)")
        for a in alts[:5]:
            desc = ", ".join([f"lock {x['player']} {x['slot']}" for x in a["locks"]]
                             + [f"pass {x['player']}" for x in a["passes"]]) or "keep all open"
            if a.get("lineup"):
                desc += "; lineup " + " ".join(f"{x['slot']}:{x['player'].split()[-1]}" for x in a["lineup"])
            pw = f"  P(win) {a['p_win']:.3f}" if a.get("p_win") is not None else ""
            out.append(f"  {a['expected_points']:7.2f}  {a['vs_best']:+6.2f}  ±{a['paired_se']:.2f}{pw}  {desc}")
    if rec.get("flags"):
        out.append("")
        out.append("Flags: " + ", ".join(rec["flags"]))
    out.append(f"({rec['n_scenarios']} scenarios, {rec['n_candidates_screened']} actions screened, "
               f"{rec['runtime_s']:.1f} s; expires {fmt_local(datetime.fromisoformat(rec['expires_at']), tz)})")
    return "\n".join(out)
