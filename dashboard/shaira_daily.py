"""Phase 4 — Shaira daily monitoring + report.

Surveys the workspace tables in chat_log.db for the last 24h of Shaira's
activity, then asks Claude to compose a Glen-facing daily report card:
factual (what happened) → analytical (what it means) → proposed (what next)
→ trend. Stored in the daily_reports table, surfaced on the dashboard and
archived to the vault.
"""
import json
import sqlite3
from datetime import datetime, timezone, timedelta

# Diplomatic stuck-language Shaira tends to use instead of saying "I'm stuck"
# (from memory project_shaira_coordination.md).
STUCK_PHRASES = [
    "still working on", "deepening my understanding", "reviewing how",
    "strengthening", "continuing to refine", "familiariz", "still trying to",
    "looking into", "exploring how", "getting a better sense", "still figuring",
]


def _parse(ts):
    if not ts:
        return None
    try:
        dt = datetime.fromisoformat(str(ts).replace("Z", "+00:00"))
        return dt.replace(tzinfo=timezone.utc) if dt.tzinfo is None else dt
    except Exception:
        return None


def gather_metrics(db_path, owner="shaira", window_hours=24):
    """Pull raw activity metrics for the window. Pure DB read — no Claude."""
    now = datetime.now(timezone.utc)
    cutoff   = (now - timedelta(hours=window_hours)).isoformat()
    cutoff3d = (now - timedelta(days=3)).isoformat()
    cutoff7d = (now - timedelta(days=7)).isoformat()

    with sqlite3.connect(db_path) as cx:
        cx.row_factory = sqlite3.Row

        sessions = cx.execute(
            "SELECT todo_id, started_at, ended_at, duration_seconds "
            "FROM todo_time_sessions WHERE owner=? AND started_at>=?",
            (owner, cutoff)).fetchall()
        time_logged, open_sessions = 0, 0
        for s in sessions:
            if s["duration_seconds"]:
                time_logged += s["duration_seconds"]
            elif not s["ended_at"]:
                open_sessions += 1
                st = _parse(s["started_at"])
                if st:
                    time_logged += max(0, int((now - st).total_seconds()))

        completed = cx.execute(
            "SELECT id, title, done_at, action_note FROM todos "
            "WHERE owner=? AND phase='complete' AND done_at>=? ORDER BY done_at DESC",
            (owner, cutoff)).fetchall()
        completed_7d = cx.execute(
            "SELECT COUNT(*) FROM todos WHERE owner=? AND phase='complete' AND done_at>=?",
            (owner, cutoff7d)).fetchone()[0]

        in_process = cx.execute(
            "SELECT id, title, first_started_at FROM todos "
            "WHERE owner=? AND phase='in_process' ORDER BY first_started_at",
            (owner,)).fetchall()
        plan_count = cx.execute(
            "SELECT COUNT(*) FROM todos WHERE owner=? AND phase='plan'", (owner,)
        ).fetchone()[0]

        pending_asks = cx.execute("""
            SELECT tm.todo_id, tm.role, tm.content, tm.created_at, t.title
            FROM todo_messages tm JOIN todos t ON t.id=tm.todo_id
            WHERE t.owner=? AND tm.role IN ('justus_to_glen','justus_to_rae')
              AND tm.created_at > COALESCE((
                SELECT MAX(tm2.created_at) FROM todo_messages tm2
                WHERE tm2.todo_id=tm.todo_id
                  AND tm2.role = CASE tm.role WHEN 'justus_to_glen' THEN 'glen' ELSE 'rae' END
              ), '')
            ORDER BY tm.created_at
        """, (owner,)).fetchall()

        recent_msgs = cx.execute("""
            SELECT tm.todo_id, tm.role, tm.content, tm.created_at, t.title
            FROM todo_messages tm JOIN todos t ON t.id=tm.todo_id
            WHERE t.owner=? AND tm.created_at>=? ORDER BY tm.created_at
        """, (owner, cutoff3d)).fetchall()

        # Idle in-process items — nothing logged in 3+ days
        idle = []
        for ip in in_process:
            last_touch = cx.execute("""
                SELECT MAX(ts) FROM (
                  SELECT MAX(COALESCE(ended_at, started_at)) AS ts
                    FROM todo_time_sessions WHERE todo_id=?
                  UNION ALL
                  SELECT MAX(created_at) AS ts FROM todo_messages WHERE todo_id=?
                )
            """, (ip["id"], ip["id"])).fetchone()[0]
            if not last_touch or last_touch < cutoff3d:
                idle.append({"id": ip["id"], "title": ip["title"], "last_touch": last_touch})

    stuck_hits = []
    for m in recent_msgs:
        if m["role"] != "shaira":
            continue
        low = (m["content"] or "").lower()
        for p in STUCK_PHRASES:
            if p in low:
                stuck_hits.append({
                    "todo_id": m["todo_id"], "title": m["title"], "phrase": p,
                    "snippet": (m["content"] or "")[:240], "created_at": m["created_at"],
                })
                break

    return {
        "generated_at": now.isoformat(),
        "window_hours": window_hours,
        "time_logged_seconds": time_logged,
        "focus_sessions": len(sessions),
        "open_sessions": open_sessions,
        "completed": [dict(c) for c in completed],
        "completed_count": len(completed),
        "completed_7d": completed_7d,
        "in_process": [dict(i) for i in in_process],
        "in_process_count": len(in_process),
        "plan_count": plan_count,
        "pending_asks": [dict(p) for p in pending_asks],
        "pending_ask_count": len(pending_asks),
        "idle_items": idle,
        "stuck_hits": stuck_hits,
        "shaira_message_count": sum(1 for m in recent_msgs if m["role"] == "shaira"),
    }


def _fmt_dur(sec):
    sec = max(0, int(sec or 0))
    if sec < 60:
        return f"{sec}s"
    m = sec // 60
    if m < 60:
        return f"{m}m"
    return f"{m // 60}h {m % 60}m"


def compose_report(metrics, cl):
    """Turn metrics into a Glen-facing markdown card. Uses Claude when there is
    activity to interpret; a thin deterministic card otherwise."""
    m = metrics
    activity = (m["time_logged_seconds"] > 0 or m["completed_count"] > 0
                or m["shaira_message_count"] > 0 or m["pending_ask_count"] > 0)

    if not activity:
        return (
            "## Shaira — Daily Report\n\n"
            "_No workspace activity in the last 24 hours._\n\n"
            f"- **Plan queue:** {m['plan_count']} item(s) waiting\n"
            f"- **In process:** {m['in_process_count']} item(s)\n"
            f"- **Idle (3+ days):** {len(m['idle_items'])}\n\n"
            "She may not have opened the workspace yet, or worked outside it. "
            "If this persists past her next shift, check in."
        )

    facts = {
        "time_logged": _fmt_dur(m["time_logged_seconds"]),
        "focus_sessions": m["focus_sessions"],
        "still_clocked_in": bool(m["open_sessions"]),
        "completed_today": [c["title"] for c in m["completed"]],
        "in_process": [{"title": i["title"], "started": i.get("first_started_at")}
                       for i in m["in_process"]],
        "plan_waiting": m["plan_count"],
        "pending_asks_for_glen_or_rae": [
            {"title": p["title"], "to": p["role"].replace("justus_to_", ""),
             "question": p["content"][:300]} for p in m["pending_asks"]],
        "idle_3d_plus": [i["title"] for i in m["idle_items"]],
        "stuck_language_detected": [
            {"title": s["title"], "phrase": s["phrase"], "snippet": s["snippet"]}
            for s in m["stuck_hits"]],
        "completed_last_7d": m["completed_7d"],
    }

    system = (
        "You are Justus, writing a short daily report for Dr. Glen Swartwout about "
        "Shaira, his Philippines-based VA. Glen reads this with his morning coffee.\n"
        "Structure the report in EXACTLY these four sections. Use the bare section "
        "TITLES as the markdown headings (do NOT append these descriptions to them):\n"
        "## Shaira — Daily Report\n"
        "### Yesterday at a glance — factual: time logged, what she completed, what's in process.\n"
        "### What it means — analytical: read the signals. Flag stuck-language and idle items "
        "honestly but kindly. If she's blocked, say so plainly.\n"
        "### Needs Glen — proposed: anything requiring Glen's decision/approval, especially "
        "pending questions Justus routed to him.\n"
        "### Trend — one line on pace (completions this week).\n"
        "In the 'Needs Glen' section, write each item on its own line STARTING with a severity "
        "tag in square brackets by urgency + impact — [HIGH] (blocking / needs Glen now), "
        "[MED] (decide this week), [LOW] (FYI / minor). If nothing needs him, write exactly "
        "'[LOW] Nothing blocking — she has a clear runway.'\n"
        "Be concise — Glen wants signal, not padding. Total under 250 words. "
        "Stuck-language ('still working on…', 'deepening my understanding…') is a known pattern — "
        "name it directly but constructively. Do not invent facts beyond the data given."
    )
    prompt = (
        "Here is the last 24 hours of Shaira's workspace activity as JSON:\n\n"
        + json.dumps(facts, indent=2, default=str)
        + "\n\nWrite the daily report."
    )
    try:
        resp = cl.messages.create(
            model="claude-haiku-4-5-20251001",
            max_tokens=700,
            system=system,
            messages=[{"role": "user", "content": prompt}],
        )
        text = "".join(b.text for b in resp.content if hasattr(b, "text")).strip()
        return text or _fallback_card(m)
    except Exception:
        return _fallback_card(m)


def _fallback_card(m):
    """Deterministic card if the Claude call fails — never leave the dashboard blank."""
    lines = ["## Shaira — Daily Report", "",
             "### Yesterday at a glance",
             f"- Time logged: {_fmt_dur(m['time_logged_seconds'])} "
             f"across {m['focus_sessions']} focus session(s)",
             f"- Completed: {m['completed_count']}",
             f"- In process: {m['in_process_count']} · Plan queue: {m['plan_count']}", ""]
    if m["completed"]:
        lines.append("**Completed:** " + ", ".join(c["title"] for c in m["completed"]))
    if m["stuck_hits"]:
        lines += ["", "### What it means",
                  f"⚠️ Stuck-language detected on {len(m['stuck_hits'])} item(s) — "
                  "she may be blocked without saying so."]
    if m["idle_items"]:
        lines.append(f"⚠️ {len(m['idle_items'])} item(s) idle 3+ days: "
                     + ", ".join(i["title"] for i in m["idle_items"]))
    if m["pending_asks"]:
        lines += ["", "### Needs Glen"]
        for p in m["pending_asks"]:
            lines.append(f"[HIGH] {p['title']}: {p['content'][:200]}")
    else:
        lines += ["", "### Needs Glen", "[LOW] Nothing blocking — she has a clear runway."]
    lines += ["", f"### Trend", f"{m['completed_7d']} item(s) completed in the last 7 days."]
    return "\n".join(lines)


def generate_and_store(db_path, cl, owner="shaira"):
    """Gather → compose → upsert into daily_reports. Returns the report dict."""
    metrics = gather_metrics(db_path, owner)
    md = compose_report(metrics, cl)
    date_tag = datetime.now(timezone.utc).strftime("%Y-%m-%d")
    with sqlite3.connect(db_path) as cx:
        cx.execute("""
            INSERT INTO daily_reports (owner, report_date, report_md, metrics_json)
            VALUES (?,?,?,?)
            ON CONFLICT(owner, report_date) DO UPDATE SET
              report_md=excluded.report_md,
              metrics_json=excluded.metrics_json,
              created_at=datetime('now')
        """, (owner, date_tag, md, json.dumps(metrics, default=str)))
        cx.commit()
    return {"owner": owner, "report_date": date_tag, "markdown": md, "metrics": metrics}


def latest_report(db_path, owner="shaira"):
    """Most recent stored report, shaped for the dashboard 'briefing' renderer."""
    with sqlite3.connect(db_path) as cx:
        cx.row_factory = sqlite3.Row
        row = cx.execute(
            "SELECT report_date, report_md, created_at FROM daily_reports "
            "WHERE owner=? ORDER BY report_date DESC LIMIT 1", (owner,)
        ).fetchone()
    if not row:
        return {"empty": True, "message": "No daily report yet — first run pending."}
    return {
        "empty": False,
        "markdown": row["report_md"],
        "report_date": row["report_date"],
        "generated_at": row["created_at"],
    }
