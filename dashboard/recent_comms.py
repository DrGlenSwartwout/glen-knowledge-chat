"""Windowed recent-communications aggregation for the Biofield Intake balancing loop
(B3b). Pure: takes an open sqlite connection so it is testable offline. Mirrors the
queries in app.py:_member_context_for_email but adds a time window and takes a cx."""
import json
import sqlite3


def _json_list(s):
    try:
        v = json.loads(s or "[]")
    except Exception:
        return []
    return [str(x).strip() for x in v if str(x).strip()] if isinstance(v, list) else []


def _intake_summary(first_name, raw_json_str):
    parts = []
    if first_name:
        parts.append(f"first name: {first_name}")
    try:
        payload = json.loads(raw_json_str or "{}")
        data = payload.get("data", payload) or {}
        score = (data.get("total_score") or {}).get("percent") or data.get("score")
        if score:
            parts.append(f"assessment score: {score}%")
        for q in (data.get("quiz_questions") or [])[:6]:
            qt = (q.get("question") or "").strip()
            ans = ", ".join((a.get("answer") or "").strip()
                            for a in (q.get("answers") or []) if a.get("answer"))
            if qt and ans:
                parts.append(f"  {qt}: {ans}")
    except Exception:
        pass
    return "\n".join(parts)


def recent_comms(cx, email, *, days_window=7, query_log_n=20):
    out = {"intake_summary": "", "recent_inquiries": [], "recent_queries": [],
           "recent_feedback": []}
    email = (email or "").strip()
    if not email:
        return out
    cx.row_factory = sqlite3.Row
    win = f"-{int(days_window)} days"                      # int() guards against injection
    try:                                                  # intake: latest, age-agnostic
        row = cx.execute(
            "SELECT first_name, raw_json FROM inbound_leads WHERE email=? "
            "AND source IN ('scoreapp','practice-better','concierge') "
            "ORDER BY id DESC LIMIT 1", (email,)).fetchone()
        if row:
            out["intake_summary"] = _intake_summary(row["first_name"], row["raw_json"])
    except Exception:
        pass
    try:                                                  # inquiries: windowed
        out["recent_inquiries"] = [
            {"main_challenge": r["main_challenge"], "main_goal": r["main_goal"],
             "created_at": r["created_at"]}
            for r in cx.execute(
                "SELECT main_challenge, main_goal, created_at FROM inquiries "
                "WHERE client_email=? AND created_at > datetime('now', ?) "
                "ORDER BY created_at DESC", (email, win)).fetchall()]
    except Exception:
        pass
    try:                                                  # queries: windowed, col fallback
        try:
            rows = cx.execute(
                "SELECT question, ts FROM query_log WHERE email=? AND ts > datetime('now', ?) "
                "ORDER BY id DESC LIMIT ?", (email, win, int(query_log_n))).fetchall()
            out["recent_queries"] = [{"question": r["question"], "ts": r["ts"]} for r in rows]
        except Exception:
            rows = cx.execute(
                "SELECT query, ts FROM query_log WHERE email=? AND ts > datetime('now', ?) "
                "ORDER BY id DESC LIMIT ?", (email, win, int(query_log_n))).fetchall()
            out["recent_queries"] = [{"question": r["query"], "ts": r["ts"]} for r in rows]
    except Exception:
        pass
    try:                                                  # email feedback: windowed, joined
        out["recent_feedback"] = [
            {"summary": r["ai_summary"] or "",
             "topics": _json_list(r["extracted_topics"]),
             "conditions": _json_list(r["extracted_conditions"]),
             "received_at": r["received_at"]}
            for r in cx.execute(
                "SELECT pf.ai_summary, pf.extracted_topics, pf.extracted_conditions, "
                "pf.received_at FROM personal_email_feedback pf "
                "JOIN users u ON u.id = pf.user_id "
                "WHERE lower(u.email)=lower(?) AND pf.received_at > datetime('now', ?) "
                "ORDER BY pf.received_at DESC", (email, win)).fetchall()]
    except Exception:
        pass
    return out
