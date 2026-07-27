"""Per-learner MentorshipU course progress: per-lesson 'watched' + per-module
homework. Pure: stdlib + the caller's sqlite3 connection only; never imports app.
A module is completed when every lesson in it is watched AND the homework is
submitted. Reads never raise (they run on request paths)."""
from __future__ import annotations

import time


def _norm(s: str | None) -> str:
    return (s or "").strip().lower()


def _now() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def init_progress_tables(cx) -> None:
    cx.execute(
        "CREATE TABLE IF NOT EXISTS course_lesson_watched("
        "email TEXT NOT NULL, course TEXT NOT NULL, module TEXT NOT NULL, "
        "lesson TEXT NOT NULL, watched_at TEXT, "
        "UNIQUE(email, course, module, lesson))")
    cx.execute(
        "CREATE TABLE IF NOT EXISTS course_module_homework("
        "email TEXT NOT NULL, course TEXT NOT NULL, module TEXT NOT NULL, "
        "payload TEXT, submitted_at TEXT, ai_rating TEXT, ai_feedback TEXT, updated_at TEXT, "
        "UNIQUE(email, course, module))")
    cx.commit()


def mark_watched(cx, email, course, module, lesson) -> None:
    init_progress_tables(cx)
    cx.execute(
        "INSERT OR IGNORE INTO course_lesson_watched(email, course, module, lesson, watched_at) "
        "VALUES(?,?,?,?,?)", (_norm(email), course, module, lesson, _now()))
    cx.commit()


def record_homework(cx, email, course, module, payload, ai_rating=None, ai_feedback=None) -> None:
    init_progress_tables(cx)
    now = _now()
    cx.execute(
        "INSERT INTO course_module_homework(email, course, module, payload, submitted_at, "
        "ai_rating, ai_feedback, updated_at) VALUES(?,?,?,?,?,?,?,?) "
        "ON CONFLICT(email, course, module) DO UPDATE SET payload=excluded.payload, "
        "submitted_at=excluded.submitted_at, ai_rating=excluded.ai_rating, "
        "ai_feedback=excluded.ai_feedback, updated_at=excluded.updated_at",
        (_norm(email), course, module, payload, now, ai_rating, ai_feedback, now))
    cx.commit()


def watched_lessons(cx, email, course, module) -> set:
    try:
        rows = cx.execute(
            "SELECT lesson FROM course_lesson_watched WHERE email=? AND course=? AND module=?",
            (_norm(email), course, module)).fetchall()
        return {r[0] for r in rows}
    except Exception:
        return set()


def homework(cx, email, course, module):
    try:
        row = cx.execute(
            "SELECT payload, submitted_at, ai_rating, ai_feedback FROM course_module_homework "
            "WHERE email=? AND course=? AND module=?",
            (_norm(email), course, module)).fetchone()
        if not row:
            return None
        return {"payload": row[0], "submitted_at": row[1], "ai_rating": row[2], "ai_feedback": row[3]}
    except Exception:
        return None


def module_completed(cx, email, course, module, lesson_slugs) -> bool:
    try:
        if not lesson_slugs:
            return False
        watched = watched_lessons(cx, email, course, module)
        if not set(lesson_slugs).issubset(watched):
            return False
        hw = homework(cx, email, course, module)
        return bool(hw and hw.get("submitted_at"))
    except Exception:
        return False
