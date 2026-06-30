"""Sqlite store for the Glendalf fireside conversation (anonymous v1).

One row per fireside session, keyed in practice by the amg_session cookie. Pure
module: every function takes a caller-supplied sqlite3 connection and calls
init_table() on first use (mirrors dashboard/journal_store.py). No Flask import;
the caller holds the DB lock around writes.
"""
import json
import sqlite3

_JSON_COLS = ("transcript", "ash_coverage", "signals")
_NOW = "strftime('%Y-%m-%dT%H:%M:%fZ','now')"


def init_table(cx) -> None:
    cx.execute(
        f"""
        CREATE TABLE IF NOT EXISTS fireside_sessions (
          id            INTEGER PRIMARY KEY AUTOINCREMENT,
          amg_session   TEXT,
          user_email    TEXT,
          user_name     TEXT,
          started_at    TEXT DEFAULT ({_NOW}),
          last_turn_at  TEXT,
          turn_count    INTEGER NOT NULL DEFAULT 0,
          ended_at      TEXT,
          transcript    TEXT NOT NULL DEFAULT '[]',
          ash_coverage  TEXT NOT NULL DEFAULT '{{}}',
          signals       TEXT
        )
        """
    )
    cx.execute(
        "CREATE INDEX IF NOT EXISTS idx_fireside_amg "
        "ON fireside_sessions(amg_session, ended_at)"
    )
    cx.commit()


def _decode(row: sqlite3.Row) -> dict:
    d = dict(row)
    for c in _JSON_COLS:
        v = d.get(c)
        if isinstance(v, str):
            try:
                d[c] = json.loads(v)
            except (ValueError, TypeError):
                d[c] = None
    return d


def get(cx, fireside_id: int) -> dict | None:
    init_table(cx)
    _saved_rf = cx.row_factory
    cx.row_factory = sqlite3.Row
    try:
        row = cx.execute(
            "SELECT * FROM fireside_sessions WHERE id = ?", (int(fireside_id),)
        ).fetchone()
        return _decode(row) if row is not None else None
    finally:
        cx.row_factory = _saved_rf


def get_or_create(cx, amg_session: str) -> dict:
    init_table(cx)
    _saved_rf = cx.row_factory
    cx.row_factory = sqlite3.Row
    try:
        row = cx.execute(
            "SELECT * FROM fireside_sessions "
            "WHERE amg_session = ? AND ended_at IS NULL "
            "ORDER BY id DESC LIMIT 1",
            (amg_session or "",),
        ).fetchone()
        if row is not None:
            return _decode(row)
        cur = cx.execute(
            f"INSERT INTO fireside_sessions (amg_session, last_turn_at) "
            f"VALUES (?, {_NOW})",
            (amg_session or "",),
        )
        cx.commit()
        return get(cx, cur.lastrowid)
    finally:
        cx.row_factory = _saved_rf


def append_turn(cx, fireside_id: int, speaker: str, text: str) -> None:
    init_table(cx)
    _saved_rf = cx.row_factory
    cx.row_factory = sqlite3.Row
    try:
        row = cx.execute(
            "SELECT transcript FROM fireside_sessions WHERE id = ?", (int(fireside_id),)
        ).fetchone()
        if row is None:
            return
        try:
            transcript = json.loads(row["transcript"]) or []
        except (ValueError, TypeError):
            transcript = []
        ts = cx.execute(f"SELECT {_NOW}").fetchone()[0]
        transcript.append({"speaker": speaker, "text": text or "", "ts": ts})
        inc = 1 if speaker == "traveler" else 0
        cx.execute(
            "UPDATE fireside_sessions "
            "SET transcript = ?, last_turn_at = ?, turn_count = turn_count + ? "
            "WHERE id = ?",
            (json.dumps(transcript), ts, inc, int(fireside_id)),
        )
        cx.commit()
    finally:
        cx.row_factory = _saved_rf


def update_coverage(cx, fireside_id: int, coverage: dict) -> None:
    init_table(cx)
    cx.execute(
        "UPDATE fireside_sessions SET ash_coverage = ? WHERE id = ?",
        (json.dumps(coverage or {}), int(fireside_id)),
    )
    cx.commit()


def mark_ended(cx, fireside_id: int) -> None:
    init_table(cx)
    cx.execute(
        f"UPDATE fireside_sessions SET ended_at = {_NOW} WHERE id = ?",
        (int(fireside_id),),
    )
    cx.commit()
