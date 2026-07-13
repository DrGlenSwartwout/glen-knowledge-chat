"""Console-side snapshot of the vault Task Board.

Prod can't scan Glen's local vault/repos, so the local task-board refresh pushes
its generated tasks.json here (via /api/console/taskboard/sync). One row holds the
latest snapshot; /console/taskboard renders it. LOG_DB (SQLite).

Sibling of dashboard/client_scans.py — same local-push-to-prod pattern.
"""
import datetime
import json


def _now():
    return datetime.datetime.now(datetime.timezone.utc).strftime("%Y-%m-%d %H:%M:%S")


def init_task_board_table(cx):
    cx.execute("""
        CREATE TABLE IF NOT EXISTS task_board (
            id           TEXT PRIMARY KEY,
            data         TEXT NOT NULL,
            generated_at TEXT,
            synced_at    TEXT
        )
    """)
    cx.commit()


def upsert_board(cx, cards, generated_at, board_id="default"):
    """Replace the stored snapshot with the pushed cards. Returns the card count.
    The whole board is one row, so a re-push fully replaces the prior snapshot
    (no stale cards linger)."""
    payload = json.dumps({"generated_at": generated_at, "cards": cards or []},
                         ensure_ascii=False)
    cx.execute(
        "INSERT INTO task_board (id, data, generated_at, synced_at) VALUES (?,?,?,?) "
        "ON CONFLICT(id) DO UPDATE SET data=excluded.data, "
        "generated_at=excluded.generated_at, synced_at=excluded.synced_at",
        (board_id, payload, generated_at or "", _now()))
    cx.commit()
    return len(cards or [])


def get_board(cx, board_id="default"):
    """Return {generated_at, cards, synced_at}; safe empty board if nothing synced."""
    row = cx.execute(
        "SELECT data, generated_at, synced_at FROM task_board WHERE id=?",
        (board_id,)).fetchone()
    if not row:
        return {"generated_at": None, "cards": [], "synced_at": None}
    try:
        d = json.loads(row[0]) if row[0] else {}
    except (ValueError, TypeError):
        d = {}
    return {
        "generated_at": d.get("generated_at") or (row[1] or None),
        "cards": d.get("cards", []),
        "synced_at": row[2],
    }
