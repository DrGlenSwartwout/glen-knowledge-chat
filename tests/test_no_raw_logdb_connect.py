"""Guard: no raw sqlite3 connect to the chat-log DB may bypass the db adapter.

A raw ``sqlite3.connect(...)`` (any alias, any argument form -- LOG_DB, a helper
like ``_db_path()`` / ``_default_db_path()`` / ``_log_db()``, or a ``chat_log.db``
literal) always opens the local SQLite file. So on ``DB_BACKEND=postgres`` such a
call silently reads stale data and loses writes instead of routing to Postgres.
Every chat-log connection MUST go through ``db.connect(...)``.

This is default-deny: ANY raw sqlite3 connect in the scanned production modules is
a violation, UNLESS the file is a genuinely-other database (e4l / FileMaker
snapshot / a ``__main__`` CLI tool that never runs under the Postgres adapter),
enumerated in ``_ALLOWED`` with the reason. Add to ``_ALLOWED`` only for a real
non-chat_log database. See the P06 cutover repoint.
"""

import pathlib
import re

ROOT = pathlib.Path(__file__).resolve().parent.parent

_RAW = re.compile(r"(?:sqlite3|_sqlite3|_sq\d?|_wsq\d?)\.connect\(")

# Files that legitimately open a NON-chat_log database with raw sqlite3 and are
# never routed through the Postgres adapter. basename -> why.
_ALLOWED = {
    "db.py": "the adapter itself",
    "biofield_e4l.py": "e4l.db (read-only voice-scan store)",
    "biofield_reveal_import.py": "e4l.db",
    "fmp_biofield.py": "FileMaker snapshot db",
    "biofield_handoff.py": "FMP snapshot db (local-only biofield_local_app)",
    "biofield_fmp_snapshot.py": "FMP snapshot loader (fmp_snap_* tables)",
    "fmp_orders.py": "__main__ CLI export tool (LOCAL_DB), not an app runtime path",
}


def _scanned_files():
    yield ROOT / "app.py"
    yield ROOT / "incentive_engine.py"
    for p in sorted((ROOT / "dashboard").glob("*.py")):
        if p.name in _ALLOWED:
            continue
        yield p


def test_no_raw_chatlog_connect_bypasses_adapter():
    violations = []
    for path in _scanned_files():
        if not path.exists():
            continue
        for lineno, line in enumerate(path.read_text().splitlines(), 1):
            if _RAW.search(line) and "raw-sqlite-ok" not in line:
                violations.append(f"{path.relative_to(ROOT)}:{lineno}: {line.strip()}")
    assert not violations, (
        f"{len(violations)} raw chat-log connect(s) bypass the db adapter "
        "(hit local SQLite even on Postgres). Use db.connect(...); or, if this is a "
        "genuinely-other database, add the file to _ALLOWED, or append a "
        "'# raw-sqlite-ok: <reason>' marker on the line, with a reason:\n"
        + "\n".join(violations)
    )
