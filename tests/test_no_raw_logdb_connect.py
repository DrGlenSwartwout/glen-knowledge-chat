"""Guard: no raw sqlite3 connect to the chat-log DB may bypass the db adapter.

A raw ``sqlite3.connect(LOG_DB)`` (any historical alias, any ``str()``/``or``
wrapper) always opens the local SQLite file — so on ``DB_BACKEND=postgres`` it
silently reads stale data and loses writes instead of routing to Postgres.
Every chat-log connection must go through ``db.connect(...)``. This guard fails
if any bypass is (re)introduced. See the P06 cutover repoint.
"""

import pathlib
import re

ROOT = pathlib.Path(__file__).resolve().parent.parent

# (sqlite3|_sqlite3|_sq|_sq2|_sq3|_wsq|_wsq2).connect( [str(] [<name> or ] [_]LOG_DB
_RAW_LOGDB = re.compile(
    r"(?:sqlite3|_sqlite3|_sq\d?|_wsq\d?)\.connect\(\s*"
    r"(?:str\(\s*)?"
    r"(?:[A-Za-z_][A-Za-z0-9_]*\s+or\s+)?"
    r"_?LOG_DB\b"
)


def _scanned_files():
    yield ROOT / "app.py"
    yield ROOT / "incentive_engine.py"
    for p in sorted((ROOT / "dashboard").glob("*.py")):
        if p.name == "db.py":
            continue  # the adapter itself legitimately calls sqlite3.connect
        yield p


def test_no_raw_chatlog_connect_bypasses_adapter():
    violations = []
    for path in _scanned_files():
        if not path.exists():
            continue
        for lineno, line in enumerate(path.read_text().splitlines(), 1):
            if _RAW_LOGDB.search(line):
                violations.append(f"{path.relative_to(ROOT)}:{lineno}: {line.strip()}")
    assert not violations, (
        f"{len(violations)} raw chat-log connect(s) bypass the db adapter "
        "(hit local SQLite even on Postgres). Use db.connect(...):\n"
        + "\n".join(violations)
    )
