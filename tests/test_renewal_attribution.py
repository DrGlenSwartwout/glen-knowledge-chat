import sqlite3, importlib, sys
from pathlib import Path

def _app():
    repo = Path(__file__).resolve().parent.parent
    if str(repo) not in sys.path: sys.path.insert(0, str(repo))
    return importlib.import_module("app")

def _seed(path):
    cx = sqlite3.connect(path)
    cx.execute("CREATE TABLE prepay_term_grants (session_id TEXT PRIMARY KEY, email TEXT, tier_key TEXT, granted_at TEXT, attributed_practitioner_id TEXT, practitioner_share_consent INTEGER DEFAULT 0, term_end TEXT)")
    cx.execute("CREATE TABLE subscriptions (email TEXT, attributed_practitioner_id TEXT, practitioner_share_consent INTEGER, kind TEXT, created_at TEXT)")
    cx.commit(); cx.close()

def test_none_when_no_history(tmp_path):
    app = _app(); p = str(tmp_path/"log.db"); _seed(p)
    assert app._last_attributed_practitioner("pat@x.com", db_path=p) is None

def test_prepay_grant_returned(tmp_path):
    app = _app(); p = str(tmp_path/"log.db"); _seed(p)
    cx = sqlite3.connect(p); cx.execute("INSERT INTO prepay_term_grants (session_id,email,granted_at,attributed_practitioner_id,practitioner_share_consent) VALUES ('s1','pat@x.com','2026-01-01T00:00:00Z','prac-42',1)"); cx.commit(); cx.close()
    assert app._last_attributed_practitioner("PAT@x.com", db_path=p) == {"pid": "prac-42", "consent": 1}

def test_most_recent_across_sources(tmp_path):
    app = _app(); p = str(tmp_path/"log.db"); _seed(p)
    cx = sqlite3.connect(p)
    cx.execute("INSERT INTO prepay_term_grants (session_id,email,granted_at,attributed_practitioner_id,practitioner_share_consent) VALUES ('s1','pat@x.com','2026-01-01T00:00:00Z','prac-A',1)")
    cx.execute("INSERT INTO subscriptions (email,attributed_practitioner_id,practitioner_share_consent,kind,created_at) VALUES ('pat@x.com','prac-B',0,'membership','2026-06-01T00:00:00Z')")
    cx.commit(); cx.close()
    assert app._last_attributed_practitioner("pat@x.com", db_path=p) == {"pid": "prac-B", "consent": 0}  # later wins

def test_ignores_unattributed_rows(tmp_path):
    app = _app(); p = str(tmp_path/"log.db"); _seed(p)
    cx = sqlite3.connect(p); cx.execute("INSERT INTO prepay_term_grants (session_id,email,granted_at,attributed_practitioner_id) VALUES ('s1','pat@x.com','2026-01-01T00:00:00Z',NULL)"); cx.commit(); cx.close()
    assert app._last_attributed_practitioner("pat@x.com", db_path=p) is None
