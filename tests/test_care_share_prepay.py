import sqlite3

def test_prepay_grant_columns_present():
    import app as appmod
    cx = sqlite3.connect(":memory:")
    cx.execute("CREATE TABLE IF NOT EXISTS prepay_term_grants (session_id TEXT PRIMARY KEY, email TEXT, tier_key TEXT, granted_at TEXT)")
    appmod._ensure_prepay_grant_columns(cx)
    cols = {r[1] for r in cx.execute("PRAGMA table_info(prepay_term_grants)")}
    assert {"attributed_practitioner_id", "practitioner_share_consent", "term_end"} <= cols
