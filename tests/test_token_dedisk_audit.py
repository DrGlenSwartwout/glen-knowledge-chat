import pathlib, re

ROOT = pathlib.Path(__file__).resolve().parent.parent

def _src(rel): return (ROOT / rel).read_text()

def test_gmail_token_uses_db_connect_not_raw_sqlite():
    s = _src("dashboard/gmail_token.py")
    # oauth_tokens access must go through the adapter, not a raw sqlite3 handle.
    assert "db.connect(" in s
    assert "sqlite3.connect(" not in s

def test_money_qb_rt_not_file_backed_for_writes():
    s = _src("dashboard/money.py")
    # The rotated RT is written to the DB row, never back to the /data file.
    assert 'open(QB_RT_CACHE, "w")' not in s
    assert "_qb_rt_write_at(" in s

def test_admin_upload_writes_db_not_disk():
    s = _src("app.py")
    # The one-time uploader persists to oauth_tokens, not google-token.json.
    seg = s[s.index("def admin_upload_gmail_token"): s.index("def admin_upload_gmail_token") + 1200]
    assert "_write_db_token(" in seg
    assert 'open(target, "w")' not in seg
