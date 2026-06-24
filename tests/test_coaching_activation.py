import json
import sqlite3
from datetime import datetime, timedelta, timezone
import app as appmod
from dashboard import coaching

EMAIL = "activate-test@example.com"


def _seed():
    cx = sqlite3.connect(appmod.LOG_DB)
    cx.row_factory = sqlite3.Row
    coaching.init_coaching_table(cx)
    cx.execute("CREATE TABLE IF NOT EXISTS memberships (id TEXT PRIMARY KEY, email TEXT, "
               "granted_at TEXT, expires_at TEXT, granted_by TEXT, source TEXT, "
               "truly_vip_ref TEXT, notes TEXT, last_reminder_at TEXT)")
    cx.execute("CREATE TABLE IF NOT EXISTS orders (id INTEGER PRIMARY KEY AUTOINCREMENT, "
               "created_at TEXT, source TEXT, external_ref TEXT, email TEXT, status TEXT DEFAULT 'new')")
    for t in ("coaching_windows", "memberships", "orders"):
        cx.execute(f"DELETE FROM {t} WHERE email=?" if t != "coaching_windows" else
                   "DELETE FROM coaching_windows WHERE email=?", (EMAIL,))
    cx.execute("DELETE FROM auth_tokens WHERE email=?", (EMAIL,))
    def iso(d):
        return (datetime.utcnow() + timedelta(days=d)).isoformat() + "Z"
    cx.execute("INSERT INTO memberships (id,email,granted_at,expires_at,source) VALUES (?,?,?,?,?)",
               ("mt", EMAIL, iso(-30), iso(20), "membership"))
    cx.execute("INSERT INTO orders (created_at,source,external_ref,email) VALUES (?,?,?,?)",
               (iso(-5), "reorder", "ract", EMAIL))
    cx.commit()
    oid = cx.execute("SELECT id FROM orders WHERE email=?", (EMAIL,)).fetchone()[0]
    cx.close()
    return oid


def _cleanup():
    cx = sqlite3.connect(appmod.LOG_DB)
    cx.execute("DELETE FROM coaching_windows WHERE email=?", (EMAIL,))
    cx.execute("DELETE FROM memberships WHERE email=?", (EMAIL,))
    cx.execute("DELETE FROM orders WHERE email=?", (EMAIL,))
    cx.execute("DELETE FROM auth_tokens WHERE email=?", (EMAIL,))
    cx.commit(); cx.close()


def test_activate_get_previews_without_opening():
    oid = _seed()
    try:
        tok = appmod._mint_coaching_activate_link(EMAIL, oid)
        r = appmod.app.test_client().get(f"/coaching/activate/{tok}")
        assert r.status_code == 200
        body = r.get_data(as_text=True)
        assert '"confirmed": false' in body
        cx = sqlite3.connect(appmod.LOG_DB); cx.row_factory = sqlite3.Row
        assert coaching.active_window(cx, EMAIL) is None     # GET did not open
        cx.close()
    finally:
        _cleanup()


def test_activate_post_opens_window():
    oid = _seed()
    try:
        tok = appmod._mint_coaching_activate_link(EMAIL, oid)
        r = appmod.app.test_client().post(f"/coaching/activate/{tok}")
        assert r.status_code == 200
        body = r.get_data(as_text=True)
        assert '"confirmed": true' in body and '"ok": true' in body
        cx = sqlite3.connect(appmod.LOG_DB); cx.row_factory = sqlite3.Row
        assert coaching.active_window(cx, EMAIL) is not None  # window opened
        cx.close()
    finally:
        _cleanup()


def test_activate_invalid_token():
    r = appmod.app.test_client().get("/coaching/activate/garbage")
    assert r.status_code == 200
    assert '"valid": false' in r.get_data(as_text=True)


def test_self_serve_start_opens_for_logged_in_member():
    _seed()
    try:
        c = appmod.app.test_client()
        c.set_cookie("rm_member_email", EMAIL)
        r = c.post("/coaching/start")
        assert r.status_code == 200
        d = json.loads(r.get_data(as_text=True))
        assert d["ok"] is True and d["created"] is True and d["ends_at"]
    finally:
        _cleanup()


def test_self_serve_start_no_member_offers_99():
    c = appmod.app.test_client()
    c.set_cookie("rm_member_email", "nobody-here@example.com")
    r = c.post("/coaching/start")
    d = json.loads(r.get_data(as_text=True))
    assert d["ok"] is False and d.get("offer_99") is True
