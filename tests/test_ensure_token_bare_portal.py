import sqlite3
from dashboard import client_portal as cp
from dashboard import portal_biofield_reports as pbr

def _db():
    cx = sqlite3.connect(":memory:")
    cp.init_client_portal_table(cx)
    pbr.init_table(cx)
    return cx

def test_ensure_token_creates_portal_without_a_system_b_report():
    cx = _db()
    tok = cp.ensure_token(cx, "a@x.com", "Ann")
    assert tok                                             # a raw token
    # a client_portals row now exists for the email
    assert cx.execute("SELECT COUNT(*) FROM client_portals WHERE email='a@x.com'").fetchone()[0] == 1
    # and NO portal_biofield_reports row was written
    assert cx.execute("SELECT COUNT(*) FROM portal_biofield_reports WHERE email='a@x.com'").fetchone()[0] == 0

def test_ensure_token_is_idempotent():
    cx = _db()
    t1 = cp.ensure_token(cx, "b@x.com", "Bee")
    t2 = cp.ensure_token(cx, "b@x.com", "Bee")
    assert t1 == t2                                        # same stable token, no duplicate portal
    assert cx.execute("SELECT COUNT(*) FROM client_portals WHERE email='b@x.com'").fetchone()[0] == 1
