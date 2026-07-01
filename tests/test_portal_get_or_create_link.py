import sqlite3

from dashboard import client_portal as cp
from dashboard import notify_state as ns


def _cx():
    cx = sqlite3.connect(":memory:")
    cp.init_client_portal_table(cx)
    ns.init_table(cx)
    return cx


def test_ensure_token_is_idempotent_never_rotates():
    cx = _cx()
    t1 = cp.ensure_token(cx, "a@b.com", "A B")
    t2 = cp.ensure_token(cx, "a@b.com", "A B")
    assert t1 and t1 == t2  # cached link returned verbatim, no rotation across waves


def test_ensure_token_mints_pending_portal_for_new_email():
    cx = _cx()
    t = cp.ensure_token(cx, "new@x.com", "New Person")
    assert t
    assert cx.execute("SELECT 1 FROM client_portals WHERE email=?", ("new@x.com",)).fetchone()
