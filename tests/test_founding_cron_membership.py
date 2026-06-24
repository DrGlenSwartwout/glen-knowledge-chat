import sqlite3
import app as appmod


def test_founding_product_charge_extends_comp_membership(monkeypatch):
    cx = sqlite3.connect(":memory:"); cx.row_factory = sqlite3.Row
    cx.execute("CREATE TABLE IF NOT EXISTS memberships (id TEXT, email TEXT, granted_at TEXT,"
               " expires_at TEXT, granted_by TEXT, source TEXT, truly_vip_ref TEXT, notes TEXT)")
    calls = []
    monkeypatch.setattr(appmod, "_extend_membership_grant",
        lambda c, email, until, source="x": calls.append((email, source)))
    # Simulate the success-branch tail for a founding product sub:
    sub = {"id": 1, "email": "f@x.com", "founding": 1}
    updated = {"next_charge_date": "2026-09-01"}
    appmod._maybe_extend_founding_membership(cx, sub, updated)
    assert calls == [("f@x.com", "founding")]

    # non-founding product sub: no extension
    calls.clear()
    appmod._maybe_extend_founding_membership(cx, {"id": 2, "email": "p@x.com", "founding": 0}, updated)
    assert calls == []
