import sqlite3, sys
from pathlib import Path
import pytest


def _mods():
    repo_root = Path(__file__).resolve().parent.parent
    if str(repo_root) not in sys.path:
        sys.path.insert(0, str(repo_root))
    try:
        from dashboard import studio_credit, studio_credit_actions, actions
        return studio_credit, studio_credit_actions, actions
    except Exception as e:
        pytest.skip(f"module not importable: {e}")


class _Actor:
    role = "owner"
    name = "glen"


def _grant_spy(cx, email, days):
    import uuid
    g = "2026-06-20T00:00:00Z"
    cx.execute("INSERT INTO memberships VALUES (?,?,?,?,?,?,?,?)",
               (str(uuid.uuid4()), email, g, "2026-07-20T00:00:00Z",
                "studio_credit", "studio_credit", "", ""))
    return {"membership_id": "mem-x", "magic_link_url": "https://x/tok"}


def _setup(tmp_path):
    sc, sca, _ = _mods()
    cx = sqlite3.connect(str(tmp_path / "t.db"))
    cx.row_factory = sqlite3.Row
    sc.migrate(cx)
    cx.execute("CREATE TABLE IF NOT EXISTS memberships "
               "(id TEXT PRIMARY KEY, email TEXT, granted_at TEXT, expires_at TEXT, "
               " granted_by TEXT, source TEXT, truly_vip_ref TEXT, notes TEXT)")
    cx.commit()
    sca.configure(grant_fn=_grant_spy)
    sca.register()
    return sc, sca, cx


def test_add_then_approve_via_executor(tmp_path):
    sc, sca, cx = _setup(tmp_path)
    from dashboard.actions import get_action
    add = get_action("studio_credit.add")
    assert add.risk_tier == "low_write" and set(add.permission) == {"owner", "ops"}
    r = add.executor({"email": "a@x.com", "invoice_ref": "INV1"},
                     {"cx": cx, "actor": _Actor()})
    cid = r["id"]
    appr = get_action("studio_credit.approve")
    r2 = appr.executor({"id": cid}, {"cx": cx, "actor": _Actor()})
    assert r2["ok"] is True and r2["membership_id"] == "mem-x"
    assert sc.get(cx, cid)["status"] == "approved"


def test_approve_blocked_within_year_executor(tmp_path):
    sc, sca, cx = _setup(tmp_path)
    from dashboard.actions import get_action
    cx.execute("INSERT INTO memberships VALUES "
               "('m0','a@x.com','2026-06-01T00:00:00Z','2026-07-01T00:00:00Z',"
               "'studio_credit','studio_credit','','')")
    cx.commit()
    cid = get_action("studio_credit.add").executor(
        {"email": "a@x.com"}, {"cx": cx, "actor": _Actor()})["id"]
    res = get_action("studio_credit.approve").executor(
        {"id": cid}, {"cx": cx, "actor": _Actor()})
    assert res["ok"] is False and res["warning"] == "granted_within_year"
