import json
import sqlite3
import app as appmod
from dashboard import coaching, dispatch as _dispatch, rbac as _rbac


def _seed():
    cx = sqlite3.connect(appmod.LOG_DB)
    cx.row_factory = sqlite3.Row
    coaching.init_coaching_table(cx)
    cx.execute("DELETE FROM coaching_windows WHERE email LIKE 'cohort-%'")
    cx.commit()
    return cx


def test_grant_action_opens_admin_window():
    cx = _seed()
    actor = _rbac.Actor(role=_rbac.OWNER, name="glen") if hasattr(_rbac, "Actor") else None
    res = _dispatch.dispatch_action(cx, "coaching.grant",
                                    {"email": "cohort-a@example.com"},
                                    actor or _rbac.resolve_actor(appmod.dashboard.CONSOLE_SECRET,
                                                                 console_secret=appmod.dashboard.CONSOLE_SECRET),
                                    source="panel", confirmed=True)
    assert res["status"] == "done"
    w = coaching.active_window(cx, "cohort-a@example.com")
    assert w is not None and w["source"] == "admin"


def test_grant_respects_no_stacking():
    cx = _seed()
    coaching.open_window(cx, email="cohort-b@example.com", order_id=99,
                         days=coaching.WINDOW_DAYS, source="self_serve")
    actor = _rbac.resolve_actor(appmod.dashboard.CONSOLE_SECRET,
                                console_secret=appmod.dashboard.CONSOLE_SECRET)
    res = _dispatch.dispatch_action(cx, "coaching.grant",
                                    {"email": "cohort-b@example.com"}, actor,
                                    source="panel", confirmed=True)
    assert res["status"] == "done"
    assert cx.execute("SELECT COUNT(*) FROM coaching_windows WHERE email=?",
                      ("cohort-b@example.com",)).fetchone()[0] == 1  # no second window


def test_cohort_api_requires_auth():
    r = appmod.app.test_client().get("/api/coaching-cohort")
    assert r.status_code == 401


def test_cohort_api_lists_windows():
    cx = _seed()
    coaching.open_window(cx, email="cohort-c@example.com", order_id=1,
                         days=coaching.WINDOW_DAYS, source="self_serve")
    cx.close()
    key = appmod.dashboard.CONSOLE_SECRET
    r = appmod.app.test_client().get(f"/api/coaching-cohort?active=1&key={key}")
    assert r.status_code == 200
    data = json.loads(r.get_data(as_text=True))
    assert data["ok"] is True
    assert any(w["email"] == "cohort-c@example.com" for w in data["data"])
