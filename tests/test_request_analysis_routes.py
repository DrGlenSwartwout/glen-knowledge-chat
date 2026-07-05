import importlib, sqlite3, sys
from pathlib import Path
import pytest


def _app(tmp_path, monkeypatch, *, flag="1"):
    monkeypatch.setenv("DATA_DIR", str(tmp_path))
    monkeypatch.setenv("SCAN_REQUEST_ENABLED", flag)
    monkeypatch.setenv("SCAN_LIST_ENABLED", "1")
    monkeypatch.delenv("CONSOLE_SECRET", raising=False)
    repo = Path(__file__).resolve().parent.parent
    if str(repo) not in sys.path: sys.path.insert(0, str(repo))
    try:
        import app as appmod; importlib.reload(appmod)
        import dashboard as _d; monkeypatch.setattr(_d, "CONSOLE_SECRET", "", raising=False)
    except Exception as e:
        pytest.skip(f"app not importable: {e}")
    return appmod


def _mint(appmod, email, scan_date="2026-06-28"):
    from dashboard import client_portal as cp, client_scans as cs
    with sqlite3.connect(appmod.LOG_DB) as cx:
        cp.init_client_portal_table(cx); cs.init_client_scans_table(cx)
        cs.upsert_scans(cx, email, [{"scan_date": scan_date, "scan_id": 9}])
        tok = cp.upsert_portal(cx, email, "N", {}); cx.commit()
    return tok[0] if isinstance(tok, (tuple, list)) else tok


def test_free_member_one_then_quota(tmp_path, monkeypatch):
    appmod = _app(tmp_path, monkeypatch)
    monkeypatch.setattr(appmod, "_is_paid_member", lambda e: False)
    token = _mint(appmod, "k@x.com")
    if not token: pytest.skip("no mint")
    c = appmod.app.test_client()
    r = c.post(f"/api/portal/{token}/request-analysis", json={"scan_id": 9, "scan_date": "2026-06-28"})
    assert r.get_json()["status"] == "pending"
    # second scan same month → quota exceeded
    from dashboard import client_scans as cs
    with sqlite3.connect(appmod.LOG_DB) as cx:
        cs.upsert_scans(cx, "k@x.com", [{"scan_date": "2026-05-01"}]); cx.commit()
    r2 = c.post(f"/api/portal/{token}/request-analysis", json={"scan_id": 1, "scan_date": "2026-05-01"})
    assert r2.get_json().get("reason") == "monthly_quota"


def test_paid_member_unlimited(tmp_path, monkeypatch):
    appmod = _app(tmp_path, monkeypatch)
    monkeypatch.setattr(appmod, "_is_paid_member", lambda e: True)
    token = _mint(appmod, "p@x.com")
    if not token: pytest.skip("no mint")
    c = appmod.app.test_client()
    from dashboard import client_scans as cs
    with sqlite3.connect(appmod.LOG_DB) as cx:
        cs.upsert_scans(cx, "p@x.com", [{"scan_date": "2026-05-01"}]); cx.commit()
    assert c.post(f"/api/portal/{token}/request-analysis", json={"scan_id": 9, "scan_date": "2026-06-28"}).get_json()["status"] == "pending"
    assert c.post(f"/api/portal/{token}/request-analysis", json={"scan_id": 1, "scan_date": "2026-05-01"}).get_json()["status"] == "pending"


def test_requested_flag_and_flag_off(tmp_path, monkeypatch):
    appmod = _app(tmp_path, monkeypatch)
    monkeypatch.setattr(appmod, "_is_paid_member", lambda e: True)
    token = _mint(appmod, "k@x.com")
    if not token: pytest.skip("no mint")
    c = appmod.app.test_client()
    c.post(f"/api/portal/{token}/request-analysis", json={"scan_id": 9, "scan_date": "2026-06-28"})
    j = c.get(f"/api/portal/{token}").get_json()
    req = {s["scan_date"]: s.get("requested") for s in j.get("available_scans", [])}
    assert req.get("2026-06-28") is True
    # flag off → endpoint inert
    appmod2 = _app(tmp_path, monkeypatch, flag="0")
    c2 = appmod2.app.test_client()
    assert c2.post(f"/api/portal/{token}/request-analysis", json={"scan_id": 9, "scan_date": "2026-06-28"}).get_json()["status"] == "disabled"


def test_failed_requeue_no_new_charge(tmp_path, monkeypatch):
    appmod = _app(tmp_path, monkeypatch)
    monkeypatch.setattr(appmod, "_is_paid_member", lambda e: False)
    token = _mint(appmod, "f@x.com")
    if not token: pytest.skip("no mint")
    c = appmod.app.test_client()
    r = c.post(f"/api/portal/{token}/request-analysis", json={"scan_id": 9, "scan_date": "2026-06-28"})
    assert r.get_json()["status"] == "pending"
    from dashboard import analysis_requests as ar, analysis_quota as aq
    with sqlite3.connect(appmod.LOG_DB) as cx:
        row = cx.execute("SELECT id FROM analysis_requests WHERE email=? AND scan_date=?",
                         ("f@x.com", "2026-06-28")).fetchone()
        ar.mark(cx, row[0], "failed")
    # re-request the SAME scan → re-queues to pending, no new charge attempted
    r2 = c.post(f"/api/portal/{token}/request-analysis", json={"scan_id": 9, "scan_date": "2026-06-28"})
    assert r2.get_json()["status"] == "pending"
    with sqlite3.connect(appmod.LOG_DB) as cx:
        # exactly one claim total this month — the failed-requeue never re-claimed
        assert aq.claimed_this_month(cx, "f@x.com") is True
        st = cx.execute("SELECT status FROM analysis_requests WHERE email=? AND scan_date=?",
                        ("f@x.com", "2026-06-28")).fetchone()[0]
        assert st == "pending"


def test_member_query_household_and_unlinked_fallback(tmp_path, monkeypatch):
    monkeypatch.setenv("HOUSEHOLD_VIEW_ENABLED", "1")
    appmod = _app(tmp_path, monkeypatch)
    monkeypatch.setattr(appmod, "_is_paid_member", lambda e: False)
    from dashboard import household as hh, client_scans as cs, analysis_quota as aq
    primary, member = "care@x.com", "kid@x.com"
    token = _mint(appmod, primary, scan_date="2026-06-01")
    if not token: pytest.skip("no mint")
    with sqlite3.connect(appmod.LOG_DB) as cx:
        hh.init_household_tables(cx)
        hh.add_member(cx, primary, member, "Kid", "child")
        cs.upsert_scans(cx, member, [{"scan_date": "2026-06-15", "scan_id": 5}])
        cx.commit()
    c = appmod.app.test_client()
    # linked ?member= claims the MEMBER's quota, not the primary's
    r = c.post(f"/api/portal/{token}/request-analysis?member={member}",
               json={"scan_id": 5, "scan_date": "2026-06-15"})
    assert r.get_json()["status"] == "pending"
    with sqlite3.connect(appmod.LOG_DB) as cx:
        assert aq.claimed_this_month(cx, member) is True
        assert aq.claimed_this_month(cx, primary) is False
    # unlinked ?member= falls back to the primary (no leak) → records against the primary
    r2 = c.post(f"/api/portal/{token}/request-analysis?member=stranger@x.com",
                json={"scan_id": 9, "scan_date": "2026-06-01"})
    assert r2.get_json()["status"] == "pending"
    with sqlite3.connect(appmod.LOG_DB) as cx:
        assert aq.claimed_this_month(cx, primary) is True
        assert aq.claimed_this_month(cx, "stranger@x.com") is False
