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
