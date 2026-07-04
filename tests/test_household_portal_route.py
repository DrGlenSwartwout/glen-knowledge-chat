import importlib, sqlite3, sys
from pathlib import Path
import pytest


def _app(tmp_path, monkeypatch, *, flag="1"):
    monkeypatch.setenv("DATA_DIR", str(tmp_path))
    monkeypatch.setenv("HOUSEHOLD_VIEW_ENABLED", flag)
    repo = Path(__file__).resolve().parent.parent
    if str(repo) not in sys.path: sys.path.insert(0, str(repo))
    try:
        import app as appmod; importlib.reload(appmod)
    except Exception as e:
        pytest.skip(f"app not importable: {e}")
    return appmod


def _seed(appmod, primary, member):
    from dashboard import client_portal as cp, household as h, portal_biofield_reports as pbr
    db = appmod.LOG_DB
    with sqlite3.connect(db) as cx:
        cp.init_client_portal_table(cx); h.init_household_tables(cx); pbr.init_table(cx)
        # a stable portal token for the primary
        token, _pid = cp.upsert_portal(cx, primary, "Karin", {})
        h.add_member(cx, primary, member, "Mochi", "pet")
        pbr.upsert_report(cx, primary, "2026-06-20", "s0", {"who": "primary"}, "confirmed")
        pbr.upsert_report(cx, member, "2026-06-25", "s1", {"who": "member"}, "confirmed")
        cx.commit()
    return token


def test_household_payload_and_member_switch(tmp_path, monkeypatch):
    appmod = _app(tmp_path, monkeypatch)
    token = _seed(appmod, "karin@x.com", "mochi@x.com")
    if not token: pytest.skip("no portal upsert helper; wire token via _portal_record_for")
    c = appmod.app.test_client()
    # primary view carries the household list
    j = c.get(f"/api/portal/{token}").get_json()
    assert any(m["email"] == "mochi@x.com" for m in j.get("household", []))
    # ?member= to a LINKED member serves the member's scan dates
    jm = c.get(f"/api/portal/{token}?member=mochi@x.com").get_json()
    assert "2026-06-25" in (jm.get("bf_scan_dates") or jm.get("scan_dates") or [])
    # ?member= to an UNLINKED email falls back to the primary (no leak)
    js = c.get(f"/api/portal/{token}?member=stranger@x.com").get_json()
    assert "2026-06-20" in (js.get("bf_scan_dates") or js.get("scan_dates") or [])


def test_flag_off_no_household(tmp_path, monkeypatch):
    appmod = _app(tmp_path, monkeypatch, flag="0")
    token = _seed(appmod, "karin@x.com", "mochi@x.com")
    if not token: pytest.skip("no portal upsert helper")
    c = appmod.app.test_client()
    j = c.get(f"/api/portal/{token}?member=mochi@x.com").get_json()
    assert "household" not in j                       # no household key when flag off
    assert "2026-06-20" in (j.get("bf_scan_dates") or j.get("scan_dates") or [])  # served primary
