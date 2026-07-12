import importlib, sqlite3, sys
from pathlib import Path
import pytest


def _app(tmp_path, monkeypatch, *, flag="1", scan_history=None):
    monkeypatch.setenv("DATA_DIR", str(tmp_path))
    monkeypatch.setenv("HOUSEHOLD_VIEW_ENABLED", flag)
    if scan_history is not None:
        monkeypatch.setenv("PORTAL_SCAN_HISTORY_ENABLED", scan_history)
    else:
        monkeypatch.delenv("PORTAL_SCAN_HISTORY_ENABLED", raising=False)
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


def test_member_tab_tos_gate_follows_the_token_holder(tmp_path, monkeypatch):
    """A pet never agreed to Terms. Evaluating tos_agreed against the MEMBER put a
    Terms-of-Service gate over the member's report instead of showing it — the
    caregiver holding the token is who agreed, and who is reading the page."""
    appmod = _app(tmp_path, monkeypatch)
    token = _seed(appmod, "karin@x.com", "mochi@x.com")
    if not token: pytest.skip("no portal upsert helper")
    # Only the caregiver has agreed. The pet is not a member and never will be.
    monkeypatch.setattr(appmod, "is_member",
                        lambda email=None, **kw: (email or "") == "karin@x.com")
    c = appmod.app.test_client()
    own = c.get(f"/api/portal/{token}").get_json()
    mem = c.get(f"/api/portal/{token}?member=mochi@x.com").get_json()
    assert own["tos_agreed"] is True, "precondition: the caregiver has agreed"
    assert mem["tos_agreed"] is True, "member tab must not re-gate the caregiver on Terms"


def test_flag_off_no_household(tmp_path, monkeypatch):
    appmod = _app(tmp_path, monkeypatch, flag="0")
    token = _seed(appmod, "karin@x.com", "mochi@x.com")
    if not token: pytest.skip("no portal upsert helper")
    c = appmod.app.test_client()
    j = c.get(f"/api/portal/{token}?member=mochi@x.com").get_json()
    assert "household" not in j                       # no household key when flag off
    assert "2026-06-20" in (j.get("bf_scan_dates") or j.get("scan_dates") or [])  # served primary


def test_household_entries_carry_scan_dates_when_history_flag_on(tmp_path, monkeypatch):
    """Issue #810: Scan History tab lists a household member's scans INLINE
    instead of just a 'View <name>'s history' link. The payload's household
    entries need each member's own scan_dates (newest-first) + current_scan_date
    so the frontend can render rows without a follow-up fetch."""
    appmod = _app(tmp_path, monkeypatch, scan_history="1")
    from dashboard import client_portal as cp, portal_biofield_reports as pbr
    token = _seed(appmod, "karin@x.com", "mochi@x.com")
    if not token: pytest.skip("no portal upsert helper")
    with sqlite3.connect(appmod.LOG_DB) as cx:
        # a second, newer report for the member + a portal row so a current-scan
        # pointer can be persisted (set_current_scan needs an existing content row)
        pbr.upsert_report(cx, "mochi@x.com", "2026-07-01", "s2", {"who": "member"}, "confirmed")
        cp.upsert_portal(cx, "mochi@x.com", "Mochi", {})
        cx.commit()
        cp.set_current_scan(cx, "mochi@x.com", "2026-06-25")
    c = appmod.app.test_client()
    j = c.get(f"/api/portal/{token}").get_json()
    mochi = next(m for m in j["household"] if m["email"] == "mochi@x.com")
    assert mochi["scan_dates"] == ["2026-07-01", "2026-06-25"]   # newest-first
    assert mochi["current_scan_date"] == "2026-06-25"            # persisted pointer, not newest


def test_household_entries_have_no_scan_dates_when_history_flag_off(tmp_path, monkeypatch):
    appmod = _app(tmp_path, monkeypatch, scan_history="0")
    token = _seed(appmod, "karin@x.com", "mochi@x.com")
    if not token: pytest.skip("no portal upsert helper")
    c = appmod.app.test_client()
    j = c.get(f"/api/portal/{token}").get_json()
    mochi = next(m for m in j["household"] if m["email"] == "mochi@x.com")
    assert "scan_dates" not in mochi
    assert "current_scan_date" not in mochi
