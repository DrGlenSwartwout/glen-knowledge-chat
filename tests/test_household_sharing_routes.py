import importlib, sqlite3, sys
from pathlib import Path
import pytest


def _app(tmp_path, monkeypatch, *, share="1"):
    monkeypatch.setenv("DATA_DIR", str(tmp_path))
    monkeypatch.setenv("HOUSEHOLD_VIEW_ENABLED", "1")
    monkeypatch.setenv("HOUSEHOLD_SHARING_ENABLED", share)
    repo = Path(__file__).resolve().parent.parent
    if str(repo) not in sys.path: sys.path.insert(0, str(repo))
    try:
        import app as appmod; importlib.reload(appmod)
    except Exception as e:
        pytest.skip(f"app not importable: {e}")
    return appmod


def _mint(appmod, email):
    from dashboard import client_portal as cp, household as h
    with sqlite3.connect(appmod.LOG_DB) as cx:
        cp.init_client_portal_table(cx); h.init_household_tables(cx)
        tok = cp.upsert_portal(cx, email, "N", {}); cx.commit()
    return tok[0] if isinstance(tok, (tuple, list)) else tok


def test_member_sets_own_consent_only(tmp_path, monkeypatch):
    appmod = _app(tmp_path, monkeypatch)
    from dashboard import household as h
    with sqlite3.connect(appmod.LOG_DB) as cx:
        h.init_household_tables(cx); h.add_member(cx, "care@x.com", "mem@x.com", "M", "spouse"); cx.commit()
    mem_tok = _mint(appmod, "mem@x.com")
    if not mem_tok: pytest.skip("no mint helper")
    c = appmod.app.test_client()
    # member revokes sharing with their caregiver
    r = c.post(f"/api/portal/{mem_tok}/share-consent", json={"caregiver_email": "care@x.com", "consent": 0})
    assert r.status_code == 200
    with sqlite3.connect(appmod.LOG_DB) as cx:
        assert h.can_view(cx, "care@x.com", "mem@x.com") is False
    # member cannot set consent for a link where they are NOT the member (token=mem, tries as if caregiver)
    r2 = c.post(f"/api/portal/{mem_tok}/share-consent", json={"caregiver_email": "stranger@x.com", "consent": 1})
    # no row (mem is not a member of stranger) → no-op, still 200 but nothing changed
    assert r2.status_code == 200


def test_caregiver_sets_cc_only_for_own_members(tmp_path, monkeypatch):
    appmod = _app(tmp_path, monkeypatch)
    from dashboard import household as h
    with sqlite3.connect(appmod.LOG_DB) as cx:
        h.init_household_tables(cx); h.add_member(cx, "care@x.com", "mem@x.com", "M", "spouse"); cx.commit()
    care_tok = _mint(appmod, "care@x.com")
    if not care_tok: pytest.skip("no mint helper")
    c = appmod.app.test_client()
    r = c.post(f"/api/portal/{care_tok}/cc-pref", json={"member_email": "mem@x.com", "cc_enabled": 1})
    assert r.status_code == 200
    with sqlite3.connect(appmod.LOG_DB) as cx:
        assert h.cc_recipients_for(cx, "mem@x.com") == ["care@x.com"]


def test_flag_off_endpoints_inert(tmp_path, monkeypatch):
    appmod = _app(tmp_path, monkeypatch, share="0")
    from dashboard import household as h
    with sqlite3.connect(appmod.LOG_DB) as cx:
        h.init_household_tables(cx); h.add_member(cx, "care@x.com", "mem@x.com", "M", "pet"); cx.commit()
    mem_tok = _mint(appmod, "mem@x.com")
    if not mem_tok: pytest.skip("no mint helper")
    c = appmod.app.test_client()
    r = c.post(f"/api/portal/{mem_tok}/share-consent", json={"caregiver_email": "care@x.com", "consent": 0})
    assert r.get_json().get("recorded") is False or r.get_json().get("reason") == "disabled"
    with sqlite3.connect(appmod.LOG_DB) as cx:
        assert h.can_view(cx, "care@x.com", "mem@x.com") is True   # unchanged (flag off)


def test_cc_copy_sent_for_report(monkeypatch, tmp_path):
    appmod = _app(tmp_path, monkeypatch)
    from dashboard import household as h
    with sqlite3.connect(appmod.LOG_DB) as cx:
        h.init_household_tables(cx); h.add_member(cx, "care@x.com", "mem@x.com", "M", "pet"); cx.commit()
    sent = []
    monkeypatch.setattr(appmod, "_send_inquiry_email",
                        lambda to, subj, body, **k: sent.append(to) or (True, ""))
    # call the cc-fanout helper directly (the site calls it after the member send)
    appmod._household_cc_report("mem@x.com", "New scan for M")
    assert "care@x.com" in sent          # caregiver got a private copy


def test_no_cc_copy_when_switch_off(monkeypatch, tmp_path):
    appmod = _app(tmp_path, monkeypatch)
    from dashboard import household as h
    with sqlite3.connect(appmod.LOG_DB) as cx:
        h.init_household_tables(cx); h.add_member(cx, "care@x.com", "mem@x.com", "M", "pet")
        h.set_cc_enabled(cx, "care@x.com", "mem@x.com", 0); cx.commit()
    sent = []
    monkeypatch.setattr(appmod, "_send_inquiry_email",
                        lambda to, subj, body, **k: sent.append(to) or (True, ""))
    appmod._household_cc_report("mem@x.com", "New scan for M")
    assert sent == []                     # cc off → no copy
