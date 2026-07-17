import importlib, sqlite3, sys
from pathlib import Path
import pytest


def _app(tmp_path, monkeypatch):
    monkeypatch.setenv("DATA_DIR", str(tmp_path))
    monkeypatch.delenv("CONSOLE_SECRET", raising=False)
    repo = Path(__file__).resolve().parent.parent
    if str(repo) not in sys.path: sys.path.insert(0, str(repo))
    try:
        import app as appmod; importlib.reload(appmod)
        import dashboard as _d; monkeypatch.setattr(_d, "CONSOLE_SECRET", "", raising=False)
        # _check_console_or_scoped_auth() requires a non-empty X-Console-Key,
        # so use the repo's real-key convention (see tests/test_household_api.py)
        # rather than an empty-key bypass.
        monkeypatch.setattr(appmod, "CONSOLE_SECRET", "testkey")
        # GHL push is best-effort; stub it so tests don't hit the network.
        monkeypatch.setattr(appmod, "_push_household_tags_to_ghl",
                            lambda *a, **k: (True, None), raising=False)
    except Exception as e:
        pytest.skip(f"app not importable: {e}")
    return appmod


def _person(cx, email, first, last):
    cx.execute("INSERT INTO people (email, first_name, last_name) VALUES (?,?,?)",
               (email, first, last))
    return cx.execute("SELECT id FROM people WHERE email=?", (email,)).fetchone()[0]


def _seed(appmod):
    with sqlite3.connect(appmod.LOG_DB) as cx:
        a = _person(cx, "a@x.com", "Ann", "Lee")
        b = _person(cx, "b@x.com", "Bo", "Reyes")
        cx.commit()
    return a, b


def test_connect_member_creates_grouping_only(tmp_path, monkeypatch):
    appmod = _app(tmp_path, monkeypatch)
    a, b = _seed(appmod)
    c = appmod.app.test_client()
    r = c.post(f"/api/people/{a}/connect",
               json={"other_person_id": b, "mode": "member"},
               headers={"X-Console-Key": "testkey"})
    assert r.get_json()["ok"] is True
    from dashboard import household as h
    with sqlite3.connect(appmod.LOG_DB) as cx:
        assert h.members_for(cx, "a@x.com") == []  # no caregiver link
        assert appmod._person_household_slug(cx, a)  # grouped


def test_connect_operational_caregiver_defaults_dark(tmp_path, monkeypatch):
    appmod = _app(tmp_path, monkeypatch)
    a, b = _seed(appmod)
    c = appmod.app.test_client()
    r = c.post(f"/api/people/{a}/connect",
               json={"other_person_id": b, "mode": "caregiver",
                     "caregiver_person_id": a, "cared_for_person_id": b,
                     "relationship": "partner", "consent": {"method": "portal"}},
               headers={"X-Console-Key": "testkey"})
    assert r.get_json()["ok"] is True
    from dashboard import household as h
    with sqlite3.connect(appmod.LOG_DB) as cx:
        st = h.consent_state(cx, "a@x.com", "b@x.com")
        assert st["share_consent"] == 0  # dark until confirmed
        assert appmod._person_household_slug(cx, a)  # also grouped


def test_connect_operational_caregiver_verbal_is_active(tmp_path, monkeypatch):
    appmod = _app(tmp_path, monkeypatch)
    a, b = _seed(appmod)
    c = appmod.app.test_client()
    c.post(f"/api/people/{a}/connect",
           json={"other_person_id": b, "mode": "caregiver",
                 "caregiver_person_id": a, "cared_for_person_id": b,
                 "relationship": "partner", "consent": {"method": "verbal"}},
           headers={"X-Console-Key": "testkey"})
    from dashboard import household as h
    with sqlite3.connect(appmod.LOG_DB) as cx:
        st = h.consent_state(cx, "a@x.com", "b@x.com")
        assert st["share_consent"] == 1 and st["consent_basis"] == "verbal"
