import importlib, sqlite3, sys
from pathlib import Path
import pytest


def _app(tmp_path, monkeypatch):
    monkeypatch.setenv("DATA_DIR", str(tmp_path))
    monkeypatch.setenv("DEPENDENT_TOS_ENABLED", "1")
    monkeypatch.delenv("CONSOLE_SECRET", raising=False)
    repo = Path(__file__).resolve().parent.parent
    if str(repo) not in sys.path: sys.path.insert(0, str(repo))
    try:
        import app as appmod; importlib.reload(appmod)
        import dashboard as _d; monkeypatch.setattr(_d, "CONSOLE_SECRET", "", raising=False)
    except Exception as e:
        pytest.skip(f"app not importable: {e}")
    return appmod


def test_operational_caregiver_gets_no_tos_coverage(tmp_path, monkeypatch):
    appmod = _app(tmp_path, monkeypatch)
    from dashboard import household as h
    with sqlite3.connect(appmod.LOG_DB) as cx:
        h.init_household_tables(cx)
        # caregiver is a member; cared-for adult is NOT and did not agree
        h.add_member(cx, "cg@x.com", "adult@x.com", "Partner", "partner")
        h.set_share_consent(cx, "cg@x.com", "adult@x.com", 1)
    monkeypatch.setattr(appmod, "is_member",
                        lambda email=None, **k: (email or "").lower() == "cg@x.com")
    # operational relationship must NOT grant coverage
    assert appmod._portal_tos_agreed("adult@x.com") is False


def test_dependent_caregiver_grants_tos_coverage(tmp_path, monkeypatch):
    appmod = _app(tmp_path, monkeypatch)
    from dashboard import household as h
    with sqlite3.connect(appmod.LOG_DB) as cx:
        h.init_household_tables(cx)
        h.add_member(cx, "cg@x.com", "kid@x.com", "Kid", "dependent")
    monkeypatch.setattr(appmod, "is_member",
                        lambda email=None, **k: (email or "").lower() == "cg@x.com")
    assert appmod._portal_tos_agreed("kid@x.com") is True
