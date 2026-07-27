import importlib
import sqlite3

import pytest

from tests.courses_fixture import write_sample_course

_MHOST = "http://mentorshipu.test"
_COURSE = "ash-intro"
_MODULE = "03-pro"
_LESSON = "01-advanced"
_URL = f"/api/courses/{_COURSE}/{_MODULE}/certify"


@pytest.fixture
def client(monkeypatch, tmp_path):
    monkeypatch.setenv("DATA_DIR", str(tmp_path))
    monkeypatch.setenv("COURSES_ROOT", str(tmp_path / "courses"))
    monkeypatch.setenv("MENTORSHIP_BASE_URL", _MHOST)
    write_sample_course(str(tmp_path / "courses"))
    import app as appmod
    importlib.reload(appmod)
    monkeypatch.setattr(appmod, "send_mentorship_setup_link", lambda *a, **k: ("test", None))
    appmod.app.config["TESTING"] = True
    return appmod.app.test_client(), appmod


def _mint_token(appmod, email):
    from dashboard import course_tokens
    with sqlite3.connect(appmod.LOG_DB) as cx:
        course_tokens.init_course_tokens_table(cx)
        return course_tokens.mint_course_token(cx, email, "T")


def _complete_module(appmod, email):
    from dashboard import course_progress as cp
    with sqlite3.connect(appmod.LOG_DB) as cx:
        cp.mark_watched(cx, email, _COURSE, _MODULE, _LESSON)
        cp.record_homework(cx, email, _COURSE, _MODULE, "my takeaways")


def _seed_pending(appmod, email):
    from dashboard import module_certifications as mc
    with sqlite3.connect(appmod.LOG_DB) as cx:
        mc.record_purchase(cx, email, _COURSE, _MODULE, "ch_seed", 20000)


def _stripe_on(monkeypatch):
    monkeypatch.setenv("STRIPE_ACTIVE", "1")
    monkeypatch.setenv("STRIPE_MODULE_CERT_PRICE_ID", "price_modcert")


def test_no_token_is_401(client):
    c, _ = client
    r = c.post(_URL, base_url=_MHOST)
    assert r.status_code == 401
    assert r.get_json() == {"error": "unauthorized"}


def test_price_unset_is_503(client, monkeypatch):
    c, appmod = client
    monkeypatch.setenv("STRIPE_ACTIVE", "1")
    monkeypatch.delenv("STRIPE_MODULE_CERT_PRICE_ID", raising=False)
    tok = _mint_token(appmod, "a@example.com")
    r = c.post(f"{_URL}?token={tok}", base_url=_MHOST)
    assert r.status_code == 503
    assert r.get_json() == {"error": "not available"}


def test_module_not_completed_is_403(client, monkeypatch):
    c, appmod = client
    _stripe_on(monkeypatch)
    tok = _mint_token(appmod, "b@example.com")
    r = c.post(f"{_URL}?token={tok}", base_url=_MHOST)
    assert r.status_code == 403
    assert r.get_json() == {"error": "module not completed"}


def test_already_pending_is_409(client, monkeypatch):
    c, appmod = client
    _stripe_on(monkeypatch)
    email = "c@example.com"
    tok = _mint_token(appmod, email)
    _complete_module(appmod, email)
    _seed_pending(appmod, email)
    r = c.post(f"{_URL}?token={tok}", base_url=_MHOST)
    assert r.status_code == 409
    assert r.get_json() == {"error": "already pending"}


def test_happy_path_returns_checkout_url(client, monkeypatch):
    c, appmod = client
    _stripe_on(monkeypatch)
    email = "d@example.com"
    tok = _mint_token(appmod, email)
    _complete_module(appmod, email)

    from dashboard import stripe_pay
    captured = {}

    def fake_create(price_id, **kwargs):
        captured["price_id"] = price_id
        captured.update(kwargs)
        return {"id": "cs_modcert", "url": "https://stripe.test/cs"}

    monkeypatch.setattr(stripe_pay, "create_price_checkout_session", fake_create)

    r = c.post(f"{_URL}?token={tok}", base_url=_MHOST)
    assert r.status_code == 200
    assert r.get_json() == {"url": "https://stripe.test/cs"}

    assert captured["price_id"] == "price_modcert"
    assert captured["mode"] == "payment"
    assert captured["customer_email"] == email
    assert captured["metadata"] == {
        "kind": "module_certification",
        "email": email,
        "course": _COURSE,
        "module": _MODULE,
    }
    assert captured["success_url"] == f"{_MHOST}/learn/{_COURSE}/{_MODULE}?certified=1"
    assert captured["cancel_url"] == f"{_MHOST}/learn/{_COURSE}/{_MODULE}"
