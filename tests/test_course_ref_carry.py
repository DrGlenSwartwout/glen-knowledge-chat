"""Task 2 of the ambassador-commission slice: carry the `?ref=<slug>` affiliate
slug from the illtowell.com certification landing page across the un-cookied
courses.mentorshipu.com host, into the Stripe checkout metadata for both
course-purchase checkout and module-certification checkout. Crediting itself
is a later task; this only proves the ref lands in Stripe metadata."""
import importlib
import sqlite3

import pytest

from tests.courses_fixture import write_sample_course

_MHOST = "http://mentorshipu.test"
_COURSE = "ash-intro"
_MODULE = "03-pro"
_LESSON = "01-advanced"
_CERT_URL = f"/api/courses/{_COURSE}/{_MODULE}/certify"


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


def _stripe_on(monkeypatch):
    monkeypatch.setenv("STRIPE_ACTIVE", "1")
    monkeypatch.setenv("STRIPE_CERT_PRICE_ID", "price_cert")
    monkeypatch.setenv("STRIPE_MEMBERSHIP_PRICE_ID", "price_mem")
    monkeypatch.setenv("STRIPE_MODULE_CERT_PRICE_ID", "price_modcert")


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


def _fake_checkout(monkeypatch):
    from dashboard import stripe_pay
    captured = {}

    def fake_create(price_id, **kwargs):
        captured["price_id"] = price_id
        captured.update(kwargs)
        return {"id": "cs_test", "url": "https://stripe.test/cs"}

    monkeypatch.setattr(stripe_pay, "create_price_checkout_session", fake_create)
    return captured


def test_checkout_membership_carries_ref_into_session_and_subscription_metadata(client, monkeypatch):
    c, appmod = client
    _stripe_on(monkeypatch)
    captured = _fake_checkout(monkeypatch)

    r = c.post("/api/courses/checkout", json={"product": "membership", "ref": "amb1"}, base_url=_MHOST)
    assert r.status_code == 200

    assert captured["metadata"]["ref"] == "amb1"
    assert captured["subscription_metadata"]["ref"] == "amb1"


def test_checkout_onetime_carries_ref_into_session_metadata_only(client, monkeypatch):
    c, appmod = client
    _stripe_on(monkeypatch)
    captured = _fake_checkout(monkeypatch)

    r = c.post("/api/courses/checkout", json={"product": "onetime", "ref": "amb1"}, base_url=_MHOST)
    assert r.status_code == 200

    assert captured["metadata"]["ref"] == "amb1"
    assert captured["subscription_metadata"] is None  # payment mode, no subscription


def test_checkout_invalid_ref_is_dropped(client, monkeypatch):
    c, appmod = client
    _stripe_on(monkeypatch)
    captured = _fake_checkout(monkeypatch)

    r = c.post("/api/courses/checkout", json={"product": "onetime", "ref": "bad ref!"}, base_url=_MHOST)
    assert r.status_code == 200

    assert "ref" not in captured["metadata"]


def test_checkout_no_ref_is_omitted(client, monkeypatch):
    c, appmod = client
    _stripe_on(monkeypatch)
    captured = _fake_checkout(monkeypatch)

    r = c.post("/api/courses/checkout", json={"product": "onetime"}, base_url=_MHOST)
    assert r.status_code == 200

    assert "ref" not in captured["metadata"]


def test_certify_module_carries_ref_into_metadata(client, monkeypatch):
    c, appmod = client
    _stripe_on(monkeypatch)
    email = "amb-learner@example.com"
    tok = _mint_token(appmod, email)
    _complete_module(appmod, email)
    captured = _fake_checkout(monkeypatch)

    r = c.post(f"{_CERT_URL}?token={tok}", json={"ref": "amb1"}, base_url=_MHOST)
    assert r.status_code == 200

    assert captured["metadata"]["ref"] == "amb1"
    assert captured["metadata"]["kind"] == "module_certification"


def test_certify_module_invalid_ref_is_dropped(client, monkeypatch):
    c, appmod = client
    _stripe_on(monkeypatch)
    email = "amb-learner2@example.com"
    tok = _mint_token(appmod, email)
    _complete_module(appmod, email)
    captured = _fake_checkout(monkeypatch)

    r = c.post(f"{_CERT_URL}?token={tok}", json={"ref": "bad ref!"}, base_url=_MHOST)
    assert r.status_code == 200

    assert "ref" not in captured["metadata"]


def test_certification_landing_page_has_ref_carry_script():
    with open("static/certification.html") as f:
        html = f.read()
    assert "/ref-capture.js" in html
    # ref-carry script must reference getRef() to pull the captured slug
    idx = html.index("/ref-capture.js")
    assert "getRef" in html[idx:]


def test_course_page_template_loads_ref_capture_js():
    import courses_blueprint as cb
    assert "ref-capture.js" in cb._PAGE
