import sqlite3

import pytest


@pytest.fixture
def appmod(monkeypatch, tmp_path):
    # Do NOT importlib.reload(app): see tests/test_courses_webhook.py for why —
    # a reload re-runs app.py's module bootstrap (BackgroundScheduler + prewarm)
    # which leaks into the rest of the suite. Redirect module globals instead.
    import app as m
    monkeypatch.setattr(m, "LOG_DB", tmp_path / "chat_log.db")
    return m


def _status(appmod, email, course, module):
    from dashboard import module_certifications as mc
    with sqlite3.connect(appmod.LOG_DB) as cx:
        mc.init_table(cx)
        return mc.status_for(cx, email, course, module)


def _count(appmod):
    with sqlite3.connect(appmod.LOG_DB) as cx:
        cx.execute("CREATE TABLE IF NOT EXISTS module_certifications("
                   "email TEXT, course TEXT, module TEXT, status TEXT DEFAULT 'pending', "
                   "stripe_ref TEXT, amount_cents INTEGER, created_at TEXT, approved_at TEXT)")
        return cx.execute("SELECT COUNT(*) FROM module_certifications").fetchone()[0]


def _session(**overrides):
    base = {
        "id": "cs_modcert_1",
        "amount_total": 20000,
        "metadata": {"kind": "module_certification", "email": "buyer@x.com",
                     "course": "ash-certification", "module": "02-body"},
        "customer_details": {"email": "buyer@x.com"},
    }
    base.update(overrides)
    return base


def test_valid_session_records_one_pending_row(appmod):
    session = _session()
    result = appmod._fulfill_module_certification(session)
    assert result == "ok"
    assert _status(appmod, "buyer@x.com", "ash-certification", "02-body") == "pending"
    assert _count(appmod) == 1


def test_replayed_session_id_does_not_double_record(appmod):
    session = _session()
    r1 = appmod._fulfill_module_certification(session)
    r2 = appmod._fulfill_module_certification(session)
    assert r1 == "ok"
    assert r2 == "ok"
    assert _count(appmod) == 1
    assert _status(appmod, "buyer@x.com", "ash-certification", "02-body") == "pending"


def test_wrong_kind_is_ignored(appmod):
    session = _session(metadata={"kind": "course_purchase", "email": "buyer@x.com",
                                  "course": "ash-certification", "module": "02-body"})
    result = appmod._fulfill_module_certification(session)
    assert result == "skip"
    assert _count(appmod) == 0


def test_missing_email_is_skipped(appmod):
    session = _session(metadata={"kind": "module_certification",
                                  "course": "ash-certification", "module": "02-body"},
                        customer_details={})
    result = appmod._fulfill_module_certification(session)
    assert result == "skip"
    assert _count(appmod) == 0


def test_missing_course_is_skipped(appmod):
    session = _session(metadata={"kind": "module_certification", "email": "buyer@x.com",
                                  "module": "02-body"})
    result = appmod._fulfill_module_certification(session)
    assert result == "skip"
    assert _count(appmod) == 0


def test_missing_module_is_skipped(appmod):
    session = _session(metadata={"kind": "module_certification", "email": "buyer@x.com",
                                  "course": "ash-certification"})
    result = appmod._fulfill_module_certification(session)
    assert result == "skip"
    assert _count(appmod) == 0
