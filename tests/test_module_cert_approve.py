import sqlite3

import pytest


@pytest.fixture
def appmod(monkeypatch, tmp_path):
    # Do NOT importlib.reload(app): a reload with DATA_DIR set re-runs app.py's
    # module bootstrap, which starts a BackgroundScheduler + prewarm daemon that
    # are never shut down and leak into the rest of the suite (timing-
    # nondeterministic bystander failures on CI). Redirect the module globals we
    # need directly instead — the proven pattern used by tests/test_courses_admin_grant.py.
    import app as m
    monkeypatch.setattr(m, "LOG_DB", tmp_path / "chat_log.db")
    monkeypatch.setattr(m, "CONSOLE_SECRET", "sekret")  # module global read by _console_key_ok
    m.app.config["TESTING"] = True
    return m


def _seed(appmod, email, course, approved_modules, pending_modules):
    from dashboard import module_certifications as mc
    with sqlite3.connect(appmod.LOG_DB) as cx:
        mc.init_table(cx)
        for i, mod in enumerate(approved_modules):
            cx.execute(
                "INSERT INTO module_certifications(email, course, module, status, "
                "stripe_ref, amount_cents, created_at, approved_at) "
                "VALUES(?,?,?, 'approved', ?, 20000, '2026-07-01T00:00:00Z', '2026-07-02T00:00:00Z')",
                (email, course, mod, f"cs_{email}_{mod}_{i}"))
        for i, mod in enumerate(pending_modules):
            cx.execute(
                "INSERT INTO module_certifications(email, course, module, status, "
                "stripe_ref, amount_cents, created_at) "
                "VALUES(?,?,?, 'pending', ?, 20000, '2026-07-01T00:00:00Z')",
                (email, course, mod, f"cs_{email}_{mod}_p{i}"))
        cx.commit()


def _status(appmod, email, course, module):
    from dashboard import module_certifications as mc
    with sqlite3.connect(appmod.LOG_DB) as cx:
        return mc.status_for(cx, email, course, module)


def _paid_level(appmod, email):
    from dashboard import course_entitlements as ce
    with sqlite3.connect(appmod.LOG_DB) as cx:
        ce.init_course_entitlements_table(cx)
        return ce.paid_level_for(cx, email)


def _cert_row_count(appmod, email):
    with sqlite3.connect(appmod.LOG_DB) as cx:
        cx.execute("CREATE TABLE IF NOT EXISTS course_entitlements("
                   "id INTEGER PRIMARY KEY, email TEXT, kind TEXT, status TEXT, "
                   "expires_at REAL, source TEXT, stripe_customer_id TEXT, "
                   "stripe_ref TEXT, created_at TEXT, updated_at TEXT)")
        return cx.execute(
            "SELECT COUNT(*) FROM course_entitlements WHERE email=? AND kind='cert_onetime'",
            (email,)).fetchone()[0]


ALL_MODULES = ["02-body", "03-mind", "04-spirit", "05-family-history",
    "06-health-history", "07-epigenetics", "08-symptoms", "09-terrain",
    "10-diagnoses", "11-treatment", "12-response", "13-prognosis"]
COURSE = "ash-certification"


def test_approve_flips_pending_to_approved(appmod):
    email = "student@x.com"
    _seed(appmod, email, COURSE, [], ["02-body"])
    r = appmod.app.test_client().post(
        "/api/console/module-certs/approve",
        json={"email": email, "course": COURSE, "module": "02-body"},
        headers={"X-Console-Key": "sekret"})
    assert r.status_code == 200
    data = r.get_json()
    assert data["ok"] is True
    assert _status(appmod, email, COURSE, "02-body") == "approved"


def test_approve_requires_console_key(appmod):
    email = "nokeytest@x.com"
    _seed(appmod, email, COURSE, [], ["02-body"])
    r = appmod.app.test_client().post(
        "/api/console/module-certs/approve",
        json={"email": email, "course": COURSE, "module": "02-body"})
    assert r.status_code in (401, 403)
    # Must not have been approved.
    assert _status(appmod, email, COURSE, "02-body") == "pending"


def test_approving_final_module_grants_full_certification(appmod):
    email = "graduate@x.com"
    approved = ALL_MODULES[:11]
    final = ALL_MODULES[11]
    _seed(appmod, email, COURSE, approved, [final])
    r = appmod.app.test_client().post(
        "/api/console/module-certs/approve",
        json={"email": email, "course": COURSE, "module": final},
        headers={"X-Console-Key": "sekret"})
    assert r.status_code == 200
    data = r.get_json()
    assert data["ok"] is True
    assert data["full_certification_granted"] is True
    assert data["certified_count"] == 12
    assert _paid_level(appmod, email) == 2


def test_approving_non_final_module_does_not_grant_certification(appmod):
    email = "midway@x.com"
    approved = ALL_MODULES[:5]
    pending = ALL_MODULES[5:]
    _seed(appmod, email, COURSE, approved, pending)
    r = appmod.app.test_client().post(
        "/api/console/module-certs/approve",
        json={"email": email, "course": COURSE, "module": pending[0]},
        headers={"X-Console-Key": "sekret"})
    assert r.status_code == 200
    data = r.get_json()
    assert data["ok"] is True
    assert data["full_certification_granted"] is False
    assert data["certified_count"] == 6
    assert _paid_level(appmod, email) != 2


def test_reapproving_after_full_grant_does_not_double_grant(appmod):
    email = "regrad@x.com"
    approved = ALL_MODULES[:11]
    final = ALL_MODULES[11]
    _seed(appmod, email, COURSE, approved, [final])
    client = appmod.app.test_client()
    r1 = client.post(
        "/api/console/module-certs/approve",
        json={"email": email, "course": COURSE, "module": final},
        headers={"X-Console-Key": "sekret"})
    assert r1.status_code == 200
    assert r1.get_json()["full_certification_granted"] is True
    # Re-approve the same (already-approved) module — must not error and must
    # not double-grant the credential.
    r2 = client.post(
        "/api/console/module-certs/approve",
        json={"email": email, "course": COURSE, "module": final},
        headers={"X-Console-Key": "sekret"})
    assert r2.status_code == 200
    data2 = r2.get_json()
    assert data2["ok"] is False  # already approved — approve() flips pending only
    assert data2["full_certification_granted"] is True  # still fully certified
    assert _paid_level(appmod, email) == 2
    assert _cert_row_count(appmod, email) == 1


def test_pending_list_endpoint(appmod):
    email = "lister@x.com"
    _seed(appmod, email, COURSE, ["02-body"], ["03-mind", "04-spirit"])
    r = appmod.app.test_client().get(
        "/api/console/module-certs?status=pending",
        headers={"X-Console-Key": "sekret"})
    assert r.status_code == 200
    items = r.get_json()["items"]
    modules = {i["module"] for i in items if i["email"] == email}
    assert modules == {"03-mind", "04-spirit"}
    for i in items:
        if i["email"] == email:
            assert i["status"] == "pending"
            assert i["course"] == COURSE
