# tests/test_analysis_quota_gate.py
"""Task 8: gate the two 'request an analysis' routes with the free-tier
monthly quota (dashboard/analysis_quota.py). Flag ANALYSIS_QUOTA_ENABLED
default OFF -> unchanged behavior. Paid members always bypass."""
import sqlite3
import pytest


@pytest.fixture
def client(monkeypatch, tmp_path):
    import app as appmod
    monkeypatch.setattr(appmod, "LOG_DB", str(tmp_path / "chat_log.db"))
    appmod._init_auth_tables()
    appmod.app.config["TESTING"] = True
    return appmod.app.test_client(), appmod


def _seed_portal(appmod, email="free@example.com", name="Free Tier"):
    from dashboard import client_portal as cp
    cx = sqlite3.connect(appmod.LOG_DB)
    cp.init_client_portal_table(cx)
    token, _ = cp.upsert_portal(cx, email, name, {"greeting": "hi"})
    cx.close()
    return token


def _status(appmod, email):
    from dashboard import client_portal as cp
    cx = sqlite3.connect(appmod.LOG_DB)
    cp.init_client_portal_table(cx)
    s = cp.get_biofield_status(cx, email)
    cx.close()
    return s


# ── /api/portal/<token>/biofield/request ─────────────────────────────────────

def test_portal_request_flag_off_unlimited(client, monkeypatch):
    c, appmod = client
    monkeypatch.setattr(appmod, "ANALYSIS_QUOTA_ENABLED", False, raising=False)
    tok = _seed_portal(appmod, "offflag@example.com")
    r1 = c.post(f"/api/portal/{tok}/biofield/request")
    assert r1.status_code == 200 and r1.get_json()["ok"] is True
    # reset status so a second successful transition is observable
    from dashboard import client_portal as cp
    cx = sqlite3.connect(appmod.LOG_DB)
    cp.set_biofield_status(cx, "offflag@example.com", "confirmed")
    cx.close()
    r2 = c.post(f"/api/portal/{tok}/biofield/request")
    assert r2.status_code == 200 and r2.get_json()["ok"] is True
    assert _status(appmod, "offflag@example.com") == "requested"


def test_portal_request_free_tier_second_blocked(client, monkeypatch):
    c, appmod = client
    monkeypatch.setattr(appmod, "ANALYSIS_QUOTA_ENABLED", True, raising=False)
    monkeypatch.setattr(appmod, "_active_membership_for_email", lambda e: None)
    email = "freeportal@example.com"
    tok = _seed_portal(appmod, email)

    r1 = c.post(f"/api/portal/{tok}/biofield/request")
    assert r1.status_code == 200 and r1.get_json()["ok"] is True
    assert _status(appmod, email) == "requested"

    # simulate the status having moved on since the first request, so a
    # second (blocked) call leaving it untouched is observable
    from dashboard import client_portal as cp
    cx = sqlite3.connect(appmod.LOG_DB)
    cp.set_biofield_status(cx, email, "confirmed")
    cx.close()

    r2 = c.post(f"/api/portal/{tok}/biofield/request")
    assert r2.status_code == 200
    j2 = r2.get_json()
    assert j2["ok"] is False and j2["reason"] == "monthly_quota"
    assert _status(appmod, email) == "confirmed"  # unchanged by the blocked request


def test_portal_request_paid_member_unlimited(client, monkeypatch):
    c, appmod = client
    monkeypatch.setattr(appmod, "ANALYSIS_QUOTA_ENABLED", True, raising=False)
    monkeypatch.setattr(appmod, "_active_membership_for_email", lambda e: {"email": e})
    email = "paidportal@example.com"
    tok = _seed_portal(appmod, email)

    for _ in range(3):
        from dashboard import client_portal as cp
        cx = sqlite3.connect(appmod.LOG_DB)
        cp.set_biofield_status(cx, email, "confirmed")
        cx.close()
        r = c.post(f"/api/portal/{tok}/biofield/request")
        assert r.status_code == 200 and r.get_json()["ok"] is True
        assert _status(appmod, email) == "requested"


def test_portal_request_non_actionable_does_not_consume_quota(client, monkeypatch):
    """Task 8 fix: a 409 (scan no longer actionable) must NOT burn the
    free-tier monthly claim - a subsequent VALID request the same month
    still succeeds."""
    c, appmod = client
    monkeypatch.setattr(appmod, "ANALYSIS_QUOTA_ENABLED", True, raising=False)
    monkeypatch.setattr(appmod, "_active_membership_for_email", lambda e: None)
    from dashboard import portal_biofield_reports as R
    import datetime
    email = "nonactionable@example.com"
    tok = _seed_portal(appmod, email)
    cx = sqlite3.connect(appmod.LOG_DB)
    R.init_table(cx)
    old = (datetime.date.today() - datetime.timedelta(days=60)).isoformat()
    today = datetime.date.today().isoformat()
    R.upsert_report(cx, email, old, "s0", {"layers": []}, "ai_draft")
    cx.close()

    r1 = c.post(f"/api/portal/{tok}/biofield/request", json={"scan_date": old})
    assert r1.status_code == 409

    cx = sqlite3.connect(appmod.LOG_DB)
    R.upsert_report(cx, email, today, "s1", {"layers": []}, "ai_draft")
    cx.close()

    r2 = c.post(f"/api/portal/{tok}/biofield/request", json={"scan_date": today})
    assert r2.status_code == 200
    j2 = r2.get_json()
    assert j2["ok"] is True and j2["status"] == "requested"


def test_portal_request_write_404_does_not_consume_quota(client, monkeypatch):
    """Defense-in-depth: even if the underlying status write unexpectedly
    404s after the actionability check passes, the free-tier claim must not
    be burned - a subsequent request the same month still succeeds."""
    c, appmod = client
    monkeypatch.setattr(appmod, "ANALYSIS_QUOTA_ENABLED", True, raising=False)
    monkeypatch.setattr(appmod, "_active_membership_for_email", lambda e: None)
    from dashboard import portal_biofield_reports as R
    import datetime
    email = "write404@example.com"
    tok = _seed_portal(appmod, email)
    cx = sqlite3.connect(appmod.LOG_DB)
    R.init_table(cx)
    today = datetime.date.today().isoformat()
    R.upsert_report(cx, email, today, "s1", {"layers": []}, "ai_draft")
    cx.close()

    real_set = R.set_report_status
    fail = {"on": True}

    def _flaky(cx, email, scan_date, status):
        if fail["on"]:
            return False
        return real_set(cx, email, scan_date, status)

    monkeypatch.setattr(R, "set_report_status", _flaky)

    r1 = c.post(f"/api/portal/{tok}/biofield/request", json={"scan_date": today})
    assert r1.status_code == 404

    fail["on"] = False
    r2 = c.post(f"/api/portal/{tok}/biofield/request", json={"scan_date": today})
    assert r2.status_code == 200
    j2 = r2.get_json()
    assert j2["ok"] is True and j2["status"] == "requested"


def test_portal_interest_route_not_gated(client, monkeypatch):
    """The quota only applies to the 'requested' transition, not 'interested'."""
    c, appmod = client
    monkeypatch.setattr(appmod, "ANALYSIS_QUOTA_ENABLED", True, raising=False)
    monkeypatch.setattr(appmod, "_active_membership_for_email", lambda e: None)
    email = "interest@example.com"
    tok = _seed_portal(appmod, email)
    r1 = c.post(f"/api/portal/{tok}/biofield/interest")
    r2 = c.post(f"/api/portal/{tok}/biofield/interest")
    assert r1.get_json()["ok"] is True
    assert r2.get_json()["ok"] is True  # never blocked


# ── /biofield/request ─────────────────────────────────────────────────────────

def test_biofield_request_flag_off_unlimited(client, monkeypatch):
    c, appmod = client
    monkeypatch.setattr(appmod, "ANALYSIS_QUOTA_ENABLED", False, raising=False)
    sent = []
    monkeypatch.setattr(appmod, "send_magic_link_email",
                        lambda to, name, url: sent.append(to))
    email = "offflag2@example.com"
    r1 = c.post("/biofield/request", json={"email": email})
    r2 = c.post("/biofield/request", json={"email": email})
    assert r1.get_json()["ok"] is True and r2.get_json()["ok"] is True
    assert len(sent) == 2


def test_biofield_request_free_tier_second_blocked(client, monkeypatch):
    c, appmod = client
    monkeypatch.setattr(appmod, "ANALYSIS_QUOTA_ENABLED", True, raising=False)
    monkeypatch.setattr(appmod, "_active_membership_for_email", lambda e: None)
    sent = []
    monkeypatch.setattr(appmod, "send_magic_link_email",
                        lambda to, name, url: sent.append(to))
    email = "freebio@example.com"

    r1 = c.post("/biofield/request", json={"email": email})
    assert r1.status_code == 200 and r1.get_json()["ok"] is True
    assert len(sent) == 1

    r2 = c.post("/biofield/request", json={"email": email})
    assert r2.status_code == 200
    j2 = r2.get_json()
    assert j2["ok"] is False and j2["reason"] == "monthly_quota"
    assert len(sent) == 1  # no second magic link sent


def test_biofield_request_send_failure_does_not_consume_quota(client, monkeypatch):
    """Task 8 fix: if send_magic_link_email raises (SMTP/network failure) after
    the claim, the free-tier claim must be released - a subsequent request the
    same month still goes through and actually sends."""
    c, appmod = client
    monkeypatch.setattr(appmod, "ANALYSIS_QUOTA_ENABLED", True, raising=False)
    monkeypatch.setattr(appmod, "_active_membership_for_email", lambda e: None)
    sent = []

    def _boom(to, name, url):
        raise RuntimeError("smtp down")

    monkeypatch.setattr(appmod, "send_magic_link_email", _boom)
    email = "flakybio@example.com"

    r1 = c.post("/biofield/request", json={"email": email})
    assert r1.status_code == 200 and r1.get_json()["ok"] is True  # always 200, no leak
    assert len(sent) == 0

    monkeypatch.setattr(appmod, "send_magic_link_email",
                        lambda to, name, url: sent.append(to))
    r2 = c.post("/biofield/request", json={"email": email})
    assert r2.status_code == 200 and r2.get_json()["ok"] is True
    assert len(sent) == 1  # the retried request actually sent this time


def test_biofield_request_paid_member_unlimited(client, monkeypatch):
    c, appmod = client
    monkeypatch.setattr(appmod, "ANALYSIS_QUOTA_ENABLED", True, raising=False)
    monkeypatch.setattr(appmod, "_active_membership_for_email", lambda e: {"email": e})
    sent = []
    monkeypatch.setattr(appmod, "send_magic_link_email",
                        lambda to, name, url: sent.append(to))
    email = "paidbio@example.com"

    for _ in range(3):
        r = c.post("/biofield/request", json={"email": email})
        assert r.status_code == 200 and r.get_json()["ok"] is True
    assert len(sent) == 3


# ── trial members: active grant but NOT paid -> free-tier quota, not unlimited ──
# The gate must match the plan's global "member = _is_paid_member" rule: an
# unconverted $1-trial buyer has an active membership GRANT (so
# _active_membership_for_email is truthy) but membership_category=='trial',
# so _is_paid_member(email) is False and they must be quota-gated like any
# other free-tier user. Only a FULLY-paid member (_is_paid_member True) stays
# unlimited.

def test_portal_request_trial_member_gets_free_tier_quota(client, monkeypatch):
    c, appmod = client
    monkeypatch.setattr(appmod, "ANALYSIS_QUOTA_ENABLED", True, raising=False)
    monkeypatch.setattr(appmod, "_active_membership_for_email",
                        lambda e: {"email": e, "source": "biofield_trial"})
    monkeypatch.setattr(appmod, "membership_category", lambda e: "trial")
    email = "trialportal@example.com"
    tok = _seed_portal(appmod, email)

    r1 = c.post(f"/api/portal/{tok}/biofield/request")
    assert r1.status_code == 200 and r1.get_json()["ok"] is True
    assert _status(appmod, email) == "requested"

    from dashboard import client_portal as cp
    cx = sqlite3.connect(appmod.LOG_DB)
    cp.set_biofield_status(cx, email, "confirmed")
    cx.close()

    r2 = c.post(f"/api/portal/{tok}/biofield/request")
    assert r2.status_code == 200
    j2 = r2.get_json()
    assert j2["ok"] is False and j2["reason"] == "monthly_quota"
    assert _status(appmod, email) == "confirmed"  # unchanged by the blocked request


def test_portal_request_full_paid_member_unlimited(client, monkeypatch):
    c, appmod = client
    monkeypatch.setattr(appmod, "ANALYSIS_QUOTA_ENABLED", True, raising=False)
    monkeypatch.setattr(appmod, "_active_membership_for_email", lambda e: {"email": e})
    monkeypatch.setattr(appmod, "membership_category", lambda e: "full")
    email = "fullpaidportal@example.com"
    tok = _seed_portal(appmod, email)

    for _ in range(3):
        from dashboard import client_portal as cp
        cx = sqlite3.connect(appmod.LOG_DB)
        cp.set_biofield_status(cx, email, "confirmed")
        cx.close()
        r = c.post(f"/api/portal/{tok}/biofield/request")
        assert r.status_code == 200 and r.get_json()["ok"] is True
        assert _status(appmod, email) == "requested"


def test_biofield_request_trial_member_gets_free_tier_quota(client, monkeypatch):
    c, appmod = client
    monkeypatch.setattr(appmod, "ANALYSIS_QUOTA_ENABLED", True, raising=False)
    monkeypatch.setattr(appmod, "_active_membership_for_email",
                        lambda e: {"email": e, "source": "biofield_trial"})
    monkeypatch.setattr(appmod, "membership_category", lambda e: "trial")
    sent = []
    monkeypatch.setattr(appmod, "send_magic_link_email",
                        lambda to, name, url: sent.append(to))
    email = "trialbio@example.com"

    r1 = c.post("/biofield/request", json={"email": email})
    assert r1.status_code == 200 and r1.get_json()["ok"] is True
    assert len(sent) == 1

    r2 = c.post("/biofield/request", json={"email": email})
    assert r2.status_code == 200
    j2 = r2.get_json()
    assert j2["ok"] is False and j2["reason"] == "monthly_quota"
    assert len(sent) == 1  # no second magic link sent to a trial member


def test_biofield_request_full_paid_member_unlimited(client, monkeypatch):
    c, appmod = client
    monkeypatch.setattr(appmod, "ANALYSIS_QUOTA_ENABLED", True, raising=False)
    monkeypatch.setattr(appmod, "_active_membership_for_email", lambda e: {"email": e})
    monkeypatch.setattr(appmod, "membership_category", lambda e: "full")
    sent = []
    monkeypatch.setattr(appmod, "send_magic_link_email",
                        lambda to, name, url: sent.append(to))
    email = "fullpaidbio@example.com"

    for _ in range(3):
        r = c.post("/biofield/request", json={"email": email})
        assert r.status_code == 200 and r.get_json()["ok"] is True
    assert len(sent) == 3
