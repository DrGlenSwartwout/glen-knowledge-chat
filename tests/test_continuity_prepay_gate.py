import sqlite3
from dashboard import continuity_view as cv


def _cx():
    cx = sqlite3.connect(":memory:"); cx.row_factory = sqlite3.Row
    cx.execute("CREATE TABLE subscriptions (email TEXT, attributed_practitioner_id TEXT, practitioner_share_consent INT, kind TEXT, status TEXT)")
    cx.execute("CREATE TABLE prepay_term_grants (session_id TEXT PRIMARY KEY, email TEXT, tier_key TEXT, granted_at TEXT, attributed_practitioner_id TEXT, practitioner_share_consent INT DEFAULT 0, term_end TEXT)")
    return cx


def _grant(cx, email, pid, consent, term_end):
    cx.execute("INSERT INTO prepay_term_grants (session_id,email,tier_key,attributed_practitioner_id,practitioner_share_consent,term_end) VALUES (?,?,?,?,?,?)",
               ("s"+email, email, "12mo", pid, consent, term_end)); cx.commit()


def _sub(cx, email, pid, consent):
    cx.execute("INSERT INTO subscriptions (email,attributed_practitioner_id,practitioner_share_consent,kind,status) VALUES (?,?,?,?,?)",
               (email, pid, consent, "membership", "active")); cx.commit()


def test_prepay_in_term_consented_is_authorized():
    cx = _cx(); _grant(cx, "pat@x.com", "prac-42", 1, "2999-01-01")
    assert cv.authorized_patient(cx, "prac-42", "pat@x.com") is True
    assert "pat@x.com" in [r["email"] for r in cv.roster(cx, "prac-42")]


def test_prepay_expired_denied():
    cx = _cx(); _grant(cx, "pat@x.com", "prac-42", 1, "2000-01-01")
    assert cv.authorized_patient(cx, "prac-42", "pat@x.com") is False
    assert "pat@x.com" not in [r["email"] for r in cv.roster(cx, "prac-42")]


def test_prepay_unconsented_denied():
    cx = _cx(); _grant(cx, "pat@x.com", "prac-42", 0, "2999-01-01")
    assert cv.authorized_patient(cx, "prac-42", "pat@x.com") is False
    assert "pat@x.com" not in [r["email"] for r in cv.roster(cx, "prac-42")]


def test_prepay_other_doctor_denied():
    cx = _cx(); _grant(cx, "pat@x.com", "prac-42", 1, "2999-01-01")
    assert cv.authorized_patient(cx, "prac-99", "pat@x.com") is False
    assert "pat@x.com" not in [r["email"] for r in cv.roster(cx, "prac-99")]


def test_subscriptions_only_still_authorized_no_regression():
    """Existing subscriptions-only path keeps working when a prepay_term_grants
    table is present but has no matching row (no regression from the OR)."""
    cx = _cx(); _sub(cx, "sub@x.com", "prac-42", 1)
    assert cv.authorized_patient(cx, "prac-42", "sub@x.com") is True
    assert "sub@x.com" in [r["email"] for r in cv.roster(cx, "prac-42")]


def test_both_subscription_and_prepay_dedup_in_roster():
    """A patient with BOTH a consented subscription AND a consented in-term
    prepay grant for the same doctor appears exactly once in the roster."""
    cx = _cx()
    _sub(cx, "both@x.com", "prac-42", 1)
    _grant(cx, "both@x.com", "prac-42", 1, "2999-01-01")
    result = cv.roster(cx, "prac-42")
    emails = [r["email"] for r in result]
    assert emails.count("both@x.com") == 1
