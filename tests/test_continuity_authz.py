import sqlite3

from dashboard import continuity_view as cv, subscriptions as subs


def _cx():
    """Real init entrypoint (mirrors tests/test_subscriptions_consent.py): create the
    table then run the full migration chain so both attributed_practitioner_id and
    practitioner_share_consent columns exist. subs.init_tables() does NOT exist —
    the brief's placeholder was wrong."""
    cx = sqlite3.connect(":memory:")
    cx.row_factory = sqlite3.Row
    subs.init_subscriptions_table(cx)
    subs.migrate_add_membership_columns(cx)
    subs.migrate_add_term_cap_column(cx)
    subs.migrate_add_attribution_column(cx)
    subs.migrate_add_consent_column(cx)
    return cx


def _mk(cx, email, pid, consent):
    sid = subs.create_membership(
        cx,
        email=email,
        stripe_customer_id="c",
        stripe_payment_method_id="p",
        amount_cents=9900,
        next_charge_date="2026-08-01",
        attributed_practitioner_id=pid,
    )
    if consent:
        cx.execute(
            "UPDATE subscriptions SET practitioner_share_consent=1 WHERE id=?", (sid,)
        )
        cx.commit()
    return sid


def test_authorized_for_consented_continuity_patient():
    cx = _cx()
    _mk(cx, "pat@x.com", "prac-42", consent=True)
    assert cv.authorized_patient(cx, "prac-42", "pat@x.com") is True


def test_denied_other_doctor():
    cx = _cx()
    _mk(cx, "pat@x.com", "prac-42", consent=True)
    assert cv.authorized_patient(cx, "prac-99", "pat@x.com") is False


def test_denied_without_consent():
    cx = _cx()
    _mk(cx, "pat@x.com", "prac-42", consent=False)
    assert cv.authorized_patient(cx, "prac-42", "pat@x.com") is False


def test_denied_unknown_patient():
    cx = _cx()
    assert cv.authorized_patient(cx, "prac-42", "nobody@x.com") is False


def test_denied_falsy_practitioner_id():
    cx = _cx()
    _mk(cx, "pat@x.com", "prac-42", consent=True)
    assert cv.authorized_patient(cx, "", "pat@x.com") is False
    assert cv.authorized_patient(cx, None, "pat@x.com") is False


def test_denied_falsy_patient_email():
    cx = _cx()
    _mk(cx, "pat@x.com", "prac-42", consent=True)
    assert cv.authorized_patient(cx, "prac-42", "") is False
    assert cv.authorized_patient(cx, "prac-42", None) is False
