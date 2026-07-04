import sqlite3

from dashboard import continuity_view as cv, subscriptions as subs


def _cx():
    """Real init entrypoint (mirrors tests/test_continuity_authz.py): create the
    table then run the full migration chain so both attributed_practitioner_id and
    practitioner_share_consent columns exist."""
    cx = sqlite3.connect(":memory:")
    cx.row_factory = sqlite3.Row
    subs.init_subscriptions_table(cx)
    subs.migrate_add_membership_columns(cx)
    subs.migrate_add_term_cap_column(cx)
    subs.migrate_add_attribution_column(cx)
    subs.migrate_add_consent_column(cx)
    return cx


def _mk(cx, email, pid, consent, status="active"):
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
    if status != "active":
        cx.execute("UPDATE subscriptions SET status=? WHERE id=?", (status, sid))
    cx.commit()
    return sid


def test_roster_matches_the_gate_predicate_exactly():
    """Two consented continuity members for prac-42, one for a different doctor,
    one unconsented for prac-42, and one CANCELLED consented for prac-42 — the
    roster must return EXACTLY the two active/consented prac-42 patients. This
    proves roster() and authorized_patient() can never disagree."""
    cx = _cx()
    _mk(cx, "a@x.com", "prac-42", consent=True)
    _mk(cx, "b@x.com", "prac-42", consent=True)
    _mk(cx, "other-doctor@x.com", "prac-99", consent=True)
    _mk(cx, "unconsented@x.com", "prac-42", consent=False)
    _mk(cx, "cancelled@x.com", "prac-42", consent=True, status="cancelled")

    result = cv.roster(cx, "prac-42")
    emails = {r["email"] for r in result}

    assert emails == {"a@x.com", "b@x.com"}
    assert len(result) == 2

    # cross-check against the gate itself: every roster patient passes it, and
    # neither excluded patient does.
    for r in result:
        assert cv.authorized_patient(cx, "prac-42", r["email"]) is True
    assert cv.authorized_patient(cx, "prac-42", "other-doctor@x.com") is False
    assert cv.authorized_patient(cx, "prac-42", "unconsented@x.com") is False
    assert cv.authorized_patient(cx, "prac-42", "cancelled@x.com") is False


def test_roster_allows_paused_membership():
    cx = _cx()
    _mk(cx, "a@x.com", "prac-42", consent=True, status="paused")
    result = cv.roster(cx, "prac-42")
    assert {r["email"] for r in result} == {"a@x.com"}


def test_roster_empty_for_falsy_practitioner_id():
    cx = _cx()
    _mk(cx, "a@x.com", "prac-42", consent=True)
    assert cv.roster(cx, "") == []
    assert cv.roster(cx, None) == []


def test_roster_empty_when_no_patients():
    cx = _cx()
    assert cv.roster(cx, "prac-42") == []


def test_roster_dedupes_same_email_case_insensitively():
    cx = _cx()
    _mk(cx, "dup@x.com", "prac-42", consent=True)
    _mk(cx, "DUP@x.com", "prac-42", consent=True)
    result = cv.roster(cx, "prac-42")
    assert len(result) == 1
    assert result[0]["email"] == "dup@x.com"


def test_roster_falls_back_to_email_local_part_when_no_name_available():
    cx = _cx()
    _mk(cx, "jane.doe@x.com", "prac-42", consent=True)
    result = cv.roster(cx, "prac-42")
    assert result[0]["name"] == "jane.doe"


def test_roster_uses_people_table_name_when_available():
    cx = _cx()
    _mk(cx, "jane@x.com", "prac-42", consent=True)
    cx.execute("CREATE TABLE people (id INTEGER PRIMARY KEY, email TEXT, name TEXT)")
    cx.execute("INSERT INTO people (email, name) VALUES ('jane@x.com', 'Jane Doe')")
    cx.commit()
    result = cv.roster(cx, "prac-42")
    assert result[0]["name"] == "Jane Doe"
