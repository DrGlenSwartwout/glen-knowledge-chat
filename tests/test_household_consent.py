import sqlite3
from dashboard import household as h


def _cx():
    cx = sqlite3.connect(":memory:")
    h.init_household_tables(cx)
    return cx


def test_dependent_defaults_shared_and_authority():
    cx = _cx()
    h.add_member(cx, "cg@x.com", "kid@x.com", "Kid", "dependent")
    st = h.consent_state(cx, "cg@x.com", "kid@x.com")
    assert st["share_consent"] == 1 and st["consent_basis"] == "caregiver-authority"


def test_blank_relationship_keeps_legacy_shared_default():
    cx = _cx()
    h.add_member(cx, "cg@x.com", "m@x.com", "M")   # no relationship
    st = h.consent_state(cx, "cg@x.com", "m@x.com")
    assert st["share_consent"] == 1 and st["consent_basis"] == ""


def test_operational_defaults_dark_and_pending():
    cx = _cx()
    h.add_member(cx, "cg@x.com", "adult@x.com", "Partner", "partner")
    st = h.consent_state(cx, "cg@x.com", "adult@x.com")
    assert st["share_consent"] == 0 and st["consent_basis"] == ""


def test_operational_with_verbal_basis_is_active_but_unconfirmed():
    cx = _cx()
    h.add_member(cx, "cg@x.com", "adult@x.com", "Partner", "partner",
                 consent_basis="verbal", consent_by="rae")
    st = h.consent_state(cx, "cg@x.com", "adult@x.com")
    assert st["share_consent"] == 1 and st["consent_basis"] == "verbal"
    assert st["consent_confirmed_at"] in (None, "")


def test_confirm_consent_hard_and_idempotent():
    cx = _cx()
    h.add_member(cx, "cg@x.com", "adult@x.com", "Partner", "partner")
    assert h.confirm_consent(cx, "cg@x.com", "adult@x.com") is True
    st = h.consent_state(cx, "cg@x.com", "adult@x.com")
    assert st["share_consent"] == 1 and st["consent_basis"] == "portal-confirmed"
    assert st["consent_confirmed_at"]
    first = st["consent_confirmed_at"]
    h.confirm_consent(cx, "cg@x.com", "adult@x.com")  # idempotent, no downgrade
    st2 = h.consent_state(cx, "cg@x.com", "adult@x.com")
    assert st2["consent_confirmed_at"] == first and st2["share_consent"] == 1
