"""Family plan entitlement: one caregiver's paid plan un-blurs the whole household.

The plan is bought by the caregiver. Every household member linked to that
caregiver WITH share_consent=1 gets full scan results. A member who has revoked
consent is not covered — consent gates the caregiver's reach in both directions
(same rule household.can_view already enforces for the view switcher).
"""

import sqlite3

import pytest

from dashboard import family_plan as fp
from dashboard import household as hh


CAREGIVER = "caregiver@example.com"
PET = "pet@example.com"
SPOUSE = "spouse@example.com"
STRANGER = "someone-else@example.com"


@pytest.fixture()
def cx():
    con = sqlite3.connect(":memory:")
    con.row_factory = sqlite3.Row
    hh.init_household_tables(con)
    fp.init_family_plan_table(con)
    yield con
    con.close()


def _link(cx, member, relationship="pet"):
    hh.add_member(cx, CAREGIVER, member, relationship=relationship)


def test_plan_price_is_147_with_a_197_value_anchor():
    assert fp.PLAN["amount_cents"] == 14700
    assert fp.PLAN["value_cents"] == 19700


def test_activate_with_special_amount_sets_that_price(cx):
    # A special per-household price (mirrors Biofield-test special pricing).
    fp.activate(cx, CAREGIVER, next_charge_at="2026-08-09", amount_cents=9900)
    assert fp.get(cx, CAREGIVER)["amount_cents"] == 9900


def test_activate_without_amount_uses_standard_plan_price(cx):
    fp.activate(cx, CAREGIVER, next_charge_at="2026-08-09")
    assert fp.get(cx, CAREGIVER)["amount_cents"] == fp.PLAN["amount_cents"]


def test_special_amount_is_what_falls_due_for_the_cron(cx):
    # The charge cron bills sub["amount_cents"], so the special price is what recurs.
    fp.activate(cx, CAREGIVER, next_charge_at="2026-01-01", amount_cents=9900)
    due = fp.due(cx, "2026-06-01")
    assert due and due[0]["amount_cents"] == 9900


def test_comped_household_can_carry_a_special_price_but_never_bills(cx):
    fp.activate(cx, CAREGIVER, next_charge_at=None, source="comp", amount_cents=9900)
    assert fp.get(cx, CAREGIVER)["amount_cents"] == 9900
    assert fp.due(cx, "2026-12-01") == []


def test_negative_special_amount_is_rejected(cx):
    with pytest.raises(ValueError):
        fp.activate(cx, CAREGIVER, next_charge_at="2026-08-09", amount_cents=-1)


def test_caregiver_without_a_plan_covers_nobody(cx):
    _link(cx, PET)
    assert fp.is_active(cx, CAREGIVER) is False
    assert fp.covers(cx, PET) is False


def test_active_plan_covers_a_consented_member(cx):
    _link(cx, PET)
    fp.activate(cx, CAREGIVER, next_charge_at="2026-08-09")
    assert fp.covers(cx, PET) is True


def test_active_plan_covers_the_caregiver_themself(cx):
    fp.activate(cx, CAREGIVER, next_charge_at="2026-08-09")
    assert fp.covers(cx, CAREGIVER) is True


def test_member_who_revoked_consent_is_not_covered(cx):
    _link(cx, PET)
    fp.activate(cx, CAREGIVER, next_charge_at="2026-08-09")
    hh.set_share_consent(cx, CAREGIVER, PET, 0)
    assert fp.covers(cx, PET) is False


def test_cancelled_plan_stops_covering_members(cx):
    _link(cx, PET)
    fp.activate(cx, CAREGIVER, next_charge_at="2026-08-09")
    fp.set_status(cx, CAREGIVER, "cancelled")
    assert fp.covers(cx, PET) is False


def test_past_due_plan_still_covers_during_grace(cx):
    """A failed renewal must not instantly blur a client's report mid-month."""
    _link(cx, PET)
    fp.activate(cx, CAREGIVER, next_charge_at="2026-08-09")
    fp.set_status(cx, CAREGIVER, "past_due")
    assert fp.covers(cx, PET) is True


def test_plan_does_not_leak_to_an_unlinked_stranger(cx):
    _link(cx, PET)
    fp.activate(cx, CAREGIVER, next_charge_at="2026-08-09")
    assert fp.covers(cx, STRANGER) is False


def test_covers_is_case_and_whitespace_insensitive(cx):
    _link(cx, PET)
    fp.activate(cx, CAREGIVER, next_charge_at="2026-08-09")
    assert fp.covers(cx, "  " + PET.upper() + " ") is True


def test_every_consented_member_is_covered_by_one_plan(cx):
    _link(cx, PET, relationship="pet")
    _link(cx, SPOUSE, relationship="spouse")
    fp.activate(cx, CAREGIVER, next_charge_at="2026-08-09")
    assert fp.covers(cx, PET) is True
    assert fp.covers(cx, SPOUSE) is True


def test_comped_plan_covers_without_a_stripe_customer(cx):
    """Glen comps Karin: no card on file, same entitlement."""
    _link(cx, PET)
    fp.activate(cx, CAREGIVER, next_charge_at=None, source="comp")
    assert fp.is_active(cx, CAREGIVER) is True
    assert fp.covers(cx, PET) is True
