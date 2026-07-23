import sqlite3
from dashboard import household as hh

def _cx():
    cx = sqlite3.connect(":memory:"); cx.row_factory = sqlite3.Row
    hh.init_household_tables(cx)
    return cx

def test_pay_consent_default_off_and_grant_flow():
    cx = _cx()
    hh.add_member(cx, "steve@x.com", "michael@x.com", relationship="partner")
    # default: no pay consent
    assert hh.can_pay(cx, "steve@x.com", "michael@x.com") is False
    assert hh.payable_members_for(cx, "steve@x.com") == []
    # member grants pay consent with line-item visibility
    hh.set_pay_consent(cx, "steve@x.com", "michael@x.com", 1, share_scope="line_items")
    assert hh.can_pay(cx, "steve@x.com", "michael@x.com") is True
    pm = hh.payable_members_for(cx, "steve@x.com")
    assert pm == [{"member_email": "michael@x.com", "label": "", "pay_share_scope": "line_items"}]
    # revoke is non-destructive to the link, just flips the flag
    hh.set_pay_consent(cx, "steve@x.com", "michael@x.com", 0)
    assert hh.can_pay(cx, "steve@x.com", "michael@x.com") is False

def test_pay_consent_self_pay_guard():
    cx = _cx()
    hh.add_member(cx, "steve@x.com", "steve@x.com", relationship="")
    hh.set_pay_consent(cx, "steve@x.com", "steve@x.com", 1)
    assert hh.can_pay(cx, "steve@x.com", "steve@x.com") is False
