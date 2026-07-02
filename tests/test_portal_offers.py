# tests/test_portal_offers.py
"""The upgrade-ladder resolver: given a person, return the eligible rungs
(flag-on AND not owned) in ladder order. Pure + cx-based, mirrors portal_view."""
import sqlite3

import pytest


def _conn(tmp_path):
    from dashboard import subscriptions as subs
    from dashboard import biofield_store as bf
    cx = sqlite3.connect(str(tmp_path / "t.db"))
    subs.init_subscriptions_table(cx)
    subs.migrate_add_membership_columns(cx)
    subs.migrate_add_term_cap_column(cx)
    bf.init_table(cx)
    return cx


ALL = {"live_group", "biofield"}


def test_new_client_gets_live_group_first(tmp_path):
    from dashboard import portal_offers as po
    cx = _conn(tmp_path)
    offers = po.next_offers(cx, "new@x.com", ["client"], enabled_keys=ALL)
    assert [o["key"] for o in offers] == ["live_group", "biofield"]
    assert offers[0]["price_cents"] == 9900
    assert offers[0]["checkout_path"] == "/portal/offer/live-group/checkout"


def test_group_member_skips_to_biofield(tmp_path):
    from dashboard import portal_offers as po
    from dashboard import subscriptions as subs
    cx = _conn(tmp_path)
    subs.create_membership(cx, email="m@x.com", stripe_customer_id="c",
                           stripe_payment_method_id="pm", amount_cents=9900,
                           next_charge_date="2026-07-16")
    offers = po.next_offers(cx, "m@x.com", ["client"], enabled_keys=ALL)
    assert [o["key"] for o in offers] == ["biofield"]


def test_owns_both_returns_empty(tmp_path):
    from dashboard import portal_offers as po
    from dashboard import subscriptions as subs
    from dashboard import biofield_store as bf
    cx = _conn(tmp_path)
    subs.create_membership(cx, email="b@x.com", stripe_customer_id="c",
                           stripe_payment_method_id="pm", amount_cents=9900,
                           next_charge_date="2026-07-16")
    bf.seed_paid(cx, "b@x.com", via="checkout", order_ref="o1")
    assert po.next_offers(cx, "b@x.com", ["client"], enabled_keys=ALL) == []


def test_flag_off_rung_is_excluded(tmp_path):
    from dashboard import portal_offers as po
    cx = _conn(tmp_path)
    # only biofield flag on -> live_group hidden even though unowned
    offers = po.next_offers(cx, "new@x.com", ["client"], enabled_keys={"biofield"})
    assert [o["key"] for o in offers] == ["biofield"]
