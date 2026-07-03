# tests/test_inhouse_repertoire.py
"""Task 5b: the in-house pricing path (_portal_priced_lines / _price_inhouse_invoice
-> _inhouse_line_unit_cents) — the ACTUAL checkout/invoice charge path — must honor
repertoire pricing for paid members, and must agree TO THE CENT with the display
path (_price_cart / dashboard.pricing.compute(), Task 3/5). Display MUST equal
charge. Mirrors the harness in tests/test_portal_reorder_module.py."""
import sqlite3
from datetime import datetime, timedelta

import pytest


@pytest.fixture
def appmod(monkeypatch, tmp_path):
    import app as appmod
    monkeypatch.setattr(appmod, "LOG_DB", str(tmp_path / "chat_log.db"))
    appmod._init_auth_tables()
    monkeypatch.setattr(appmod, "REPERTOIRE_ENABLED", True)
    appmod.app.config["TESTING"] = True
    return appmod


def _seed_active_membership(appmod, email, *, source="founding"):
    expires = (datetime.utcnow() + timedelta(days=30)).isoformat() + "Z"
    with sqlite3.connect(appmod.LOG_DB) as cx:
        appmod.init_membership_tables(cx)
        cx.execute(
            "INSERT INTO memberships (id, email, granted_at, expires_at, granted_by, source) "
            "VALUES (?,?,?,?,?,?)",
            (f"mem_{email}", email, datetime.utcnow().isoformat() + "Z", expires,
             "test", source))
        cx.commit()


def _add_repertoire(appmod, email, slugs):
    with sqlite3.connect(appmod.LOG_DB) as cx:
        appmod.repertoire.init_repertoire_table(cx)
        appmod.repertoire.add_skus(cx, email, slugs)


SLUG = "neuro-magnesium"


# ── display === charge: _portal_priced_lines agrees with _price_cart/compute ──

def test_portal_priced_lines_matches_price_cart_for_member_repertoire_sku(appmod):
    email = "member@example.com"
    _seed_active_membership(appmod, email)
    _add_repertoire(appmod, email, [SLUG])

    lines, items_rec, subtotal = appmod._portal_priced_lines(
        [{"slug": SLUG, "qty": 1}], email=email)
    charge_unit_cents = items_rec[0]["unit_cents"]

    display_unit_cents = appmod._price_cart(
        [{"slug": SLUG, "qty": 1}],
        ship={"country": "US", "state": "TX"}, email=email,
    )["priced"]["lines"][0]["line_total_cents"]

    assert charge_unit_cents == display_unit_cents
    regular_cents = appmod._get_product(SLUG)["price_cents"]
    assert charge_unit_cents < regular_cents


def test_portal_priced_lines_non_member_pays_regular(appmod):
    email = "nonmember@example.com"
    lines, items_rec, subtotal = appmod._portal_priced_lines(
        [{"slug": SLUG, "qty": 1}], email=email)
    regular_cents = appmod._get_product(SLUG)["price_cents"]
    assert items_rec[0]["unit_cents"] == regular_cents


def test_portal_priced_lines_flag_off_pays_regular(appmod, monkeypatch):
    email = "member@example.com"
    _seed_active_membership(appmod, email)
    _add_repertoire(appmod, email, [SLUG])
    monkeypatch.setattr(appmod, "REPERTOIRE_ENABLED", False)

    lines, items_rec, subtotal = appmod._portal_priced_lines(
        [{"slug": SLUG, "qty": 1}], email=email)
    regular_cents = appmod._get_product(SLUG)["price_cents"]
    assert items_rec[0]["unit_cents"] == regular_cents


def test_portal_priced_lines_member_non_repertoire_sku_pays_regular(appmod):
    """A member's first buy of a SKU not yet in their repertoire prices at the
    regular/volume rate (no repertoire discount) — repertoire is reorder-only."""
    email = "member@example.com"
    _seed_active_membership(appmod, email)
    _add_repertoire(appmod, email, ["some-other-sku"])

    lines, items_rec, subtotal = appmod._portal_priced_lines(
        [{"slug": SLUG, "qty": 1}], email=email)
    regular_cents = appmod._get_product(SLUG)["price_cents"]
    assert items_rec[0]["unit_cents"] == regular_cents


def test_portal_priced_lines_explicit_override_wins_over_repertoire(appmod):
    """A practitioner-special price_cents override on the portal item still wins
    over repertoire — explicit owner/practitioner intent is never overridden."""
    email = "member@example.com"
    _seed_active_membership(appmod, email)
    _add_repertoire(appmod, email, [SLUG])

    lines, items_rec, subtotal = appmod._portal_priced_lines(
        [{"slug": SLUG, "qty": 1, "price_cents": 1234}], email=email)
    assert items_rec[0]["unit_cents"] == 1234


# ── same guarantee on the owner in-house invoice path ────────────────────────

def test_price_inhouse_invoice_honors_repertoire_for_member(appmod):
    email = "member@example.com"
    _seed_active_membership(appmod, email)
    _add_repertoire(appmod, email, [SLUG])

    priced = appmod._price_inhouse_invoice(
        [{"slug": SLUG, "qty": 1}], email=email, pickup=True,
        ship={"country": "US", "state": "TX"})
    assert priced is not None
    unit_cents = priced["items_rec"][0]["unit_cents"]
    regular_cents = appmod._get_product(SLUG)["price_cents"]
    assert unit_cents < regular_cents

    display_unit_cents = appmod._price_cart(
        [{"slug": SLUG, "qty": 1}],
        ship={"country": "US", "state": "TX"}, email=email,
    )["priced"]["lines"][0]["line_total_cents"]
    assert unit_cents == display_unit_cents


def test_price_inhouse_invoice_non_member_pays_regular(appmod):
    email = "nonmember@example.com"
    priced = appmod._price_inhouse_invoice(
        [{"slug": SLUG, "qty": 1}], email=email, pickup=True,
        ship={"country": "US", "state": "TX"})
    assert priced is not None
    regular_cents = appmod._get_product(SLUG)["price_cents"]
    assert priced["items_rec"][0]["unit_cents"] == regular_cents


def test_price_inhouse_invoice_fresh_db_no_repertoire_table_does_not_crash(appmod):
    """Fresh-DB guard: repertoire table not yet created must not break invoicing."""
    email = "member@example.com"
    _seed_active_membership(appmod, email)
    # deliberately do NOT init the repertoire table

    priced = appmod._price_inhouse_invoice(
        [{"slug": SLUG, "qty": 1}], email=email, pickup=True,
        ship={"country": "US", "state": "TX"})
    assert priced is not None
    regular_cents = appmod._get_product(SLUG)["price_cents"]
    assert priced["items_rec"][0]["unit_cents"] == regular_cents
