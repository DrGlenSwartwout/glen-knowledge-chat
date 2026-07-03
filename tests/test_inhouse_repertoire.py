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

# A non-FF SKU: broadly eligible under the OLD rule (not info_only, not a true
# pure powder) but _qty_eligible()==False (no qty_pricing flag — not a Functional
# Formulation). Glen 2026-07 (margin): the member repertoire reorder discount
# is now restricted to FF products ONLY — the repertoire slug set is FF-filtered
# (_ff_filter_slugs) before it ever reaches the pricing engine, at every
# pricing site (_price_cart, _resolve_repertoire_slugs, _portal_reorder_module's
# your_cents). So a non-FF repertoire SKU now prices at REGULAR in both display
# and charge — see test_..._non_ff_repertoire_sku_pays_regular below (this
# used to assert the opposite: that it GOT the discount; that expectation was
# the broad-eligibility bug this file originally caught in Task 5b, now
# superseded by the FF-only restriction).
NON_FF_SLUG = "ei8-microbes-liver-integrator"


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


def test_portal_priced_lines_matches_price_cart_for_non_ff_repertoire_sku(appmod):
    """FF-only restriction (Glen 2026-07, margin): a non-FF SKU
    (_qty_eligible()==False) that WAS broadly eligible under the old rule must NOT get
    the repertoire discount — at charge OR display. The repertoire slug set is
    FF-filtered (_ff_filter_slugs) before it reaches either pricing path, so a
    member's non-FF repertoire SKU prices at REGULAR everywhere, and display
    still agrees with charge to the cent (both regular)."""
    p = appmod._get_product(NON_FF_SLUG)
    # non-FF, yet broadly eligible under the pre-FF-only rule (not info_only,
    # not a true pure powder) — i.e. it WOULD have been discounted before:
    assert not p.get("info_only") and not appmod._is_pure_powder(p)
    assert appmod._qty_eligible(p) is False

    email = "member@example.com"
    _seed_active_membership(appmod, email)
    _add_repertoire(appmod, email, [NON_FF_SLUG])

    lines, items_rec, subtotal = appmod._portal_priced_lines(
        [{"slug": NON_FF_SLUG, "qty": 1}], email=email)
    charge_unit_cents = items_rec[0]["unit_cents"]

    display_unit_cents = appmod._price_cart(
        [{"slug": NON_FF_SLUG, "qty": 1}],
        ship={"country": "US", "state": "TX"}, email=email,
    )["priced"]["lines"][0]["line_total_cents"]

    assert charge_unit_cents == display_unit_cents
    regular_cents = p["price_cents"]
    assert charge_unit_cents == regular_cents


def test_mixed_cart_ff_discounted_non_ff_regular_display_and_charge(appmod):
    """A member with BOTH an FF repertoire SKU and a non-FF repertoire SKU in
    their cart: the FF one is discounted, the non-FF one is regular — in BOTH
    the display path (_price_cart/compute) and the charge path
    (_portal_priced_lines/_inhouse_line_unit_cents), agreeing to the cent."""
    email = "member@example.com"
    _seed_active_membership(appmod, email)
    _add_repertoire(appmod, email, [SLUG, NON_FF_SLUG])

    cart = [{"slug": SLUG, "qty": 1}, {"slug": NON_FF_SLUG, "qty": 1}]
    lines, items_rec, subtotal = appmod._portal_priced_lines(cart, email=email)
    charge_by_slug = {r["slug"]: r["unit_cents"] for r in items_rec}

    display = appmod._price_cart(
        cart, ship={"country": "US", "state": "TX"}, email=email,
    )["priced"]["lines"]
    display_by_slug = {l["slug"]: l["line_total_cents"] for l in display}

    ff_regular = appmod._get_product(SLUG)["price_cents"]
    non_ff_regular = appmod._get_product(NON_FF_SLUG)["price_cents"]

    # FF SKU: discounted, display == charge.
    assert charge_by_slug[SLUG] < ff_regular
    assert charge_by_slug[SLUG] == display_by_slug[SLUG]
    # non-FF SKU: regular, display == charge.
    assert charge_by_slug[NON_FF_SLUG] == non_ff_regular
    assert display_by_slug[NON_FF_SLUG] == non_ff_regular
    assert charge_by_slug[NON_FF_SLUG] == display_by_slug[NON_FF_SLUG]


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


# ── client all-FF flat rate persists through Edit Invoice ─────────────────────
# Bug (Glen 2026-07-03): the console editor marked EVERY loaded line as a manual
# override, so a client's all-FFs flat rate never re-applied on Edit Invoice. The
# fix stamps override=True ONLY on owner-typed lines; auto/flat lines re-price on
# edit. These lock the server half of that contract.

def _set_ff_flat(appmod, email, cents):
    from dashboard import client_prices as cp
    with sqlite3.connect(appmod.LOG_DB) as cx:
        cp.init_table(cx)
        cp.set_ff_flat(cx, email, cents)


FF_FLAT_CENTS = 5000  # $50 flat for all this client's FFs


def test_inhouse_ff_flat_applies_and_is_not_flagged_override(appmod):
    email = "flatclient@example.com"
    _set_ff_flat(appmod, email, FF_FLAT_CENTS)
    priced = appmod._price_inhouse_invoice(
        [{"slug": SLUG, "qty": 1}], email=email, pickup=True,
        ship={"country": "US", "state": "TX"})
    rec = priced["items_rec"][0]
    assert rec["unit_cents"] == FF_FLAT_CENTS
    # not an owner override -> Edit Invoice will re-price it (flat re-applies)
    assert "override" not in rec


def test_inhouse_explicit_override_is_flagged(appmod):
    email = "flatclient@example.com"
    _set_ff_flat(appmod, email, FF_FLAT_CENTS)
    priced = appmod._price_inhouse_invoice(
        [{"slug": SLUG, "qty": 1, "unit_cents": 1234}], email=email, pickup=True,
        ship={"country": "US", "state": "TX"})
    rec = priced["items_rec"][0]
    assert rec["unit_cents"] == 1234
    assert rec["override"] is True   # frozen on Edit Invoice


def test_edit_reprices_unflagged_ff_line_to_current_flat(appmod):
    """Re-pricing a stored FF line with NO override flag (how the fixed editor
    submits auto-priced lines) picks up the client's CURRENT flat rate, even after
    it changes — the persistence the Edit Invoice bug broke."""
    email = "flatclient@example.com"
    _set_ff_flat(appmod, email, FF_FLAT_CENTS)
    p1 = appmod._price_inhouse_invoice(
        [{"slug": SLUG, "qty": 1}], email=email, pickup=True,
        ship={"country": "US", "state": "TX"})
    assert p1["items_rec"][0]["unit_cents"] == FF_FLAT_CENTS
    _set_ff_flat(appmod, email, 3000)   # owner lowers the flat, then edits
    p2 = appmod._price_inhouse_invoice(
        [{"slug": SLUG, "qty": 1}], email=email, pickup=True,
        ship={"country": "US", "state": "TX"})
    assert p2["items_rec"][0]["unit_cents"] == 3000
