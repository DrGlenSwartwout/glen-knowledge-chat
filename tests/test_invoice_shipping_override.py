"""Manual shipping-fee override on the in-house invoice pricer.

Edit Invoice (and the order builder) can send `shipping_cents` to REPLACE the
geometry-computed shipping fee — for products whose box dims aren't in the catalog,
or to match a hand-quoted rate. This exercises the real `_price_inhouse_invoice`
branch with the rate engine (`_price_cart`) and catalog (`_get_product`) stubbed so
the test stays deterministic and secretless.
"""
import pytest

app_mod = pytest.importorskip("app")


@pytest.fixture
def stub_pricer(monkeypatch):
    # A single non-FF, shippable product priced by an explicit per-line override, so the
    # line math is trivial and the test isolates the shipping branch.
    monkeypatch.setattr(app_mod, "_get_product",
                        lambda slug: {"slug": "widget", "name": "Widget", "price_cents": 1000}
                        if slug == "widget" else None)
    # Geometry engine always quotes $7.00; GET absorbed at $0.
    monkeypatch.setattr(app_mod, "_price_cart",
                        lambda *a, **k: {"shipping_cents": 700, "priced": {"get_cents": 0}})
    # Keep membership/repertoire out of it.
    monkeypatch.setattr(app_mod, "_is_paid_member", lambda *a, **k: False)
    monkeypatch.setattr(app_mod, "_resolve_repertoire_slugs", lambda *a, **k: [])
    return app_mod


LINES = [{"slug": "widget", "qty": 1, "unit_cents": 1000}]
SHIP = {"name": "B", "street": "1 A St", "city": "Hilo", "state": "HI", "zip": "96720", "country": "US"}


def test_no_override_uses_computed_shipping(stub_pricer):
    p = stub_pricer._price_inhouse_invoice(LINES, email="", pickup=False, ship=SHIP)
    assert p["shipping_cents"] == 700
    assert p["total_cents"] == 1000 + 700


def test_override_replaces_computed_shipping(stub_pricer):
    p = stub_pricer._price_inhouse_invoice(
        LINES, email="", pickup=False, ship=SHIP, shipping_override_cents_in=1234)
    assert p["shipping_cents"] == 1234                 # replaces the $7.00 quote
    assert p["total_cents"] == 1000 + 1234


def test_override_zero_is_honored(stub_pricer):
    # $0.00 is a real choice (free shipping), not "blank" — it must stick.
    p = stub_pricer._price_inhouse_invoice(
        LINES, email="", pickup=False, ship=SHIP, shipping_override_cents_in=0)
    assert p["shipping_cents"] == 0
    assert p["total_cents"] == 1000


def test_blank_override_falls_back_to_computed(stub_pricer):
    for blank in (None, ""):
        p = stub_pricer._price_inhouse_invoice(
            LINES, email="", pickup=False, ship=SHIP, shipping_override_cents_in=blank)
        assert p["shipping_cents"] == 700, blank


def test_negative_override_clamped_to_zero(stub_pricer):
    p = stub_pricer._price_inhouse_invoice(
        LINES, email="", pickup=False, ship=SHIP, shipping_override_cents_in=-500)
    assert p["shipping_cents"] == 0


def test_pickup_forces_zero_over_override(stub_pricer):
    # Pickup means no shipment — an override must never resurrect a shipping charge.
    p = stub_pricer._price_inhouse_invoice(
        LINES, email="", pickup=True, ship=SHIP, shipping_override_cents_in=1234)
    assert p["shipping_cents"] == 0
    assert p["total_cents"] == 1000
