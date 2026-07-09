"""A non-physical line must not add to the box count.

Imports app (needs real secrets + writable DATA_DIR), so it's skipped under plain
pytest and runs under the Doppler harness:
  doppler run -p remedy-match -c prd -- env DATA_DIR=/tmp/scratch python3 -m pytest tests/test_price_cart_shippable.py
"""
import sys
from pathlib import Path

import pytest

repo_root = Path(__file__).resolve().parent.parent
if str(repo_root) not in sys.path:
    sys.path.insert(0, str(repo_root))

try:
    import app
except Exception as _e:  # pragma: no cover - exercised only under plain pytest
    pytest.skip(f"app import requires real secrets: {_e}", allow_module_level=True)

US = {"name": "T", "state": "HI", "country": "US"}


def test_a_service_line_does_not_change_shipping(monkeypatch):
    """Four bottles ship for the same price with or without a Biofield Analysis
    added. Before this fix the service counted as a fifth bottle and could push
    the order up a box size.

    The expected amount is derived straight from dashboard.shipping.quote() on a
    box_counts dict built ONLY from the bottle line (never hardcoded), so the
    assertion proves the mixed cart went through the real box-fit catalog path.
    Comparing the two invoice totals to each other, without this independent
    derivation, is not a durable regression guard: pre-fix, the service line broke
    quote() (unknown "default" bottle type) and _shipping_for_cart silently fell
    back to the qty-based fallback rate. A coincidence is not a guard — and here
    it runs two levels deep: with the *current* rates/capacity data, box-fit(4
    bottles) needs an M box, and the qty-fallback table also lands 5 items (4
    bottles + 1 mis-counted service) in the M band, so even the fallback cents
    equal the box-fit cents for this specific quantity. Comparing final cents
    alone therefore can't tell fixed from buggy for this data. So we also spy on
    dashboard.shipping.quote() and assert the box_counts dict it was actually
    called with for the mixed cart is byte-for-byte the bottle-only box_counts —
    proving the service line never reached the box-fit call at all, independent
    of whatever a rate table or capacity row happens to produce numerically.
    """
    bottle_slug, bottle_qty = "neuro-magnesium", 4
    product = app._get_product(bottle_slug)
    bottle_type = app._shipping.resolve_bottle_type(bottle_slug, product)
    expected_box_counts = {bottle_type: bottle_qty}
    expected = app._shipping.quote(expected_box_counts)
    assert not expected.get("error")                     # box-fit catalog actually resolved
    assert expected["shipping_cents"] > 0

    seen_box_counts = []
    real_quote = app._shipping.quote

    def _spy_quote(box_counts, *a, **kw):
        seen_box_counts.append(dict(box_counts))
        return real_quote(box_counts, *a, **kw)

    monkeypatch.setattr(app._shipping, "quote", _spy_quote)

    bottles_only = app._price_inhouse_invoice(
        [{"slug": bottle_slug, "qty": bottle_qty}],
        email="", pickup=False, ship=US)
    bottles_plus_service = app._price_inhouse_invoice(
        [{"slug": bottle_slug, "qty": bottle_qty},
         {"slug": "biofield-analysis", "qty": 1}],
        email="", pickup=False, ship=US)

    assert bottles_only["shipping_cents"] == expected["shipping_cents"]
    assert bottles_plus_service["shipping_cents"] == expected["shipping_cents"]
    # The real assertion: the service line must never have reached the box-fit
    # call. Both carts must have quoted the identical box_counts derived from
    # the bottle line alone -- no "default" key, no inflated count.
    assert seen_box_counts == [expected_box_counts, expected_box_counts]


def test_services_only_cart_has_no_shipping_without_pickup():
    """The derived rule: nothing physical -> no shipping, no flag required."""
    res = app._price_inhouse_invoice(
        [{"slug": "biofield-analysis", "qty": 1}],
        email="", pickup=False, ship=US)
    assert res["shipping_cents"] == 0


def test_bottles_still_ship_when_not_pickup():
    """Guard against the fix over-reaching: real products still cost shipping."""
    res = app._price_inhouse_invoice(
        [{"slug": "neuro-magnesium", "qty": 1}],
        email="", pickup=False, ship=US)
    assert res["shipping_cents"] > 0
