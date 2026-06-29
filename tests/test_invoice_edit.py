"""Console invoice editor: shared in-house pricing + best-effort QBO push gating.

Imports app (needs real secrets + writable DATA_DIR), so it's skipped under plain
pytest and runs under the Doppler harness:
  doppler run -p remedy-match -c prd -- env DATA_DIR=/tmp/scratch python3 -m pytest tests/test_invoice_edit.py
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


def test_price_inhouse_honors_overrides_service_and_pickup():
    # 2 capsules at a $50 override + a Biofield service line at its $100 special;
    # local pickup → no shipping; no discount/points → total == subtotal.
    res = app._price_inhouse_invoice(
        [{"slug": "neuro-magnesium", "qty": 2, "unit_cents": 5000},
         {"slug": "biofield-analysis", "qty": 1, "unit_cents": 10000}],
        email="someone@example.com", pickup=True,
        ship={"name": "T", "state": "HI", "country": "US"})
    assert res is not None
    assert res["subtotal_cents"] == 20000
    assert res["shipping_cents"] == 0
    assert res["discount_cents"] == 0 and res["points_redeemed_cents"] == 0
    assert res["total_cents"] == 20000
    by_slug = {i["slug"]: i for i in res["items_rec"]}
    assert by_slug["neuro-magnesium"]["unit_cents"] == 5000
    assert by_slug["neuro-magnesium"]["line_cents"] == 10000
    bio = by_slug["biofield-analysis"]
    assert bio["unit_cents"] == 10000 and bio["line_cents"] == 10000 and bio.get("service") is True


def test_price_inhouse_discount_reduces_total():
    res = app._price_inhouse_invoice(
        [{"slug": "neuro-magnesium", "qty": 1, "unit_cents": 6997}],
        email="", pickup=True, ship={"country": "US"},
        discount_cents_in=1000)
    assert res["subtotal_cents"] == 6997
    assert res["discount_cents"] == 1000
    assert res["total_cents"] == 5997


def test_price_inhouse_pickup_neutralizes_non_us_country():
    # A pickup order with a stored non-US address must still price (no shipment) —
    # _price_cart's non-US guard would otherwise reject it.
    res = app._price_inhouse_invoice(
        [{"slug": "neuro-magnesium", "qty": 1, "unit_cents": 5000}],
        email="", pickup=True, ship={"country": "PH", "state": ""})
    assert res is not None
    assert res["shipping_cents"] == 0
    assert res["total_cents"] == 5000


def test_price_inhouse_none_when_no_valid_products():
    assert app._price_inhouse_invoice(
        [{"slug": "not-a-real-slug", "qty": 1}],
        email="", pickup=True, ship={"country": "US"}) is None


def test_qbo_push_skips_inhouse_ref():
    # INH-* orders have no QBO invoice — the push is a no-op, never raises.
    out = app._push_invoice_edit_to_qbo("INH-ABC123", {"items_rec": [], "shipping_cents": 0,
                                                       "discount_cents": 0, "points_redeemed_cents": 0})
    assert out["pushed"] is False and "skipped" in out
