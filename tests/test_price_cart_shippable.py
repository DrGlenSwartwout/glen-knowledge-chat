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


def test_a_service_line_does_not_change_shipping():
    """Four bottles ship for the same price with or without a Biofield Analysis
    added. Before this fix the service counted as a fifth bottle and could push
    the order up a box size."""
    bottles_only = app._price_inhouse_invoice(
        [{"slug": "neuro-magnesium", "qty": 4}],
        email="", pickup=False, ship=US)
    bottles_plus_service = app._price_inhouse_invoice(
        [{"slug": "neuro-magnesium", "qty": 4},
         {"slug": "biofield-analysis", "qty": 1}],
        email="", pickup=False, ship=US)
    assert bottles_only["shipping_cents"] > 0            # a real box was quoted
    assert bottles_plus_service["shipping_cents"] == bottles_only["shipping_cents"]


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
