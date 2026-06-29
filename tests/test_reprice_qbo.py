"""Backfill helper: rebuild priced order items from a QBO invoice's Line array.

Imports app (needs real secrets + a writable DATA_DIR), so it's skipped under plain
pytest and runs under the Doppler harness:
  doppler run -p remedy-match -c prd -- env DATA_DIR=/tmp/scratch python3 -m pytest tests/test_reprice_qbo.py
"""
import sys
from pathlib import Path

import pytest

repo_root = Path(__file__).resolve().parent.parent
if str(repo_root) not in sys.path:
    sys.path.insert(0, str(repo_root))

# app validates Pinecone/Stripe creds at import (raises non-ImportError without them),
# so importorskip isn't enough — skip the whole module if any import error occurs.
try:
    import app
except Exception as _e:  # pragma: no cover - exercised only under plain pytest
    pytest.skip(f"app import requires real secrets: {_e}", allow_module_level=True)


def test_qbo_invoice_to_items_prices_lines_and_drops_shipping():
    cat = {"neuro-magnesium": {"name": "Neuro Magnesium"}, "vitality": {"name": "Vitality"}}
    inv = {"Line": [
        {"DetailType": "SalesItemLineDetail", "Amount": 100.0,
         "SalesItemLineDetail": {"Qty": 2, "ItemRef": {"name": "Neuro Magnesium"}}},
        {"DetailType": "SalesItemLineDetail", "Amount": 50.0, "Description": "Vitality",
         "SalesItemLineDetail": {"Qty": 1}},
        {"DetailType": "SalesItemLineDetail", "Amount": 9.0,
         "SalesItemLineDetail": {"Qty": 1, "ItemRef": {"name": "Shipping"}}},
        {"DetailType": "SubTotalLineDetail", "Amount": 150.0},
    ]}
    items, line_sum = app._qbo_invoice_to_items(inv, cat)
    assert len(items) == 2  # shipping + subtotal lines dropped
    assert items[0] == {"name": "Neuro Magnesium", "qty": 2, "desc": "Neuro Magnesium",
                        "slug": "neuro-magnesium", "unit_cents": 5000, "line_cents": 10000}
    assert items[1]["unit_cents"] == 5000 and items[1]["name"] == "Vitality"
    assert line_sum == 15000


def test_qbo_invoice_to_items_empty():
    items, line_sum = app._qbo_invoice_to_items({"Line": []}, {})
    assert items == [] and line_sum == 0
