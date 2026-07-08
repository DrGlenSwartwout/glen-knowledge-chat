"""Every Functional Formulation must sit on the $69.97 base.

The invoice's $80 "Value" anchor is DERIVED, not stored: `_invoice_line_view` emits
srp_cents = _FF_SRP_CENTS only when price_cents == _FF_BASE_CENTS exactly. An FF
priced at $70.00 (as 58 of them were after the FMP import, which maps FMP's
sold_price straight through) silently collapses Value down to equal Regular.
`regular_cents` in products.json is NOT the anchor — nothing reads it.
"""
import importlib
import json
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent

# Genuinely-different price points, not $69.97 capsule FFs. Value == Regular on these.
OFF_BASE_ALLOWED = {
    "cds": 3500,
    "cds-activator": 3500,
    "wholomega-120-gelcaps": 19000,   # 120-count; the 30-count is a normal FF
}


def _app():
    if str(ROOT) not in sys.path:
        sys.path.insert(0, str(ROOT))
    try:
        return importlib.import_module("app")
    except Exception as e:
        pytest.skip(f"app not importable: {e}")


def _products():
    return json.loads((ROOT / "data" / "products.json").read_text())["products"]


def test_every_ff_sits_on_the_6997_base():
    appmod = _app()
    offenders = {}
    for slug, p in _products().items():
        if not (p.get("qty_pricing") and not p.get("info_only")):
            continue
        price = p.get("price_cents")
        if slug in OFF_BASE_ALLOWED:
            assert price == OFF_BASE_ALLOWED[slug], f"{slug} allowlisted at a stale price"
            continue
        if price != appmod._FF_BASE_CENTS:
            offenders[slug] = price
    assert not offenders, (
        f"{len(offenders)} FF(s) off the $69.97 base — they will show Value == Regular "
        f"on the invoice instead of the $80 anchor: {offenders}"
    )


def test_on_base_ff_gets_the_80_dollar_value_anchor():
    """The whole point of the base: price 6997 -> Value $80 struck through, Regular $69.97."""
    appmod = _app()
    out = appmod._invoice_line_view(
        {"slug": "mucosa-syntropy-powder", "name": "Mucosa Syntropy Powder",
         "qty": 1, "unit_cents": 6997, "line_cents": 6997}
    )
    assert out["srp_cents"] == 8000        # Value
    assert out["regular_cents"] == 6997    # Regular


def test_catalog_names_have_no_embedded_newlines():
    """A newline in `name` renders as a broken line on the invoice + catalog picker."""
    bad = [s for s, p in _products().items()
           if "\n" in (p.get("name") or "") or "\r" in (p.get("name") or "")]
    assert not bad, f"product name(s) contain a newline: {bad}"
