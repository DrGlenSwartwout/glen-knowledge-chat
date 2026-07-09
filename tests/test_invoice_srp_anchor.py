"""The invoice's struck-through Value (SRP) anchor.

Priority: the product's own regular_cents (FMP retail_sug_price) when it exceeds the
charge price, else the flat $80 anchor for a $69.97 FF, else no anchor at all.
Before this was wired, regular_cents sat unread on 634 products and only FFs priced
at exactly 6997 ever showed a Value.
"""
import importlib
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent


def _app():
    if str(ROOT) not in sys.path:
        sys.path.insert(0, str(ROOT))
    try:
        return importlib.import_module("app")
    except Exception as e:
        pytest.skip(f"app not importable: {e}")


def _view(appmod, product, unit_cents=None, slug="x"):
    unit = product["price_cents"] if unit_cents is None else unit_cents
    orig = appmod._get_product
    try:
        appmod._get_product = lambda s: dict(product, slug=s) if s == slug else orig(s)
        return appmod._invoice_line_view(
            {"slug": slug, "name": "X", "qty": 1, "unit_cents": unit, "line_cents": unit})
    finally:
        appmod._get_product = orig


def test_explicit_srp_wins_for_a_non_ff():
    """An essence at $70 with an $80 SRP now anchors — it never did before."""
    appmod = _app()
    out = _view(appmod, {"price_cents": 7000, "regular_cents": 8000})
    assert out["srp_cents"] == 8000 and out["regular_cents"] == 7000


def test_infoceutical_anchors_at_its_own_srp():
    appmod = _app()
    out = _view(appmod, {"price_cents": 3997, "regular_cents": 4000})
    assert out["srp_cents"] == 4000 and out["regular_cents"] == 3997


def test_off_base_ff_uses_its_own_srp_not_the_flat_80():
    """CDS ($35/$40) and WholOmega 120ct ($190/$230) are FFs off the $69.97 base.
    They previously showed no anchor because the flat rule keys on price == 6997."""
    appmod = _app()
    cds = _view(appmod, {"price_cents": 3500, "regular_cents": 4000, "qty_pricing": True})
    assert cds["srp_cents"] == 4000
    wo = _view(appmod, {"price_cents": 19000, "regular_cents": 23000, "qty_pricing": True})
    assert wo["srp_cents"] == 23000


def test_ff_without_explicit_srp_falls_back_to_the_flat_80():
    """151 FFs carry no regular_cents; the derived $80 rule must still hold."""
    appmod = _app()
    out = _view(appmod, {"price_cents": 6997, "qty_pricing": True})
    assert out["srp_cents"] == 8000 and out["regular_cents"] == 6997


def test_ff_with_explicit_srp_is_unchanged_at_80():
    appmod = _app()
    out = _view(appmod, {"price_cents": 6997, "regular_cents": 8000, "qty_pricing": True})
    assert out["srp_cents"] == 8000


def test_incoherent_srp_below_price_is_ignored():
    """FMP had rows with retail_sug_price < sold_price. Never anchor off those."""
    appmod = _app()
    out = _view(appmod, {"price_cents": 7000, "regular_cents": 4000})
    assert out["srp_cents"] == 7000 == out["regular_cents"]   # Value == Regular -> no anchor


def test_srp_equal_to_price_is_not_an_anchor():
    appmod = _app()
    out = _view(appmod, {"price_cents": 7000, "regular_cents": 7000})
    assert out["srp_cents"] == out["regular_cents"]


def test_info_only_never_anchors():
    appmod = _app()
    out = _view(appmod, {"price_cents": 6997, "regular_cents": 8000, "info_only": True})
    assert out["srp_cents"] == 6997


def test_no_product_carries_an_incoherent_srp():
    import json
    prods = json.loads((ROOT / "data" / "products.json").read_text())["products"]
    bad = {s: (p["price_cents"], p["regular_cents"]) for s, p in prods.items()
           if p.get("regular_cents") is not None and p["regular_cents"] <= p.get("price_cents", 0)}
    assert not bad, f"regular_cents must exceed price_cents or be absent: {bad}"
