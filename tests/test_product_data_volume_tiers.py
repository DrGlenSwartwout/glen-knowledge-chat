import importlib, sys
from pathlib import Path
import pytest

def _app():
    root = Path(__file__).resolve().parent.parent
    if str(root) not in sys.path:
        sys.path.insert(0, str(root))
    try:
        return importlib.import_module("app")
    except Exception as e:
        pytest.skip(f"app not importable: {e}")

def test_product_data_shows_same_sku_qty_tiers(monkeypatch):
    """The /begin/product-data qty tiers are single-SKU (type-1 same_sku_pct) pricing,
    not the order-total volume_pct ramp (that one's OWNER in-house only). The 12-tier is
    the FF minimum unit price ($50): the 29% ramp lands at $49.68, then clamps up to the
    ff_min_unit_cents floor — the display must match the charge."""
    appmod = _app()
    FF = {"slug": "brain", "name": "Brain Boost", "qty_pricing": True, "price_cents": 6997}
    monkeypatch.setattr(appmod, "_get_product", {"brain": FF}.get)
    c = appmod.app.test_client()
    d = c.get("/begin/product-data/brain").get_json()
    tiers = {t["min"]: t["unit_cents"] for t in d["qty_pricing"]}
    assert tiers == {1: 6997, 3: 6628, 6: 6075, 12: 5000}
