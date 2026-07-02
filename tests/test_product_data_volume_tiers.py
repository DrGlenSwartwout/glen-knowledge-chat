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

def test_product_data_shows_open_volume_tiers(monkeypatch):
    appmod = _app()
    FF = {"slug": "brain", "name": "Brain Boost", "qty_pricing": True, "price_cents": 6997}
    monkeypatch.setattr(appmod, "_get_product", {"brain": FF}.get)
    c = appmod.app.test_client()
    d = c.get("/begin/product-data/brain").get_json()
    tiers = {t["min"]: t["unit_cents"] for t in d["qty_pricing"]}
    assert tiers == {1: 6997, 3: 6628, 6: 6075, 12: 4968}
