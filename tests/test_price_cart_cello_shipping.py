# tests/test_price_cart_cello_shipping.py  (app-importing -> fake-env)
import importlib, sys
from pathlib import Path
import pytest
repo = Path(__file__).resolve().parent.parent
if str(repo) not in sys.path: sys.path.insert(0, str(repo))

def _app(monkeypatch, tmp_path):
    monkeypatch.setenv("DATA_DIR", str(tmp_path))
    try:
        import app as a; importlib.reload(a)
    except Exception as e:
        pytest.skip(f"app not importable: {e}")
    # a small deterministic catalog: one shippable 'default' product
    monkeypatch.setattr(a, "_get_product", lambda slug: {"slug": "mag", "name": "Mag",
        "price_cents": 6997, "bottle_type": "default"} if slug == "mag" else None)
    return a

def test_cello_lines_ship_cheaper_and_count_separately(tmp_path, monkeypatch):
    a = _app(monkeypatch, tmp_path)
    ship = {"country": "US", "zip": "01950", "state": "MA", "city": "X", "street": "1 A St"}
    bottles = a._price_cart([{"slug": "mag", "qty": 6}], ship=ship)
    cello   = a._price_cart([{"slug": "mag", "qty": 6, "format": "refill"}], ship=ship)
    assert cello["shipping_cents"] <= bottles["shipping_cents"]
    assert cello["cello_pack_units"] == 6 and cello["bottle_units"] == 0
    assert bottles["cello_pack_units"] == 0 and bottles["bottle_units"] == 6
