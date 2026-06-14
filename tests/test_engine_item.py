# tests/test_engine_item.py
import app as appmod

def test_engine_item_capsule_defaults():
    p = {"slug": "brain-boost", "name": "Brain Boost", "price_cents": 6997, "qty_pricing": True}
    it = appmod._engine_item(p, 3)
    assert it["unit_cents"] == 6997          # TRUE single-unit list
    assert it["months"] == 3                 # 1 month per unit * qty 3
    assert it["volume_eligible"] is True
    assert "sku_discount_floor_pct" not in it["product"]   # capsule uses global floors

def test_engine_item_pure_powder_excluded_and_floored():
    p = {"slug": "sumac-pure-powder", "name": "Sumac 50:1 Pure Powder", "price_cents": 3997}
    it = appmod._engine_item(p, 2)
    assert it["volume_eligible"] is False     # Pure Powders off the curve
    assert it["product"]["sku_discount_floor_pct"] == 0.75   # ~$30 on $40
    assert it["product"]["sku_points_floor_pct"] == 0.75

def test_engine_item_explicit_fields_win():
    p = {"slug": "x", "name": "X", "price_cents": 7000,
         "volume_eligible": False, "months_per_unit": 6}
    it = appmod._engine_item(p, 2)
    assert it["months"] == 12                 # 6 * 2
    assert it["volume_eligible"] is False
