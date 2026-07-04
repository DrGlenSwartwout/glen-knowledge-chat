import json, pathlib

def _products():
    p = pathlib.Path(__file__).resolve().parent.parent / "data" / "products.json"
    return json.loads(p.read_text())["products"]

def test_hand_cradle_sku_present():
    p = _products()["hand-cradle"]
    assert p["price_cents"] == 29700
    assert p.get("info_only") is not True          # physical: goes through the packer
    assert p.get("bottle_type")                     # must have a packer dim so shipping is billed

def test_evox_session_sku_present():
    p = _products()["evox-session"]
    assert p["price_cents"] == 19700          # public list; member $100 applied by Rae at invoice
    assert p["info_only"] is True and p["service"] is True   # prepay service, no shipping
