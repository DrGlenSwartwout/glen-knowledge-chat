import json, pathlib, sqlite3, tempfile, os

from dashboard import shipping

def _products():
    p = pathlib.Path(__file__).resolve().parent.parent / "data" / "products.json"
    return json.loads(p.read_text())["products"]

def test_hand_cradle_sku_present():
    p = _products()["hand-cradle"]
    assert p["price_cents"] == 29700
    assert p.get("info_only") is not True          # physical: goes through the packer
    assert p.get("bottle_type") == "handcradle"    # registered packer dim, resolves to M box

def test_hand_cradle_packs_into_medium_flat_rate_box():
    fd, db_path = tempfile.mkstemp(suffix=".db")
    os.close(fd)
    try:
        cx = sqlite3.connect(db_path)
        shipping.init_shipping_schema(cx)
        cx.commit()
        cx.close()
        assert shipping.pick_box({"handcradle": 1}, db_path=db_path) == "M"
    finally:
        os.remove(db_path)

def test_evox_session_sku_present():
    p = _products()["evox-session"]
    assert p["price_cents"] == 19700          # public list; member $100 applied by Rae at invoice
    assert p["info_only"] is True and p["service"] is True   # prepay service, no shipping
