import json
from pathlib import Path


def test_clear_the_way_and_scar_silk_are_distinct_functional_formulations():
    products = json.loads(
        (Path(__file__).resolve().parent.parent / "data" / "products.json").read_text()
    )["products"]

    clear = products["clear-the-way"]
    scar = products["scar-silk"]

    assert clear["name"] == "Clear the Way"
    assert clear["fmp_id"] == "1125"
    assert clear["bottle_type"] == "30 Caps"
    assert clear["qty_pricing"] is True
    assert "5 Ways" not in clear["name"]

    assert scar["name"] == "Scar Silk"
    assert scar["fmp_id"] == "324"
    assert scar["bottle_type"] == "30 Caps"
