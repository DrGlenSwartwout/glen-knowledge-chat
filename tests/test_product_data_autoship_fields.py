import pytest
app = pytest.importorskip("app")

def _data(slug):
    with app.app.test_client() as c:
        r = c.get(f"/begin/product-data/{slug}")
        assert r.status_code == 200, (slug, r.status_code)
        return r.get_json()

def test_bundle_exposes_bundle_ladder():
    d = _data("crystalline-lens-program")
    assert d["autoship_eligible"] is True
    assert d["bundle"] is True
    assert d["autoship"]["first_pct"] == 12
    assert d["autoship"]["cap_pct"] == 29

def test_ff_single_sku_exposes_standard_ladder():
    d = _data("wholomega")  # an FF single SKU (qty_pricing) -> autoship-eligible
    assert d["autoship_eligible"] is True
    assert d["bundle"] is False
    assert d["autoship"]["first_pct"] == 3
    assert d["autoship"]["cap_pct"] == 25

def test_device_bundle_not_autoship_eligible():
    d = _data("dental-bundle")
    assert d["autoship_eligible"] is False
    assert "autoship" not in d  # no ladder block for non-eligible

def test_non_ff_device_single_sku_not_eligible():
    d = _data("water-ionizer-5plate")  # not an FF (qty_pricing unset) -> excluded
    assert d["autoship_eligible"] is False
    assert "autoship" not in d

def test_is_paid_member_present_and_false_for_anon():
    d = _data("crystalline-lens-program")
    assert d["is_paid_member"] is False  # no auth cookie in the test client
