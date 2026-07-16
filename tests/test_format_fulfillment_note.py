"""A non-default packaging format is recorded as a fulfillment note on the order line
(kanban name + QBO description), for any flow that prices through _price_cart. The
QBO line NAME stays clean so item mapping is unaffected. Run under Doppler (imports app)."""
import pytest

app = pytest.importorskip("app")

SHIP = {"country": "US", "state": "HI", "name": "T"}


def _price(slug, **extra):
    item = {"slug": slug, "qty": 1}
    item.update(extra)
    return app._price_cart([item], ship=SHIP)


def test_refill_format_decorates_name_and_description():
    plain = _price("wholomega")["items_rec"][0]["name"]
    pc = _price("wholomega", format="refill")
    assert pc["items_rec"][0]["name"] == plain + " — Cellophane refill packs"
    assert pc["qbo_lines"][0]["description"] == plain + " — Cellophane refill packs"
    assert pc["qbo_lines"][0]["name"] == plain  # QBO line NAME stays clean for item mapping


def test_larger_format_decorates():
    plain = _price("wholomega")["items_rec"][0]["name"]
    assert _price("wholomega", format="larger")["items_rec"][0]["name"] == plain + " — Larger bottle"


def test_bottle_default_is_not_decorated():
    plain = _price("wholomega")["items_rec"][0]["name"]
    assert "—" not in plain
    assert _price("wholomega", format="bottle")["items_rec"][0]["name"] == plain
    assert _price("wholomega", format="bottle")["qbo_lines"][0]["description"] == plain


def test_no_format_is_not_decorated():
    plain = _price("wholomega")["items_rec"][0]["name"]
    assert _price("wholomega")["items_rec"][0]["name"] == plain
    assert "—" not in _price("wholomega")["qbo_lines"][0]["description"]
