import sqlite3

from dashboard import portal_health_history as hh


def test_current_products_roundtrip_and_clear_unchecked_text():
    cx = sqlite3.connect(":memory:")
    hh.save(cx, "Client@Example.com", {
        "prescriptions_yes": True,
        "prescriptions_text": "Pfizer Lipitor",
        "otc_yes": False,
        "otc_text": "must not persist",
        "supplements_yes": True,
        "supplements_text": "NOW Magnesium\nNordic Naturals Omega-3",
    })
    got = hh.get(cx, "client@example.com")
    assert got["prescriptions_text"] == "Pfizer Lipitor"
    assert got["otc_yes"] is False
    assert got["otc_text"] == ""
    assert got["supplements_text"].splitlines() == [
        "NOW Magnesium", "Nordic Naturals Omega-3"]
    assert hh.has(cx, "CLIENT@example.com") is True
