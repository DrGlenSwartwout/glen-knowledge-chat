import sys
from pathlib import Path

repo_root = Path(__file__).resolve().parent.parent
if str(repo_root) not in sys.path:
    sys.path.insert(0, str(repo_root))


def test_is_configured_reads_env(monkeypatch):
    from dashboard import easypost as EP
    monkeypatch.delenv("EASYPOST_API_KEY", raising=False)
    assert EP.is_configured() is False
    monkeypatch.setenv("EASYPOST_API_KEY", "ezk_test")
    assert EP.is_configured() is True


def test_build_shipment_shape():
    from dashboard import easypost as EP
    order = {"name": "Ann Buyer", "address": {"street": "1 Main St", "city": "Hilo",
             "state": "HI", "zip": "96720"}, "items": [{"qty": 2}]}
    s = EP.build_shipment(order, from_address={"name": "Remedy Match", "street": "x",
                          "city": "Hilo", "state": "HI", "zip": "96720"})
    assert s["to_address"]["name"] == "Ann Buyer"
    assert s["to_address"]["zip"] == "96720"
    assert s["from_address"]["zip"] == "96720"
    assert s["parcel"]["weight"] > 0  # ounces, derived from item count


def test_clicknship_url_constant():
    from dashboard import easypost as EP
    assert EP.CLICKNSHIP_URL.startswith("https://")
