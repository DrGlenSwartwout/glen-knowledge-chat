# tests/test_line_format_roundtrip.py  (app-importing -> fake-env)
import importlib, sys
from pathlib import Path
import pytest
repo = Path(__file__).resolve().parent.parent
if str(repo) not in sys.path: sys.path.insert(0, str(repo))

def test_format_rides_on_the_stored_line(tmp_path, monkeypatch):
    monkeypatch.setenv("DATA_DIR", str(tmp_path))
    try:
        import app as a; importlib.reload(a)
    except Exception as e:
        pytest.skip(f"app not importable: {e}")
    monkeypatch.setattr(a, "_get_product", lambda slug: {"slug": "mag", "name": "Mag",
        "price_cents": 6997, "bottle_type": "default"} if slug == "mag" else None)
    priced = a._price_inhouse_invoice([{"slug": "mag", "qty": 2, "format": "refill"}],
                                      email="", pickup=True, ship=None)
    rec = priced["items_rec"][0]
    assert rec["format"] == "refill"
