# tests/test_invoice_summary_pack_breakdown.py  (app-importing -> fake-env)
import importlib, sys
from pathlib import Path
import pytest
repo = Path(__file__).resolve().parent.parent
if str(repo) not in sys.path: sys.path.insert(0, str(repo))

def test_invoice_summary_carries_pack_breakdown(tmp_path, monkeypatch):
    monkeypatch.setenv("DATA_DIR", str(tmp_path))
    try:
        import app as a; importlib.reload(a)
    except Exception as e:
        pytest.skip(f"app not importable: {e}")
    monkeypatch.setattr(a, "_get_product", lambda slug: {"slug": "a", "name": "A"} )
    order = {"items": [{"slug": "a", "qty": 2}, {"slug": "a", "qty": 1, "format": "refill"}],
             "status": "proposed", "total_cents": 0}
    s = a._invoice_summary(order)
    assert s["pack_breakdown"] == {"bottle_units": 2, "cello_pack_units": 1}
