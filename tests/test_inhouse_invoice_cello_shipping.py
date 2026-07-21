# tests/test_inhouse_invoice_cello_shipping.py  (app-importing -> fake-env)
#
# Regression test for the console order path: `_price_inhouse_invoice` built the
# cart it hands to `_price_cart` WITHOUT the line `format`, so cello (`refill`)
# lines were priced/packed as rigid bottles on the console order-entry / invoice-
# edit path -- the exact over-charge the cello-pack feature exists to fix. Counts
# and display already showed cello correctly (they read the stored `format`), so
# before the fix "counts say cello, rate says bottle." This test proves the fix
# THROUGH `_price_inhouse_invoice` (the path that was broken), not through
# `_price_cart` directly (already covered by test_price_cart_cello_shipping.py).
import importlib, sys
from pathlib import Path
import pytest
repo = Path(__file__).resolve().parent.parent
if str(repo) not in sys.path: sys.path.insert(0, str(repo))


def _app(monkeypatch, tmp_path):
    # Fresh DATA_DIR -> a fresh shipping db, which seeds `cello-refill` dims
    # (_STANDARD_BOTTLES) on init, matching a real fresh deploy.
    monkeypatch.setenv("DATA_DIR", str(tmp_path))
    try:
        import app as a; importlib.reload(a)
    except Exception as e:
        pytest.skip(f"app not importable: {e}")
    # A deterministic, shippable, non-FF product -- bottle_type "default" mirrors
    # the console order-entry test pattern in test_price_cart_cello_shipping.py.
    monkeypatch.setattr(a, "_get_product", lambda slug: {
        "slug": "mag", "name": "Mag", "price_cents": 6997, "bottle_type": "default",
    } if slug == "mag" else None)
    return a


def test_inhouse_invoice_refill_line_ships_cheaper_than_bottle_line(tmp_path, monkeypatch):
    a = _app(monkeypatch, tmp_path)
    ship = {"country": "US", "zip": "01950", "state": "MA", "city": "X", "street": "1 A St"}

    bottle_result = a._price_inhouse_invoice(
        [{"slug": "mag", "qty": 6}], email="c@x.com", pickup=False, ship=ship)
    refill_result = a._price_inhouse_invoice(
        [{"slug": "mag", "qty": 6, "format": "refill"}], email="c@x.com", pickup=False, ship=ship)

    assert bottle_result is not None and refill_result is not None
    assert bottle_result["shipping_cents"] > 0
    assert refill_result["shipping_cents"] > 0
    # Strict less-than (not <=): with the bug, `format` never reached the cart, so
    # both lines built an IDENTICAL cart ({"slug": "mag", "qty": 6}, no format) and
    # priced EQUAL -- a `<=` assertion would pass on the broken code too. Confirmed
    # empirically: pre-fix both = 2300c; post-fix bottle=2300c, refill=1300c.
    assert refill_result["shipping_cents"] < bottle_result["shipping_cents"]
