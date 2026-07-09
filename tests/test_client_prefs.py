"""Per-client fulfillment preference. Mirrors the client_prices test shape."""
import sqlite3
import sys
from pathlib import Path

import pytest

repo_root = Path(__file__).resolve().parent.parent
if str(repo_root) not in sys.path:
    sys.path.insert(0, str(repo_root))


def _cx():
    from dashboard import client_prefs as C
    cx = sqlite3.connect(":memory:")
    C.init_table(cx)
    return C, cx


def test_unset_client_defaults_to_no_pickup():
    C, cx = _cx()
    assert C.get_pickup_default(cx, "nobody@x.com") is False


def test_set_get_round_trip_and_is_case_insensitive():
    C, cx = _cx()
    C.set_pickup_default(cx, "Bobbi@X.com", True)
    assert C.get_pickup_default(cx, "bobbi@x.com") is True
    assert C.get_pickup_default(cx, "  BOBBI@x.com  ") is True


def test_set_is_idempotent_and_reversible():
    C, cx = _cx()
    C.set_pickup_default(cx, "bobbi@x.com", True)
    C.set_pickup_default(cx, "bobbi@x.com", True)     # upsert, not a second row
    assert cx.execute("SELECT COUNT(*) FROM client_prefs").fetchone()[0] == 1
    C.set_pickup_default(cx, "bobbi@x.com", False)    # explicit flip back
    assert C.get_pickup_default(cx, "bobbi@x.com") is False


def test_scoped_to_client():
    C, cx = _cx()
    C.set_pickup_default(cx, "bobbi@x.com", True)
    assert C.get_pickup_default(cx, "other@x.com") is False


def test_empty_email_is_rejected_on_write_and_false_on_read():
    C, cx = _cx()
    assert C.get_pickup_default(cx, "") is False
    assert C.get_pickup_default(cx, None) is False
    with pytest.raises(ValueError):
        C.set_pickup_default(cx, "  ", True)


def test_init_table_is_idempotent():
    C, cx = _cx()
    C.init_table(cx)  # second call must not raise
    C.set_pickup_default(cx, "a@x.com", True)
    assert C.get_pickup_default(cx, "a@x.com") is True


def test_only_the_console_endpoint_writes_the_pickup_default():
    """The design's load-bearing promise: creating or saving an order never
    writes a client's pickup default. Exactly one call site in app.py may write
    it — the explicit console endpoint. If this count changes, an order path has
    almost certainly started persisting a per-order override as a preference."""
    src = (repo_root / "app.py").read_text()
    assert src.count("set_pickup_default") == 1


def test_the_order_builder_never_posts_a_pickup_default_with_an_order():
    """The order payloads carry `pickup` (per-order) and never `pickup_default`."""
    src = (repo_root / "static" / "order-new.html").read_text()
    for fn in ("async function createInvoice()", "async function editInvoice()"):
        start = src.index(fn)
        body = src[start:src.index("\n}", start)]
        assert "pickup_default" not in body, f"{fn} must not send pickup_default"
