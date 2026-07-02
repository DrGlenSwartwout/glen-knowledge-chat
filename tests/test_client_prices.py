import sqlite3
import sys
from pathlib import Path

repo_root = Path(__file__).resolve().parent.parent
if str(repo_root) not in sys.path:
    sys.path.insert(0, str(repo_root))


def _cx():
    from dashboard import client_prices as C
    cx = sqlite3.connect(":memory:")
    C.init_table(cx)
    return C, cx


def test_set_get_is_case_insensitive_and_upserts():
    C, cx = _cx()
    C.set_price(cx, "JC@X.com", "neuro-magnesium", 4500, "FF rate")
    assert C.get_price(cx, "jc@x.com", "neuro-magnesium") == 4500
    C.set_price(cx, "jc@x.com", "neuro-magnesium", 4200)     # upsert
    assert C.get_price(cx, "JC@x.com", "neuro-magnesium") == 4200


def test_scoped_to_client_and_slug():
    C, cx = _cx()
    C.set_price(cx, "jc@x.com", "terrain-restore", 5000)
    assert C.get_price(cx, "jc@x.com", "nope") is None
    assert C.get_price(cx, "other@x.com", "terrain-restore") is None


def test_price_map_and_list_and_remove():
    C, cx = _cx()
    C.set_price(cx, "jc@x.com", "a", 100)
    C.set_price(cx, "jc@x.com", "b", 200)
    assert C.price_map(cx, "jc@x.com") == {"a": 100, "b": 200}
    assert len(C.list_for(cx, "jc@x.com")) == 2
    assert C.remove(cx, "jc@x.com", "a") is True
    assert C.price_map(cx, "jc@x.com") == {"b": 200}
    assert C.clients_with_prices(cx) == [{"email": "jc@x.com", "count": 1}]


def test_ff_flat_is_separate_from_per_sku_views():
    C, cx = _cx()
    C.set_price(cx, "jc@x.com", "neuro-magnesium", 4500)
    C.set_ff_flat(cx, "jc@x.com", 4200)
    assert C.get_ff_flat(cx, "jc@x.com") == 4200
    # the flat rate must NOT leak into per-SKU views
    assert "neuro-magnesium" in C.price_map(cx, "jc@x.com")
    assert C.FF_FLAT_SLUG not in C.price_map(cx, "jc@x.com")
    assert all(r["slug"] != C.FF_FLAT_SLUG for r in C.list_for(cx, "jc@x.com"))
    assert C.get_ff_flat(cx, "nobody@x.com") is None
    assert C.remove_ff_flat(cx, "jc@x.com") is True
    assert C.get_ff_flat(cx, "jc@x.com") is None


def test_rejects_negative_and_blank():
    C, cx = _cx()
    for bad in [("jc@x.com", "s", -1), ("", "s", 100), ("jc@x.com", "", 100)]:
        try:
            C.set_price(cx, *bad)
            assert False, f"expected ValueError for {bad}"
        except ValueError:
            pass
