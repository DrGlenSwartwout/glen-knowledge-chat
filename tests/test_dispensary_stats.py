"""Dispensary product-dispense ranking + practice-type FF recommendations."""
import json as _json
import os
import sqlite3
import tempfile

from dashboard import dispensary_stats as ds

CAT = {"bone-builder": {"name": "Bone Builder"}, "nous-energy": {"name": "Nous Energy"}}


# ── rank_dispense_rows (pure) ──────────────────────────────────────────────
def test_rank_merges_channels_and_sorts_by_total():
    rows = ds.rank_dispense_rows({"bone-builder": 10, "nous-energy": 2},
                                 {"bone-builder": 5}, {}, catalog=CAT)
    assert [r["slug"] for r in rows] == ["bone-builder", "nous-energy"]  # 15 vs 2
    top = rows[0]
    assert top["name"] == "Bone Builder"
    assert top["url"] == "/begin/product/bone-builder"
    assert top["dispensed"] == 10 and top["dropshipped"] == 5
    assert top["patient_portal"] == 0 and top["total"] == 15


def test_rank_unknown_slug_falls_back_to_slug_name():
    rows = ds.rank_dispense_rows({"mystery-x": 3}, {}, {}, catalog=CAT)
    assert rows[0]["name"] == "mystery-x"


# ── dispense_stats (collector) ─────────────────────────────────────────────
def _seed(tmp_path):
    p = os.path.join(tmp_path, "chat_log.db")
    cx = sqlite3.connect(p)
    cx.executescript(
        "CREATE TABLE wholesale_orders(invoice_id TEXT PRIMARY KEY, practitioner_id TEXT);"
        "CREATE TABLE dispensary_orders(invoice_id TEXT PRIMARY KEY, practitioner_id TEXT, bottles INT);"
        "CREATE TABLE orders(id INTEGER PRIMARY KEY, source TEXT, external_ref TEXT, items_json TEXT);")
    cx.execute("INSERT INTO wholesale_orders VALUES('INV1','p1')")
    cx.execute("INSERT INTO dispensary_orders VALUES('INV2','p1',3)")
    cx.execute("INSERT INTO orders(source,external_ref,items_json) VALUES('a','INV1',?)",
               (_json.dumps([{"slug": "bone-builder", "qty": 10}]),))
    cx.execute("INSERT INTO orders(source,external_ref,items_json) VALUES('b','INV2',?)",
               (_json.dumps([{"slug": "bone-builder", "qty": 3}, {"slug": "nous-energy", "qty": 1}]),))
    cx.commit(); cx.close()
    return p


def test_dispense_stats_buckets_by_channel(tmp_path):
    rows = ds.dispense_stats("p1", db_path=_seed(str(tmp_path)), catalog=CAT)
    by = {r["slug"]: r for r in rows}
    assert by["bone-builder"]["dispensed"] == 10
    assert by["bone-builder"]["dropshipped"] == 3
    assert by["bone-builder"]["total"] == 13 and rows[0]["slug"] == "bone-builder"
    assert by["nous-energy"]["dropshipped"] == 1 and by["nous-energy"]["patient_portal"] == 0


def test_dispense_stats_never_raises_on_bad_db():
    assert ds.dispense_stats("p1", db_path="/nonexistent/x.db") == []


# ── recommended_ffs (resolver) ─────────────────────────────────────────────
def test_recommended_resolves_type_then_default_and_excludes():
    recs = {"default": [{"slug": "a", "blurb": "A."}, {"slug": "b", "blurb": "B."}],
            "OD": [{"slug": "c", "blurb": "C."}]}
    p = os.path.join(tempfile.mkdtemp(), "r.json")
    open(p, "w").write(_json.dumps(recs))
    cat = {"a": {"name": "A"}, "b": {"name": "B"}, "c": {"name": "C"}}
    od = ds.recommended_ffs("od", recs_path=p, catalog=cat)          # case-insensitive
    assert [r["slug"] for r in od] == ["c"] and od[0]["url"] == "/begin/product/c"
    dflt = ds.recommended_ffs("Unknown", exclude_slugs=["a"], recs_path=p, catalog=cat)
    assert [r["slug"] for r in dflt] == ["b"]                        # default minus excluded
