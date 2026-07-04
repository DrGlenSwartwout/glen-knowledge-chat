"""Dispensary product-dispense ranking + practice-type FF recommendations.

Fixtures use PRODUCTION-shaped order rows: wholesale checkout ingests real line
items under source='wholesale'; the dispensary sale ingests a slug-less aggregate
stub under source='dispensary' (so drop-ship per-product is deferred).
"""
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
    cx.execute("INSERT INTO dispensary_orders VALUES('INV2','p1',5)")
    cx.execute("INSERT INTO dispensary_orders VALUES('INV3','p1',2)")
    cx.execute("INSERT INTO orders(source,external_ref,items_json) VALUES('wholesale','INV1',?)",
               (_json.dumps([{"slug": "bone-builder", "qty": 10}, {"slug": "nous-energy", "qty": 2}]),))
    cx.execute("INSERT INTO orders(source,external_ref,items_json) VALUES('dispensary','INV2',?)",
               (_json.dumps([{"slug": "bone-builder", "qty": 4}, {"slug": "nous-energy", "qty": 1}]),))
    cx.execute("INSERT INTO orders(source,external_ref,items_json) VALUES('dispensary','INV3',?)",
               (_json.dumps([{"name": "Dispensary", "qty": 2}]),))  # slug-less (webhook) → contributes 0 per-product
    cx.commit(); cx.close()
    return p


def test_dispense_stats_ranks_dispensed_and_dropshipped(tmp_path):
    rows = ds.dispense_stats("p1", db_path=_seed(str(tmp_path)), catalog=CAT)
    by = {r["slug"]: r for r in rows}
    assert rows[0]["slug"] == "bone-builder"
    assert by["bone-builder"]["dispensed"] == 10 and by["bone-builder"]["dropshipped"] == 4
    assert by["bone-builder"]["total"] == 14
    assert by["nous-energy"]["dispensed"] == 2 and by["nous-energy"]["dropshipped"] == 1
    assert all(r["patient_portal"] == 0 for r in rows)  # patient-portal still deferred
    assert set(by) == {"bone-builder", "nous-energy"}   # slug-less webhook order adds no phantom rows


def test_dispense_stats_scopes_by_source_sharing_external_ref(tmp_path):
    # a non-wholesale order sharing the external_ref must not leak into 'dispensed'
    p = _seed(str(tmp_path))
    cx = sqlite3.connect(p)
    cx.execute("INSERT INTO orders(source,external_ref,items_json) VALUES('retail','INV1',?)",
               (_json.dumps([{"slug": "bone-builder", "qty": 999}]),))
    cx.commit(); cx.close()
    rows = ds.dispense_stats("p1", db_path=p, catalog=CAT)
    assert {r["slug"]: r["dispensed"] for r in rows}["bone-builder"] == 10  # not 999


def test_dispense_stats_bad_qty_line_does_not_drop_invoice(tmp_path):
    p = os.path.join(str(tmp_path), "chat_log.db")
    cx = sqlite3.connect(p)
    cx.executescript(
        "CREATE TABLE wholesale_orders(invoice_id TEXT PRIMARY KEY, practitioner_id TEXT);"
        "CREATE TABLE orders(id INTEGER PRIMARY KEY, source TEXT, external_ref TEXT, items_json TEXT);")
    cx.execute("INSERT INTO wholesale_orders VALUES('INV1','p1')")
    cx.execute("INSERT INTO orders(source,external_ref,items_json) VALUES('wholesale','INV1',?)",
               (_json.dumps([{"slug": "bad", "qty": "NaN"}, {"slug": "good", "qty": 4}]),))
    cx.commit(); cx.close()
    rows = ds.dispense_stats("p1", db_path=p, catalog={"good": {"name": "Good"}})
    assert {r["slug"]: r["dispensed"] for r in rows}.get("good") == 4  # good survives bad line


def test_dispense_stats_never_raises_on_bad_db():
    assert ds.dispense_stats("p1", db_path="/nonexistent/x.db") == []


# ── patient_portal_items (union: referred patients + dispensary clients; first + reorders) ──
def _seed_portal(tmp_path):
    p = os.path.join(tmp_path, "chat_log.db")
    cx = sqlite3.connect(p)
    cx.executescript(
        "CREATE TABLE referral_redemptions(referee_email TEXT PRIMARY KEY, owner_email TEXT);"
        "CREATE TABLE dispensary_orders(invoice_id TEXT, practitioner_id TEXT, customer_email TEXT);"
        "CREATE TABLE orders(id INTEGER PRIMARY KEY, source TEXT, external_ref TEXT, email TEXT, status TEXT, items_json TEXT);")
    cx.execute("INSERT INTO referral_redemptions VALUES('referred@x.com','Doc@X.com')")   # doc referred this patient
    cx.execute("INSERT INTO dispensary_orders VALUES('D1','p1','Client@X.com')")           # doc's dispensary client (no referral row)

    def ins(src, ref, email, status, items):
        cx.execute("INSERT INTO orders(source,external_ref,email,status,items_json) VALUES(?,?,?,?,?)",
                   (src, ref, email, status, _json.dumps(items)))
    ins("funnel", "F1", "referred@x.com", "new", [{"slug": "bone-builder", "qty": 3}])       # FIRST purchase counts
    ins("portal-reorder", "R1", "referred@x.com", "new", [{"slug": "bone-builder", "qty": 2}])  # reorder
    ins("reorder", "R2", "client@x.com", "new", [{"slug": "nous-energy", "qty": 4}])         # dispensary client (union)
    ins("portal-reorder", "R3", "stranger@x.com", "new", [{"slug": "bone-builder", "qty": 99}])  # not linked → excluded
    ins("retail", "R4", "referred@x.com", "new", [{"slug": "bone-builder", "qty": 50}])       # non-patient source → excluded
    ins("portal-reorder", "R5", "referred@x.com", "cancelled", [{"slug": "bone-builder", "qty": 7}])  # cancelled → excluded
    cx.commit(); cx.close()
    return p


def test_patient_portal_items_unions_referred_and_dispensary_with_first_order(tmp_path):
    pp = ds.patient_portal_items("doc@x.com", practitioner_id="p1", db_path=_seed_portal(str(tmp_path)))
    # referred: F1 funnel first (3) + R1 reorder (2) = 5; dispensary client: R2 (4); excl R3/R4/R5
    assert pp == {"bone-builder": 5, "nous-energy": 4}


def test_patient_portal_items_empty_without_email_or_id(tmp_path):
    db = _seed_portal(str(tmp_path))
    assert ds.patient_portal_items("nobody@x.com", practitioner_id="pX", db_path=db) == {}
    assert ds.patient_portal_items("", db_path=db) == {}   # no email, no id


def test_dispense_stats_includes_patient_portal(tmp_path):
    p = os.path.join(str(tmp_path), "chat_log.db")
    cx = sqlite3.connect(p)
    cx.executescript(
        "CREATE TABLE wholesale_orders(invoice_id TEXT, practitioner_id TEXT);"
        "CREATE TABLE dispensary_orders(invoice_id TEXT, practitioner_id TEXT, customer_email TEXT);"
        "CREATE TABLE referral_redemptions(referee_email TEXT PRIMARY KEY, owner_email TEXT);"
        "CREATE TABLE orders(id INTEGER PRIMARY KEY, source TEXT, external_ref TEXT, email TEXT, status TEXT, items_json TEXT);")
    cx.execute("INSERT INTO wholesale_orders VALUES('W1','p1')")
    cx.execute("INSERT INTO referral_redemptions VALUES('pat@x.com','doc@x.com')")
    cx.execute("INSERT INTO orders(source,external_ref,email,status,items_json) VALUES('wholesale','W1','','new',?)",
               (_json.dumps([{"slug": "bone-builder", "qty": 6}]),))
    cx.execute("INSERT INTO orders(source,external_ref,email,status,items_json) VALUES('funnel','F1','pat@x.com','new',?)",
               (_json.dumps([{"slug": "bone-builder", "qty": 2}]),))
    cx.commit(); cx.close()
    rows = ds.dispense_stats("p1", practitioner_email="doc@x.com", db_path=p, catalog=CAT)
    bb = {r["slug"]: r for r in rows}["bone-builder"]
    assert bb["dispensed"] == 6 and bb["patient_portal"] == 2 and bb["total"] == 8


# ── recommended_ffs (resolver) ─────────────────────────────────────────────
def _recs_file(recs):
    p = os.path.join(tempfile.mkdtemp(), "r.json")
    open(p, "w").write(_json.dumps(recs))
    return p


def test_recommended_resolves_type_then_default_and_excludes():
    p = _recs_file({"default": [{"slug": "a", "blurb": "A."}, {"slug": "b", "blurb": "B."}],
                    "OD": [{"slug": "c", "blurb": "C."}]})
    cat = {"a": {"name": "A"}, "b": {"name": "B"}, "c": {"name": "C"}}
    od = ds.recommended_ffs("od", recs_path=p, catalog=cat)          # case-insensitive
    assert [r["slug"] for r in od] == ["c"] and od[0]["url"] == "/begin/product/c"
    dflt = ds.recommended_ffs("Unknown", exclude_slugs=["a"], recs_path=p, catalog=cat)
    assert [r["slug"] for r in dflt] == ["b"]                        # default minus excluded


def test_recommended_tokenizes_compound_credentials():
    p = _recs_file({"default": [{"slug": "a", "blurb": "A."}],
                    "OD": [{"slug": "c", "blurb": "C."}],
                    "Health Coach": [{"slug": "h", "blurb": "H."}]})
    cat = {"a": {"name": "A"}, "c": {"name": "C"}, "h": {"name": "H"}}
    assert [r["slug"] for r in ds.recommended_ffs("OD, FAAO", recs_path=p, catalog=cat)] == ["c"]
    assert [r["slug"] for r in ds.recommended_ffs("Health Coach, RN", recs_path=p, catalog=cat)] == ["h"]
    assert [r["slug"] for r in ds.recommended_ffs("MD, DABCMT", recs_path=p, catalog=cat)] == ["a"]  # → default
