# tests/test_scan_recommendations_read.py  (part 1 of 2 — read helpers land in Task 3)
"""BFA is rank 1 on 161 scans and resolves to nothing.

69 of 70 infoceutical codes resolve because the catalog's storefront twin carries the
bare code as its `pinecone_title` (es1-lymph -> "ES1"). Both BFA records carry long
titles, so the bare code matches neither. A new `aliases` list fixes it without
touching `pinecone_title`, which would orphan the product's Pinecone vector.
"""
import importlib
import json
import sqlite3
import sys
from pathlib import Path

import pytest

from dashboard import scan_recommendations as sr

ROOT = Path(__file__).resolve().parent.parent
BFA_SLUG = "bfa-big-field-aligner-infoceutical"


def _app():
    if str(ROOT) not in sys.path:
        sys.path.insert(0, str(ROOT))
    try:
        return importlib.import_module("app")
    except Exception as e:
        pytest.skip(f"app not importable: {e}")


def _products():
    return json.loads((ROOT / "data" / "products.json").read_text())["products"]


def test_the_bfa_record_carries_the_bare_code_as_an_alias():
    assert _products()[BFA_SLUG]["aliases"] == ["BFA"]


def test_the_aliased_record_is_the_one_with_a_bottle_type():
    """The other BFA twin has no bottle_type; ordering it resolves the packer's
    'default' bottle, which poisons the shipping quote."""
    rec = _products()[BFA_SLUG]
    assert rec["bottle_type"] == "30ml"
    assert rec.get("description")


def test_pinecone_title_is_untouched():
    assert _products()[BFA_SLUG]["pinecone_title"] == "BFA Big Field Aligner Infoceutical"


def test_the_bare_code_bfa_now_resolves_to_a_live_product():
    app = _app()
    slug = app._resolve_remedy_slug({"name": "BFA"})
    assert slug == BFA_SLUG
    assert app._get_product(slug)


def test_the_other_infoceutical_codes_still_resolve():
    app = _app()
    for code, expected in (("ED6", "ed6-heart-driver"), ("ES7", "es7-muscle"),
                           ("ES1", "es1-lymph"), ("MB1", "mb1-brain-stem-hologram")):
        assert app._resolve_remedy_slug({"name": code}) == expected


def test_mihealth_codes_still_resolve_to_nothing():
    """ER/MR are device cycles, not products. Resolving them would be the bug."""
    app = _app()
    for code in ("ER2", "ER18", "MR4", "MR6"):
        assert not app._resolve_remedy_slug({"name": code})


def test_the_two_duplicate_title_keys_still_resolve_as_before():
    """`_TITLE_TO_SLUG` is a dict comprehension: on a duplicate key the LAST product wins.
    Two duplicate title keys exist. Rebuilding that dict with setdefault would flip them."""
    app = _app()
    assert app._resolve_remedy_slug({"name": "Brain Boost Nootropic"})
    assert app._resolve_remedy_slug({"name": "Forgiveness Flower Essence in Terrain Restore"})


def test_an_alias_never_shadows_a_real_product_name():
    """A collision would silently hand one product's code to another."""
    p = _products()
    names = {(r.get("pinecone_title") or r.get("name") or "").strip().lower() for r in p.values()}
    for slug, rec in p.items():
        for a in rec.get("aliases") or []:
            assert a.strip().lower() not in names, f"{slug} alias {a!r} collides with a product title"


# tests/test_scan_recommendations_read.py  (part 2 of 2 — store read helpers + console
# read path land in Task 3)

EMAIL = "caregiver@example.com"
HDRS = {"X-Console-Key": "testkey"}
ITEMS = [
    {"item_code": "BFA", "priority_rank": 1, "protocol_days": 15,
     "section": "Infoceuticals", "category": "BFA", "label": "Big Field Aligner"},
    {"item_code": "ED6", "priority_rank": 2, "protocol_days": 15,
     "section": "Infoceuticals", "category": "ED", "label": "Heart"},
    {"item_code": "ER2", "priority_rank": 3, "protocol_days": 2,
     "section": "miHealth Functions", "category": "ER", "label": "Large Intestine"},
]


@pytest.fixture()
def cx():
    con = sqlite3.connect(":memory:")
    con.row_factory = sqlite3.Row
    sr.init_table(con)
    sr.replace_scan(con, EMAIL, "10", "2026-07-02", ITEMS)
    sr.replace_scan(con, EMAIL, "20", "2026-06-13", ITEMS[:1])
    yield con
    con.close()


def test_scan_dates_are_newest_first(cx):
    assert sr.scan_dates_for(cx, EMAIL) == ["2026-07-02", "2026-06-13"]


def test_for_scan_date_returns_that_scan_in_rank_order(cx):
    got = [r["item_code"] for r in sr.for_scan_date(cx, EMAIL, "2026-07-02")]
    assert got == ["BFA", "ED6", "ER2"]


def test_for_an_unknown_date_returns_nothing(cx):
    assert sr.for_scan_date(cx, EMAIL, "1999-01-01") == []


def test_for_an_unknown_email_returns_nothing(cx):
    assert sr.scan_dates_for(cx, "stranger@example.com") == []


def test_split_by_section_preserves_rank_order(cx):
    info, mih = sr.split_by_section(sr.for_scan_date(cx, EMAIL, "2026-07-02"))
    assert [r["item_code"] for r in info] == ["BFA", "ED6"]
    assert [r["item_code"] for r in mih] == ["ER2"]


def test_split_by_section_on_an_empty_list(cx):
    assert sr.split_by_section([]) == ([], [])


def _app_for_client():
    if str(ROOT) not in sys.path:
        sys.path.insert(0, str(ROOT))
    return importlib.import_module("app")


@pytest.fixture()
def client(tmp_db, monkeypatch):
    app_mod = _app_for_client()
    monkeypatch.setattr(app_mod, "LOG_DB", tmp_db)
    monkeypatch.setattr(app_mod, "CONSOLE_SECRET", "testkey")
    batch = [{"email": EMAIL, "scans": [
        {"scan_id": "10", "scan_date": "2026-07-02", "items": ITEMS},
        {"scan_id": "20", "scan_date": "2026-06-13", "items": ITEMS[:1]},
    ]}]
    test_client = app_mod.app.test_client()
    test_client.post("/api/console/scan-recommendations/sync", headers=HDRS, json={"batch": batch})
    return test_client


def test_console_read_requires_the_key(client):
    assert client.get("/api/console/scan-recommendations").status_code == 401


def test_console_read_without_an_email_returns_corpus_totals(client, tmp_db):
    body = client.get("/api/console/scan-recommendations", headers=HDRS).get_json()
    assert body["ok"] is True
    assert set(body) == {"ok", "total_rows", "clients", "scans"}   # no client data leaked
    assert body["total_rows"] == 4 and body["clients"] == 1 and body["scans"] == 2


def test_console_read_with_an_email_adds_that_clients_scan(client):
    body = client.get(f"/api/console/scan-recommendations?email={EMAIL}", headers=HDRS).get_json()
    assert body["scan_date"] == "2026-07-02"
    assert [i["item_code"] for i in body["infoceuticals"]] == ["BFA", "ED6"]
    assert [m["item_code"] for m in body["mihealth"]] == ["ER2"]
