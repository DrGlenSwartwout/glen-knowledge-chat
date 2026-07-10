"""The scan's own recommendations, mirrored into prod so the portal can show them.

Keyed on (email, scan_id, item_code): a re-push UPDATEs, never duplicates. `section`
is carried verbatim from e4l_scan_results.section_context — the scan PDF's own
headings — so "the five infoceuticals" stays a query, not a protocol_days heuristic.
"""
import sqlite3

import pytest

from dashboard import scan_recommendations as sr

EMAIL = "caregiver@example.com"
SCAN = "1037250"
DATE = "2026-07-02"

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
    yield con
    con.close()


def test_upsert_writes_every_item(cx):
    assert sr.upsert_recommendations(cx, EMAIL, SCAN, DATE, ITEMS) == 3
    assert len(sr.for_scan(cx, EMAIL, SCAN)) == 3


def test_rows_come_back_in_rank_order(cx):
    sr.upsert_recommendations(cx, EMAIL, SCAN, DATE, list(reversed(ITEMS)))
    assert [r["item_code"] for r in sr.for_scan(cx, EMAIL, SCAN)] == ["BFA", "ED6", "ER2"]


def test_a_repush_updates_and_never_duplicates(cx):
    sr.upsert_recommendations(cx, EMAIL, SCAN, DATE, ITEMS)
    sr.upsert_recommendations(cx, EMAIL, SCAN, DATE, ITEMS)
    assert len(sr.for_scan(cx, EMAIL, SCAN)) == 3


def test_a_repush_applies_corrected_values(cx):
    sr.upsert_recommendations(cx, EMAIL, SCAN, DATE, ITEMS)
    fixed = [dict(ITEMS[0], label="Big Field Aligner (BFA)", priority_rank=1)]
    sr.upsert_recommendations(cx, EMAIL, SCAN, DATE, fixed)
    row = [r for r in sr.for_scan(cx, EMAIL, SCAN) if r["item_code"] == "BFA"][0]
    assert row["label"] == "Big Field Aligner (BFA)"


def test_infoceuticals_excludes_mihealth(cx):
    sr.upsert_recommendations(cx, EMAIL, SCAN, DATE, ITEMS)
    got = [r["item_code"] for r in sr.infoceuticals_for_scan(cx, EMAIL, SCAN)]
    assert got == ["BFA", "ED6"]


def test_bfa_is_rank_one_and_leads_the_infoceuticals(cx):
    sr.upsert_recommendations(cx, EMAIL, SCAN, DATE, ITEMS)
    first = sr.infoceuticals_for_scan(cx, EMAIL, SCAN)[0]
    assert first["item_code"] == "BFA" and first["priority_rank"] == 1


def test_email_is_normalised(cx):
    sr.upsert_recommendations(cx, "  CareGiver@Example.COM ", SCAN, DATE, ITEMS)
    assert len(sr.for_scan(cx, EMAIL, SCAN)) == 3


def test_two_scans_for_one_client_do_not_collide(cx):
    sr.upsert_recommendations(cx, EMAIL, SCAN, DATE, ITEMS)
    sr.upsert_recommendations(cx, EMAIL, "999", "2026-06-13", ITEMS[:1])
    assert len(sr.for_scan(cx, EMAIL, SCAN)) == 3
    assert len(sr.for_scan(cx, EMAIL, "999")) == 1


def test_an_item_missing_its_code_is_skipped_not_stored_blank(cx):
    n = sr.upsert_recommendations(cx, EMAIL, SCAN, DATE, [{"priority_rank": 1}])
    assert n == 0
    assert sr.for_scan(cx, EMAIL, SCAN) == []


def test_a_scan_with_no_items_writes_nothing(cx):
    assert sr.upsert_recommendations(cx, EMAIL, SCAN, DATE, []) == 0


def test_a_blank_email_writes_nothing(cx):
    assert sr.upsert_recommendations(cx, "", SCAN, DATE, ITEMS) == 0
