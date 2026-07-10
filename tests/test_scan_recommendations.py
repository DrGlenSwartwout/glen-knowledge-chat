"""The scan's own recommendations, mirrored into prod so the portal can show them.

Keyed on (email, scan_id, priority_rank): a scan's rows are replaced atomically
(delete + insert in one transaction), never upserted per item. `priority_rank` is
unique per scan in the real source data; `item_code` is NOT — a two-column PDF
layout can flatten to the same item_code appearing at two different ranks in one
scan, and both rows must survive. `section` is carried verbatim from
e4l_scan_results.section_context — the scan PDF's own headings — so "the five
infoceuticals" stays a query, not a protocol_days heuristic.
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


def test_replace_scan_writes_every_item(cx):
    assert sr.replace_scan(cx, EMAIL, SCAN, DATE, ITEMS) == 3
    assert len(sr.for_scan(cx, EMAIL, SCAN)) == 3


def test_rows_come_back_in_rank_order(cx):
    sr.replace_scan(cx, EMAIL, SCAN, DATE, list(reversed(ITEMS)))
    assert [r["item_code"] for r in sr.for_scan(cx, EMAIL, SCAN)] == ["BFA", "ED6", "ER2"]


def test_a_replace_applies_corrected_values(cx):
    sr.replace_scan(cx, EMAIL, SCAN, DATE, ITEMS)
    fixed = [dict(ITEMS[0], label="Big Field Aligner (BFA)", priority_rank=1)]
    sr.replace_scan(cx, EMAIL, SCAN, DATE, fixed)
    row = [r for r in sr.for_scan(cx, EMAIL, SCAN) if r["item_code"] == "BFA"][0]
    assert row["label"] == "Big Field Aligner (BFA)"


def test_infoceuticals_excludes_mihealth(cx):
    sr.replace_scan(cx, EMAIL, SCAN, DATE, ITEMS)
    got = [r["item_code"] for r in sr.infoceuticals_for_scan(cx, EMAIL, SCAN)]
    assert got == ["BFA", "ED6"]


def test_bfa_is_rank_one_and_leads_the_infoceuticals(cx):
    sr.replace_scan(cx, EMAIL, SCAN, DATE, ITEMS)
    first = sr.infoceuticals_for_scan(cx, EMAIL, SCAN)[0]
    assert first["item_code"] == "BFA" and first["priority_rank"] == 1


def test_email_is_normalised(cx):
    sr.replace_scan(cx, "  CareGiver@Example.COM ", SCAN, DATE, ITEMS)
    assert len(sr.for_scan(cx, EMAIL, SCAN)) == 3


def test_two_scans_for_one_client_do_not_collide(cx):
    sr.replace_scan(cx, EMAIL, SCAN, DATE, ITEMS)
    sr.replace_scan(cx, EMAIL, "999", "2026-06-13", ITEMS[:1])
    assert len(sr.for_scan(cx, EMAIL, SCAN)) == 3
    assert len(sr.for_scan(cx, EMAIL, "999")) == 1


def test_an_item_missing_its_code_is_skipped_not_stored_blank(cx):
    n = sr.replace_scan(cx, EMAIL, SCAN, DATE, [{"priority_rank": 1}])
    assert n == 0
    assert sr.for_scan(cx, EMAIL, SCAN) == []


def test_a_scan_with_no_items_writes_nothing(cx):
    assert sr.replace_scan(cx, EMAIL, SCAN, DATE, []) == 0


def test_a_blank_email_writes_nothing(cx):
    assert sr.replace_scan(cx, "", SCAN, DATE, ITEMS) == 0


def test_duplicate_item_codes_in_one_scan_are_both_stored(cx):
    dupes = [
        {"item_code": "ER1", "priority_rank": 6, "protocol_days": 2,
         "section": "miHealth Functions", "category": "ER", "label": "Stomach"},
        {"item_code": "ER1", "priority_rank": 7, "protocol_days": 2,
         "section": "miHealth Functions", "category": "ER", "label": "Stomach"},
    ]
    assert sr.replace_scan(cx, EMAIL, SCAN, DATE, dupes) == 2
    rows = sr.for_scan(cx, EMAIL, SCAN)
    assert len(rows) == 2
    assert [r["item_code"] for r in rows] == ["ER1", "ER1"]
    assert [r["priority_rank"] for r in rows] == [6, 7]


def test_an_empty_item_list_does_not_delete_existing_rows(cx):
    sr.replace_scan(cx, EMAIL, SCAN, DATE, ITEMS)
    assert sr.replace_scan(cx, EMAIL, SCAN, DATE, []) == 0
    assert len(sr.for_scan(cx, EMAIL, SCAN)) == 3


def test_a_list_of_only_invalid_items_does_not_delete_existing_rows(cx):
    sr.replace_scan(cx, EMAIL, SCAN, DATE, ITEMS)
    assert sr.replace_scan(cx, EMAIL, SCAN, DATE, [{"priority_rank": 1}]) == 0
    assert len(sr.for_scan(cx, EMAIL, SCAN)) == 3


def test_a_shrunken_rescan_leaves_no_stale_rows(cx):
    five = ITEMS + [
        {"item_code": "ER36", "priority_rank": 4, "protocol_days": 2,
         "section": "miHealth Functions", "category": "ER", "label": "Kidney"},
        {"item_code": "ER10", "priority_rank": 5, "protocol_days": 2,
         "section": "miHealth Functions", "category": "ER", "label": "Liver"},
    ]
    sr.replace_scan(cx, EMAIL, SCAN, DATE, five)
    assert len(sr.for_scan(cx, EMAIL, SCAN)) == 5
    two = ITEMS[:2]
    assert sr.replace_scan(cx, EMAIL, SCAN, DATE, two) == 2
    assert len(sr.for_scan(cx, EMAIL, SCAN)) == 2


def test_replace_is_idempotent(cx):
    sr.replace_scan(cx, EMAIL, SCAN, DATE, ITEMS)
    sr.replace_scan(cx, EMAIL, SCAN, DATE, ITEMS)
    assert len(sr.for_scan(cx, EMAIL, SCAN)) == 3


def test_the_returned_count_equals_the_rows_actually_stored(cx):
    payload = ITEMS + [{"priority_rank": 99}]
    n = sr.replace_scan(cx, EMAIL, SCAN, DATE, payload)
    assert n == len(sr.for_scan(cx, EMAIL, SCAN))
