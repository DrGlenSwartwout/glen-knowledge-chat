# tests/test_topic_pages_store.py
import sqlite3
import sys
from pathlib import Path

import pytest


def _repo():
    return Path(__file__).resolve().parent.parent


def _mod():
    r = str(_repo())
    if r not in sys.path:
        sys.path.insert(0, r)
    try:
        from dashboard import topic_pages
        return topic_pages
    except Exception as e:  # noqa: BLE001
        pytest.skip(f"topic_pages not importable: {e}")


def _cx(tp):
    cx = sqlite3.connect(":memory:")
    tp.init_table(cx)
    return cx


def test_upsert_and_get_roundtrip():
    tp = _mod()
    cx = _cx(tp)
    tp.upsert_section(cx, "low-energy", "overview", "People often notice tiredness.")
    tp.set_kind(cx, "low-energy", "symptom")
    tp.set_name(cx, "low-energy", "Low Energy")
    page = tp.get_page(cx, "low-energy")
    assert page["kind"] == "symptom"
    assert page["name"] == "Low Energy"
    assert page["state"] == "draft"
    assert page["content"]["overview"] == "People often notice tiredness."


def test_set_links_compliance_seo_roundtrip():
    tp = _mod()
    cx = _cx(tp)
    tp.upsert_section(cx, "methylation", "overview", "x")
    tp.set_links(cx, "methylation", {"ingredients": [{"slug": "folate", "name": "Folate"}],
                                     "products": [], "topics": []})
    tp.set_compliance(cx, "methylation", {"passed": True, "flags": [], "scanned_at": "t", "model": "m"})
    tp.set_seo(cx, "methylation", {"title": "Methylation", "meta_description": "About methylation.",
                                   "jsonld": {}})
    page = tp.get_page(cx, "methylation")
    assert page["links"]["ingredients"][0]["slug"] == "folate"
    assert page["compliance"]["passed"] is True
    assert page["seo"]["title"] == "Methylation"


def test_approve_stamps_approved_fields():
    tp = _mod()
    cx = _cx(tp)
    tp.upsert_section(cx, "detox", "overview", "x")
    tp.set_state(cx, "detox", "approved", by="glen")
    page = tp.get_page(cx, "detox")
    assert page["state"] == "approved"
    assert page["approved_by"] == "glen"
    assert page["approved_at"]


def test_get_missing_returns_none():
    tp = _mod()
    cx = _cx(tp)
    assert tp.get_page(cx, "nope") is None


def test_list_pages_reports_compliance_passed():
    tp = _mod()
    cx = _cx(tp)
    tp.upsert_section(cx, "detox", "overview", "x")
    tp.set_compliance(cx, "detox", {"passed": False, "flags": [{"phrase": "cures", "reason": "claim"}],
                                    "scanned_at": "t", "model": "m"})
    rows = tp.list_pages(cx)
    row = [r for r in rows if r["slug"] == "detox"][0]
    assert row["compliance_passed"] is False
    assert "overview" in row["sections"]
