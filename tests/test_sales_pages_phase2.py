import sqlite3
from dashboard import sales_pages as sp
from dashboard import sales_copy as sc

def _cx():
    return sqlite3.connect(":memory:")

def test_upsert_then_get_section_roundtrip():
    cx = _cx()
    assert sp.get_section(cx, "longevity", "intro") is None
    sp.upsert_section(cx, "longevity", "intro", "Hello world.", model="m1")
    assert sp.get_section(cx, "longevity", "intro") == "Hello world."

def test_upsert_accretes_sections_in_one_row():
    cx = _cx()
    sp.upsert_section(cx, "energy", "intro", "A.")
    sp.upsert_section(cx, "energy", "description", "B.")
    page = sp.get_page(cx, "energy")
    assert page["content"] == {"intro": "A.", "description": "B."}
    assert page["state"] == "draft"

def test_get_page_missing_returns_none():
    assert sp.get_page(_cx(), "nope") is None

def test_prompt_includes_compliance_and_no_disease_claim():
    system, user = sc.build_section_prompt("intro", {"name": "Longevity", "ingredients": []})
    assert "treat" in system.lower() and "cure" in system.lower() and "prevent" in system.lower()
    assert "supports" in system.lower() or "structure/function" in system.lower()

def test_prompt_grounds_in_product_name_and_ingredients():
    prod = {"name": "Longevity", "ingredients": [{"name": "Resveratrol", "dose": "200 mg"}, "Quercetin"]}
    system, user = sc.build_section_prompt("research", prod)
    assert "Longevity" in user
    assert "Resveratrol" in user and "200 mg" in user and "Quercetin" in user

def test_narrative_sections_are_exactly_three():
    assert sc.NARRATIVE_SECTIONS == ("intro", "description", "research")
