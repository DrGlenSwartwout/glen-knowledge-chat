# tests/test_biofield_report_present.py
from dashboard.biofield_report_present import render_present

REPORT = {
    "test_id": "a2",
    "client": {"name": "Kauilani Perdomo", "email": "k@example.com"},
    "date": "2026-06-24",
    "layers": [
        {"layer": 1, "head": "ET4", "most_affected": "ET4", "remedy": "MSM Lotion",
         "dosage": "1 application", "frequency": "daily", "timing": "morning"},
        {"layer": 2, "head": "ED7 Lungs", "most_affected": "Pancreas Cauda",
         "remedy": "Sulfur Syntropy", "dosage": "10 drops", "frequency": "3x a day", "timing": "before food"},
    ],
    "schedule": {
        "slots": ["On rising", "Breakfast", "Bedtime"],
        "entries": [
            {"name": "Sulfur Syntropy", "dosage": "10 drops", "frequency": "3x a day",
             "timing": "before food", "slots": ["On rising"], "food": "before food", "as_directed": False},
            {"name": "MSM Lotion", "dosage": "1 application", "frequency": "daily",
             "timing": "", "slots": ["Breakfast"], "food": "", "as_directed": False},
            {"name": "Reverse AGE", "dosage": "1 cap", "frequency": "as needed",
             "timing": "", "slots": [], "food": "", "as_directed": True},
        ],
    },
}

def test_full_document_with_branding_and_order():
    html = render_present(REPORT, narrative="You are healing beautifully.")
    assert html.lstrip().lower().startswith("<!doctype html")
    # branding verbatim
    assert "Accelerated Self Healing™" in html
    assert "In wellness, Dr. Glen &amp; Rae · illtowell.com" in html
    # schedule-forward: Remedy Schedule heading appears before Narrative, before Causal Chain
    i_sched = html.index("Remedy Schedule")
    i_narr = html.index("Narrative")
    i_chain = html.index("Causal Chain")
    assert i_sched < i_narr < i_chain
    # client + date in masthead
    assert "Kauilani Perdomo" in html and "2026-06-24" in html

def test_no_edit_chrome():
    html = render_present(REPORT, narrative="x")
    for forbidden in ("<button", "<textarea", "<input", "<nav", "onclick"):
        assert forbidden not in html.lower()

def test_schedule_slots_and_as_directed():
    html = render_present(REPORT, narrative="x")
    assert "On rising" in html and "Sulfur Syntropy" in html and "before food" in html
    assert "As directed" in html and "Reverse AGE" in html

def test_chain_table_and_escaping():
    rep = {**REPORT, "client": {"name": "A & B <x>", "email": ""}}
    html = render_present(rep, narrative="")
    assert "A &amp; B &lt;x&gt;" in html           # escaped
    assert "Sulfur Syntropy" in html and "ED7 Lungs" in html

def test_empty_narrative_section_omitted_or_placeholder():
    html = render_present(REPORT, narrative="")
    # narrative heading still present (section exists), no crash
    assert "Narrative" in html


def test_masthead_includes_logo():
    html = render_present(REPORT, "hello")
    assert 'class="logo"' in html and "data:image/png;base64," in html   # logo embedded upper-left


def test_terrain_banner_at_top_when_phase_present():
    rep = {**REPORT, "phase": 4, "location": "Toxicity"}
    html = render_present(rep, narrative="x")
    assert "Terrain Refresh (Phase 4)" in html
    assert "Toxicity" in html
    # the terrain reading sits at the very top: before the Remedy Schedule
    assert html.index("Terrain Refresh") < html.index("Remedy Schedule")


def test_no_terrain_banner_without_a_phase():
    # Base report (FMP-snapshot / older authored) has no BSI phase -> no banner.
    html = render_present(REPORT, narrative="x")
    assert "Phase 4" not in html and "Your Terrain" not in html


def test_location_omitted_when_blank_but_phase_shown():
    rep = {**REPORT, "phase": 2, "location": ""}
    html = render_present(rep, narrative="x")
    assert "Terrain Repair (Phase 2)" in html
    assert "Location" not in html


def test_terrain_restore_schedule_entry_tagged_with_phase():
    rep = {**REPORT, "phase": 5, "location": "Kidney",
           "schedule": {"slots": ["On rising"],
                        "entries": [{"name": "Terrain Restore", "dosage": "contains: Mimulus",
                                     "frequency": "daily", "timing": "", "slots": ["On rising"],
                                     "food": "", "as_directed": False}]}}
    html = render_present(rep, narrative="x")
    # the combined Terrain Restore bottle carries the scan's phase name
    assert "Terrain Relief" in html
    i = html.index("Terrain Restore")
    assert "Terrain Relief" in html[i:i + 80]
