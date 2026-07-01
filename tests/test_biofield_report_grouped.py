"""Read-only report surfaces group the causal chain by layer (Head/Tail span
their remedies) and use the Head/Tail naming, matching the editor cards."""
from dashboard.biofield_report_html import render_chain_table, render_report_html
from dashboard.biofield_report_present import render_present


_LAYERS = [
    {"layer": 1, "head": "Lymphatic", "most_affected": "Groin", "remedy": "Lymph Flow",
     "dosage": "1", "frequency": "d", "timing": "am"},
    {"layer": 2, "head": "Neural", "most_affected": "CNS", "remedy": "Nerve Pulse",
     "dosage": "1 scoop", "frequency": "2x", "timing": "am"},
    {"layer": 3, "head": "Neural", "most_affected": "CNS", "remedy": "Mag Glycinate",
     "dosage": "200mg", "frequency": "hs", "timing": ""},
]


def test_chain_table_groups_and_renames():
    html = render_chain_table(_LAYERS)
    assert "<th>Head</th>" in html and "<th>Tail</th>" in html   # renamed
    assert "Head of Chain" not in html and "Most Affected" not in html
    # the two Neural remedies share one layer -> Head cell spans both rows (rowspan=2)
    assert "rowspan=2" in html
    assert html.count("Neural") == 1                             # head printed once for the layer
    assert "Nerve Pulse" in html and "Mag Glycinate" in html     # both remedies listed


def test_internal_viewer_uses_grouped_table():
    rep = {"test_id": "a1", "client": {"name": "J", "email": "j@x.com"}, "date": "",
           "layers": _LAYERS, "schedule": {"slots": [], "entries": []}}
    html = render_report_html(rep)
    assert "<th>Head</th>" in html and "<th>Tail</th>" in html
    assert "Head of Chain" not in html
    assert "rowspan=2" in html


def test_clean_report_uses_grouped_table():
    rep = {"test_id": "a1", "client": {"name": "J", "email": "j@x.com"}, "date": "",
           "layers": _LAYERS, "schedule": {"slots": [], "entries": []}}
    html = render_present(rep, narrative="")
    assert "<th>Head</th>" in html and "<th>Tail</th>" in html
    assert "Most Affected" not in html
    assert "rowspan=2" in html and "Mag Glycinate" in html


def test_depth_badge_only_on_viewer():
    layers = [{"layer": 1, "head": "H", "most_affected": "M", "remedy": "R",
               "dosage": "", "frequency": "", "timing": "",
               "depth_status": "shallow", "depth_need": "the nucleus"}]
    assert "may not reach" in render_chain_table(layers, with_depth_badge=True)
    assert "may not reach" not in render_chain_table(layers)          # client report stays clean
