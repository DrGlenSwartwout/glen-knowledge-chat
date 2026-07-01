"""Per-layer grouping of AI-created stresses in Biofield Intake.

`list_stresses` returns a `by_layer` view: for each causal-chain layer, every
stress whose code is covered by that layer's remedy PLUS the layer's own head
stress. A stress covered by several layers' remedies appears under EACH (chosen
behavior). Stresses on no layer fall into `unassigned`. `render_stress_panel`
renders that grouping.
"""
import sqlite3

from dashboard import biofield_stress as st
from dashboard.biofield_report_html import render_stress_panel


def _seed(cx, coverage):
    findings = [{"code": "ED1", "name": "Membrane"},
                {"code": "MR2", "name": "Calm"},
                {"code": "XX9", "name": "Loose"}]
    st.seed_from_scan(cx, "5", findings, coverage)


def test_by_layer_groups_covered_and_head_stresses():
    cx = sqlite3.connect(":memory:")
    _seed(cx, {"Neuro Magnesium": ["ED1"], "Cistus": ["MR2"]})
    chain = [{"layer": 1, "head": "Membrane", "remedy": "Neuro Magnesium"},
             {"layer": 2, "head": "Calm", "remedy": "Cistus"}]
    data = st.list_stresses(cx, "5", chain)
    by = {L["layer"]: {s["code"] for s in L["stresses"]} for L in data["by_layer"]}
    assert by == {1: {"ED1"}, 2: {"MR2"}}
    # XX9 is covered by no layer's remedy -> unassigned
    assert {s["code"] for s in data["unassigned"]} == {"XX9"}


def test_by_layer_shows_stress_under_every_covering_layer():
    cx = sqlite3.connect(":memory:")
    # ED1 is covered by BOTH layers' remedies -> appears under both
    _seed(cx, {"Neuro Magnesium": ["ED1"], "Cistus": ["ED1", "MR2"]})
    chain = [{"layer": 1, "head": "Head1", "remedy": "Neuro Magnesium"},
             {"layer": 2, "head": "Head2", "remedy": "Cistus"}]
    data = st.list_stresses(cx, "5", chain)
    by = {L["layer"]: {s["code"] for s in L["stresses"]} for L in data["by_layer"]}
    assert "ED1" in by[1] and "ED1" in by[2]
    assert by[2] == {"ED1", "MR2"}


def test_by_layer_head_stress_grouped_without_coverage():
    cx = sqlite3.connect(":memory:")
    _seed(cx, {})  # no remedy coverage at all
    chain = [{"layer": 1, "head": "Membrane", "remedy": ""}]
    data = st.list_stresses(cx, "5", chain)
    by = {L["layer"]: {s["code"] for s in L["stresses"]} for L in data["by_layer"]}
    # the layer's head ("Membrane") matches the ED1 stress label by name
    assert by[1] == {"ED1"}
    # legacy keys still present for backward compatibility
    assert "active" in data and "balanced" in data


def test_render_panel_shows_layers_and_unassigned():
    data = {"by_layer": [{"layer": 1, "head": "Membrane", "remedy": "Neuro Magnesium",
                          "stresses": [{"id": 2, "code": "ED1", "label": "Membrane",
                                        "source": "scan", "balance": "required",
                                        "balanced": True, "balanced_by": "neuro magnesium"}]}],
            "unassigned": [{"id": 1, "code": "MR2", "label": "Calm", "source": "scan",
                            "balance": "optional", "balanced": False, "balanced_by": ""}]}
    html = render_stress_panel(data)
    assert "Layer 1" in html
    assert "Membrane" in html and "Neuro Magnesium" in html
    assert "Calm" in html and "Unassigned" in html
    assert "balanceStress(1" in html          # unassigned item still toggleable
    assert "balanceStress(2" in html          # layer item still toggleable
