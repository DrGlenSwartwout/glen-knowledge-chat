from dashboard.biofield_stress import layer_rids, build_assign_prompt, parse_assignments
from dashboard.biofield_report_html import render_stress_panel


def test_layer_rids():
    layers = [{"layer": 1, "rid": 11}, {"layer": 1, "rid": 12}, {"layer": 2, "rid": 21}]
    assert sorted(layer_rids(layers, 1)) == [11, 12]
    assert layer_rids(layers, 2) == [21]
    assert layer_rids(layers, 9) == []


def test_build_prompt_lists_stresses_and_layers():
    p = build_assign_prompt([{"id": 94, "code": "ED6", "label": "Heart Driver"}],
                            [{"layer": 2, "head": "Cardiovascular", "remedy": "ED6 Heart Driver"}])
    assert "id=94" in p["user"] and "Layer 2" in p["user"] and "Cardiovascular" in p["user"]
    assert "JSON" in p["system"]


def test_parse_assignments_keeps_valid_layers_only():
    resp = {"assignments": [{"id": 94, "layer": 2}, {"id": 95, "layer": 9}, {"id": "x", "layer": 1}]}
    assert parse_assignments(resp, [1, 2, 3]) == {94: 2}   # 9 not a real layer; "x" not an int


def test_panel_has_assign_and_assign_all_buttons():
    data = {"by_layer": [{"layer": 1, "head": "Muscle", "remedy": "ED9", "stresses": []}],
            "unassigned": [{"id": 94, "code": "ED6", "label": "Heart Driver", "balance": "required"}]}
    html = render_stress_panel(data)
    assert "assignStress(94)" in html            # per-stress Assign button
    assert "assignAllStresses()" in html         # Assign-all on the Unassigned line
