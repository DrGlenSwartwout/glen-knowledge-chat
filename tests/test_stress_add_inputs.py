from dashboard.biofield_report_html import render_stress_panel


def _by_layer_data():
    return {"by_layer": [{"layer": 2, "head": "Liver", "remedy": "Liver Support",
                          "remedies": ["Liver Support"], "stresses": []}],
            "unassigned": []}


def test_per_layer_add_input_present_with_layer_number():
    html = render_stress_panel(_by_layer_data())
    assert "add balanced stress" in html
    assert "addStress(this.value,2)" in html


def test_active_add_input_present():
    html = render_stress_panel(_by_layer_data())
    assert "add active stress" in html
    assert "addStress(this.value,null)" in html
