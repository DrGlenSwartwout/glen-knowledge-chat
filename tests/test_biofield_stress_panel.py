from dashboard.biofield_report_html import render_stress_panel, render_author_html


def test_panel_shows_active_and_balanced_with_tags():
    data = {"active": [{"id": 1, "code": "MR2", "label": "Calm", "source": "scan",
                        "balance": "optional", "balanced": False, "balanced_by": ""}],
            "balanced": [{"id": 2, "code": "ED1", "label": "Membrane", "source": "scan",
                          "balance": "required", "balanced": True, "balanced_by": "neuro magnesium"}]}
    html = render_stress_panel(data)
    assert "Calm" in html and "Membrane" in html
    assert "neuro magnesium" in html            # balanced_by shown
    assert "optional" in html and "required" in html
    assert "balanceStress(1" in html            # toggle on the active item


def test_author_page_has_stress_panel_and_loader():
    rep = {"test_id": "a7", "client": {"name": "J", "email": "j@x.com"}, "date": "",
           "layers": [], "schedule": []}
    html = render_author_html(rep, [], "")
    assert "id=stresspanel" in html and "loadStress()" in html
    assert "function balanceStress" in html


def test_author_page_marks_unbalanced_scan_group():
    rep = {"test_id": "a7", "client": {"name": "J", "email": "j@x.com"}, "date": "",
           "layers": [
               {"layer": 1, "head": "Live", "most_affected": "", "remedy": "R1", "rid": 1,
                "confirmed": 1, "origin": "live", "zone": "top",
                "stress_depth": None, "remedy_depth": None, "depth_status": None, "depth_need": None},
               {"layer": 2, "head": "Scan", "most_affected": "", "remedy": "R2", "rid": 2,
                "confirmed": 0, "origin": "scan", "zone": "bottom",
                "stress_depth": None, "remedy_depth": None, "depth_status": None, "depth_need": None}],
           "schedule": []}
    html = render_author_html(rep, [], "")
    # Unconfirmed scan rows are flagged by the unconf highlight on their remedy line
    # (the two-zone divider was replaced by the per-layer card layout).
    assert "rline unconf" in html
    assert "confirmRow('2')" in html      # the scan row (rid 2) offers confirm
