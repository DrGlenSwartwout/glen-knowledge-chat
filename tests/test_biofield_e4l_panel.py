"""The E4L reference panel shown on the authoring page: fresh / stale / none states,
ranked findings, and HTML-escaping of scan free-text."""
from dashboard.biofield_report_html import render_e4l_panel, render_author_html


def _ctx(**kw):
    base = {"status": "none", "found": False, "scan_id": None, "scan_date": None,
            "days_ago": None, "fresh": False, "window_days": 14, "findings": [],
            "message": "No E4L scan on file"}
    base.update(kw)
    return base


def test_fresh_panel_shows_days_ago_and_findings():
    ctx = _ctx(status="fresh", found=True, scan_id=900, scan_date="2026-06-20",
               days_ago=4, fresh=True, message="Recent E4L scan · 4 days ago",
               findings=[{"rank": 1, "code": "LV3", "name": "Liver meridian",
                          "description": "detox and anger"}])
    html = render_e4l_panel(ctx)
    assert "Recent E4L scan" in html and "4 days ago" in html
    assert "LV3" in html and "Liver meridian" in html and "detox and anger" in html
    assert "2026-06-20" in html


def test_stale_panel_warns_but_still_lists_findings():
    ctx = _ctx(status="stale", found=True, scan_id=800, scan_date="2026-05-17",
               days_ago=38, fresh=False,
               message="No fresh voice scan — last scan 38 days ago (stale)",
               findings=[{"rank": 1, "code": "LV3", "name": "Liver", "description": ""}])
    html = render_e4l_panel(ctx)
    assert "No fresh voice scan" in html and "38 days ago" in html
    assert "stale" in html.lower()
    assert "LV3" in html  # still shown


def test_none_panel_says_no_scan():
    html = render_e4l_panel(_ctx())
    assert "No E4L scan on file" in html
    assert "LV3" not in html


def test_panel_escapes_scan_free_text():
    ctx = _ctx(status="fresh", found=True, days_ago=1, fresh=True,
               message="Recent E4L scan · 1 day ago", scan_date="2026-06-23",
               findings=[{"rank": 1, "code": "X", "name": "<script>x</script>",
                          "description": "<b>raw</b>"}])
    html = render_e4l_panel(ctx)
    assert "<script>x</script>" not in html
    assert "&lt;script&gt;" in html


def test_author_page_embeds_e4l_panel_placeholder_and_loader():
    rep = {"test_id": "a1", "client": {"name": "Jane", "email": "jane@x.com"},
           "date": "2026-06-23", "layers": [], "schedule": {"slots": [], "entries": []}}
    html = render_author_html(rep)
    assert "e4lpanel" in html              # placeholder div the JS fills
    assert "/author/a1/e4l" in html        # the loader endpoint
