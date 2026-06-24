"""HTML rendering for the local Biofield Analysis viewer. Free-text FileMaker
fields (remedy/timing/names) must be HTML-escaped."""
from dashboard.biofield_report_html import render_report_html, render_list_html


def _report():
    return {
        "test_id": "10", "client": {"name": "Lewis Zardo", "email": "lz@x.com"},
        "date": "2026-06-01",
        "layers": [
            {"layer": 1, "head": "Night", "most_affected": "Night",
             "remedy": "TMG Powder", "dosage": "1 scoop", "frequency": "daily", "timing": "at night"},
            {"layer": 2, "head": "Acid", "most_affected": "Liver",
             "remedy": "Sterol Max", "dosage": "3 caps", "frequency": "daily", "timing": "with food"},
        ],
        "schedule": {
            "slots": ["On waking", "Breakfast", "Bedtime"],
            "entries": [
                {"name": "TMG Powder", "dosage": "1 scoop", "slots": ["Bedtime"],
                 "food": "", "as_directed": False},
                {"name": "Sterol Max", "dosage": "3 caps", "slots": ["Breakfast"],
                 "food": "with food", "as_directed": False},
            ],
        },
    }


def test_report_html_shows_client_layers_and_schedule():
    html = render_report_html(_report())
    assert "Lewis Zardo" in html
    assert "TMG Powder" in html and "Sterol Max" in html
    assert "Night" in html and "Acid" in html
    # schedule slot with its remedy
    assert "Bedtime" in html and "Breakfast" in html


def test_report_html_escapes_free_text():
    rep = _report()
    rep["client"]["name"] = "<script>alert(1)</script>"
    rep["layers"][0]["remedy"] = "<b>x</b>"
    html = render_report_html(rep)
    assert "<script>alert(1)</script>" not in html
    assert "&lt;script&gt;" in html
    assert "<b>x</b>" not in html


def test_report_html_depth_warning_badge():
    rep = _report()
    rep["layers"][0]["depth_status"] = "shallow"
    rep["layers"][0]["depth_need"] = "Nucleoplasm / epigenetic"
    html = render_report_html(rep)
    assert "may not reach" in html and "Nucleoplasm" in html


def test_report_html_has_notes_and_narrative_section():
    html = render_report_html(_report(), notes="kidney weak", narrative="Aloha Lewis,")
    assert "kidney weak" in html          # saved notes pre-filled
    assert "Aloha Lewis," in html         # saved narrative pre-filled
    assert "Generate narrative" in html
    assert "/test/10/notes" in html       # save-notes endpoint
    assert "/test/10/generate" in html    # generate endpoint


def test_report_html_has_video_section():
    html = render_report_html(_report(), video_script="Aloha Lewis, short version")
    assert "Aloha Lewis, short version" in html   # saved script pre-filled
    assert "Make audio" in html                   # ElevenLabs render button
    assert "/test/10/video-generate" in html      # generate-script endpoint
    assert "/test/10/audio" in html               # make-audio endpoint


def test_pages_have_console_style_header():
    html = render_list_html([])
    assert "opbar" in html and "GLEN" in html        # console-style top bar
    assert "Biofield Intake" in html
    assert "/console" in html                          # back-to-console link


def test_list_html_links_each_test():
    html = render_list_html([
        {"test_id": "10", "name": "Lewis Zardo", "email": "lz@x.com",
         "date": "2026-06-01", "layer_count": 19},
    ])
    assert "Lewis Zardo" in html
    assert "/test/10" in html
    assert "19" in html
