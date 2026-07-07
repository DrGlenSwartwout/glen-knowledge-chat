from dashboard.biofield_report_html import render_fee_panel, render_author_html


def _state(**kw):
    base = {"email": "j@x.com", "has_email": True, "available": True, "courtesy_cents": None,
            "note": "", "standard_cents": 30000, "value_cents": 99700}
    base.update(kw)
    return base


def test_panel_shows_value_and_standard():
    html = render_fee_panel(_state())
    assert "$997" in html and "$300" in html
    assert "Standard" in html                       # no courtesy => standard applies
    assert "/author/" not in html or "fee" in html  # buttons target the fee routes


def test_panel_shows_courtesy_and_clear():
    html = render_fee_panel(_state(courtesy_cents=10000, note="special"))
    assert "$100" in html and "special" in html
    assert "Clear" in html                          # clear-to-standard offered


def test_panel_presets_present():
    html = render_fee_panel(_state())
    assert "697" in html and "100" in html and "0" in html   # preset buttons


def test_panel_no_email_disables():
    html = render_fee_panel(_state(email="", has_email=False, available=False))
    assert "add a client email" in html.lower()


def test_panel_unavailable():
    html = render_fee_panel(_state(available=False))
    assert "unavailable" in html.lower()


def test_author_html_without_fee_state_still_renders():
    rep = {"test_id": "a1", "client": {"name": "Jane", "email": "j@x.com"}, "date": "2026-06-23", "layers": []}
    html = render_author_html(rep)                   # existing callers pass no fee_state
    assert "Edit Biofield Test" in html and "feepanel" not in html


def test_author_html_injects_panel_when_state_given():
    rep = {"test_id": "a1", "client": {"name": "Jane", "email": "j@x.com"}, "date": "2026-06-23", "layers": []}
    html = render_author_html(rep, fee_state=_state())
    assert "feepanel" in html and "$300" in html


def test_panel_zero_courtesy_is_comp_not_standard():
    html = render_fee_panel(_state(courtesy_cents=0, note="comp"))
    assert "Courtesy: $0" in html          # $0 is a comp courtesy...
    assert "Standard: $" not in html       # ...not a fall-through to standard
