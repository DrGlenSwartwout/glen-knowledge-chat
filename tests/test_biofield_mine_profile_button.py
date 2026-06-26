from dashboard.biofield_report_html import render_author_html


def _html():
    rep = {"test_id": "a7", "client": {"name": "J", "email": "j@x.com"}, "date": "",
           "layers": [], "schedule": []}
    return render_author_html(rep, [], "")


def test_mine_profile_button_and_handler():
    h = _html()
    assert "Mine profile" in h
    assert "function mineProfile" in h
    assert "/author/a7/mine-profile" in h
    assert "loadStress()" in h
