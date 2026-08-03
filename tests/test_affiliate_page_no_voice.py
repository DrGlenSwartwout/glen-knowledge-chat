from pathlib import Path


HTML = (Path(__file__).resolve().parent.parent / "static" / "affiliate.html").read_text()
BEGIN_PATH_HTML = (Path(__file__).resolve().parent.parent / "static" / "begin-path.html").read_text()
AFFILIATE_PORTAL_HTML = (Path(__file__).resolve().parent.parent / "static" / "affiliate-portal.html").read_text()
PORTAL_PROGRAM_HTML = (Path(__file__).resolve().parent.parent / "static" / "portal-program.html").read_text()
BEGIN_HTML = (Path(__file__).resolve().parent.parent / "static" / "begin.html").read_text()


def test_ambassador_application_is_direct_form_without_voice_embed():
    assert 'action="/affiliate/apply-form"' in HTML
    assert 'src="/embed"' not in HTML
    assert "MediaRecorder" not in HTML


def test_ambassador_entry_points_open_application_in_new_tab_without_voice():
    assert 'id="p-affiliate" href="/affiliate" target="_blank"' in BEGIN_PATH_HTML
    assert 'src="/embed"' not in BEGIN_PATH_HTML
    assert '/widget.js' not in BEGIN_PATH_HTML
    assert 'href="/affiliate" target="_blank"' in AFFILIATE_PORTAL_HTML
    assert 'src="/embed"' not in AFFILIATE_PORTAL_HTML
    assert 'target="_blank" rel="noopener noreferrer">\' + a.referral_url' in PORTAL_PROGRAM_HTML
    assert 'target="_blank" rel="noopener noreferrer">Apply</a>' in PORTAL_PROGRAM_HTML
    assert "href:'/affiliate'" in BEGIN_HTML
    assert "s.key === 'give'" in BEGIN_HTML
