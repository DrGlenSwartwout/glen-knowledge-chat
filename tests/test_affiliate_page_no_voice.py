from pathlib import Path


HTML = (Path(__file__).resolve().parent.parent / "static" / "affiliate.html").read_text()


def test_ambassador_application_is_direct_form_without_voice_embed():
    assert 'action="/affiliate/apply-form"' in HTML
    assert 'src="/embed"' not in HTML
    assert "MediaRecorder" not in HTML
