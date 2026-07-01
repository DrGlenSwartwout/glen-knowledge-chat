import re
import pathlib


def test_tos_version_is_combined_rm_e4l():
    import app
    assert app.BEGIN_TOS_VERSION == "rm-e4l-tc-2026-07-01"


def test_portal_gate_links_both_rm_and_e4l_terms():
    html = pathlib.Path("static/client-portal.html").read_text()
    # gate must name E4L and link the E4L terms alongside the RM terms
    assert "illtowell.com/terms" in html
    assert re.search(r"E4L|Energy For Life", html)
    assert re.search(r"E4L_TOS_URL|portal\.e4l\.com|truly\.vip/E4L", html)
