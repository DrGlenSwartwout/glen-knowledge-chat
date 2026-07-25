from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def test_client_portal_has_member_review_and_upgrade_actions():
    html = (ROOT / "static" / "client-portal.html").read_text()
    assert "Request professional review" in html
    assert "Upgrade and request review" in html
    assert "eye-vision-report/request-review" in html


def test_console_queue_supports_edit_approve_and_decline():
    html = (ROOT / "static" / "console-eye-vision-reviews.html").read_text()
    assert "Editable report JSON" in html
    assert "Approve" in html
    assert "Save draft" in html
    assert "Decline" in html
    approvals = (ROOT / "static" / "console-approvals.html").read_text()
    assert "/console/eye-vision-reviews" in approvals
