"""Regression: the shared Gmail transport must never send real email under pytest.

A full-suite run once flooded the real inbox with test-fixture "First Continuous
Care signup" alerts because integration tests reach dashboard.inbox.send_email
(via _notify_first_cc_signup) without mocking it. The transport now short-circuits
when PYTEST_CURRENT_TEST is set, so no test can send real email.
"""
import os
from dashboard import inbox


def test_send_email_noops_under_pytest():
    # pytest sets PYTEST_CURRENT_TEST for every running test; prod never does.
    assert os.environ.get("PYTEST_CURRENT_TEST"), "PYTEST_CURRENT_TEST should be set under pytest"
    res = inbox.send_email("nobody@example.com", "subject", "body")
    assert res == {"skipped": "pytest"}, f"expected pytest no-op, got {res!r}"


def test_send_email_html_path_also_noops():
    res = inbox.send_email("nobody@example.com", "subject", "body", html="<p>body</p>")
    assert res == {"skipped": "pytest"}
