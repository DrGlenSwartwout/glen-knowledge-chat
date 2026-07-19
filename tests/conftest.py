"""Shared pytest fixtures for the deploy-chat test suite."""

import smtplib

import pytest


@pytest.fixture
def tmp_db(tmp_path):
    """Path to an empty sqlite db file inside tmp_path. The file does not
    yet exist on disk; the test seeds whatever schema it needs."""
    return str(tmp_path / "chat_log.db")


class _BlockedSMTP:
    """Stand-in for smtplib.SMTP that refuses to connect."""

    def __init__(self, *args, **kwargs):
        raise RuntimeError(
            "Outbound SMTP is blocked during tests. A test reached a real mail "
            "send — mock the send at the call site (or patch app.smtplib.SMTP) "
            "instead of letting it dial out."
        )


@pytest.fixture(autouse=True)
def block_outbound_smtp(monkeypatch):
    """Hard-stop real email from the test suite.

    `dashboard.inbox.send_email` is guarded on PYTEST_CURRENT_TEST (#833), but
    that covers only the Gmail transport. app.py has nine `smtplib.SMTP(...)`
    call sites and only three carry that guard, so a full-suite run could still
    dial out and mail real clients — which is why the suite has never been safe
    to run whole, and therefore why this repo has no CI.

    Blocking at the transport makes the guarantee structural rather than
    per-call-site: any test that reaches a real send fails loudly with the
    message above instead of quietly emailing someone. Tests that patch
    app.smtplib.SMTP themselves still win — their patch is applied after this
    autouse fixture.
    """
    monkeypatch.setattr(smtplib, "SMTP", _BlockedSMTP)
    monkeypatch.setattr(smtplib, "SMTP_SSL", _BlockedSMTP)
