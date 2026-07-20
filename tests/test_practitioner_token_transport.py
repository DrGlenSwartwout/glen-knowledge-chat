"""_practitioner_session_pid must resolve the session token from all three
transports the front end actually uses: the ?token= query param (portal pages),
the JSON body token (some POSTs), and the X-Practitioner-Token header
(the settings page, static/practitioner-settings.html:221-225).

The header transport was unsupported, which meant /practitioner/settings could
not save anything — every branding/pricing/show_contact POST 401'd. The existing
route tests missed it because they monkeypatch _practitioner_session_pid away
entirely; these do NOT patch it — they exercise the real resolver against a
stubbed practitioner_id_from_session, so a transport regression fails here.
"""

import os
import pytest

if not os.environ.get("PINECONE_API_KEY"):
    pytest.skip("needs doppler env for import app", allow_module_level=True)

import app as appmod


@pytest.fixture
def resolver(monkeypatch):
    """Exercise the REAL _practitioner_session_pid, with only the underlying
    session-token lookup stubbed: token 'good-token' -> 'pid-xyz', else None."""
    monkeypatch.setattr(
        appmod._pp, "practitioner_id_from_session",
        lambda tok, **kw: "pid-xyz" if tok == "good-token" else None,
    )
    appmod.app.config["TESTING"] = True
    return appmod.app


def _pid_under_request(app, **request_kwargs):
    with app.test_request_context(**request_kwargs):
        return appmod._practitioner_session_pid()


def test_token_from_query_param(resolver):
    assert _pid_under_request(resolver, path="/x?token=good-token") == "pid-xyz"


def test_token_from_json_body(resolver):
    assert _pid_under_request(
        resolver, path="/x", method="POST", json={"token": "good-token"}
    ) == "pid-xyz"


def test_token_from_header(resolver):
    """The transport the settings page uses — was unsupported."""
    assert _pid_under_request(
        resolver, path="/x", headers={"X-Practitioner-Token": "good-token"}
    ) == "pid-xyz"


def test_no_token_returns_none(resolver):
    assert _pid_under_request(resolver, path="/x") is None


def test_bad_header_token_returns_none(resolver):
    assert _pid_under_request(
        resolver, path="/x", headers={"X-Practitioner-Token": "nope"}
    ) is None


def test_query_param_wins_when_both_present(resolver):
    """A valid query token still resolves even if a (bad) header is also sent —
    query param is checked first, so existing portal behaviour is unchanged."""
    assert _pid_under_request(
        resolver, path="/x?token=good-token",
        headers={"X-Practitioner-Token": "nope"},
    ) == "pid-xyz"
