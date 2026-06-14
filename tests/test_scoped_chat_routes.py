"""Tests for POST /api/practitioner/chat and POST /api/client/<code>/chat.

Stubs: appmod._chat.scoped_reply, appmod._pp.practitioner_id_by_dispensary_code,
appmod._practitioner_session_pid, appmod.is_member, appmod._dropship.practitioner_price_for.
"""

import pytest
import app as appmod


@pytest.fixture()
def client(monkeypatch):
    appmod.app.config["TESTING"] = True
    appmod.app.config["SECRET_KEY"] = "test"
    return appmod.app.test_client()


# ── /api/practitioner/chat ────────────────────────────────────────────────────

def test_practitioner_chat_requires_auth(client, monkeypatch):
    """401 when there is no practitioner session."""
    monkeypatch.setattr(appmod, "_practitioner_session_pid", lambda: None)
    resp = client.post("/api/practitioner/chat", json={"message": "hi"})
    assert resp.status_code == 401
    data = resp.get_json()
    assert data["ok"] is False


def test_practitioner_chat_happy_path(client, monkeypatch):
    """Authed practitioner gets reply + priced suggestions."""
    monkeypatch.setattr(appmod, "_practitioner_session_pid", lambda: "p1")
    monkeypatch.setattr(
        appmod._chat, "scoped_reply",
        lambda message, history, catalog: {
            "reply": "Brain Boost is a great fit for focus.",
            "suggested_slugs": ["brain-boost"],
        },
    )
    monkeypatch.setattr(
        appmod._dropship, "practitioner_price_for",
        lambda pid, slug: 8400,
    )
    # Stub _get_product so the route can build suggestion name without real products.json
    monkeypatch.setattr(
        appmod, "_get_product",
        lambda slug: {"name": "Brain Boost", "slug": slug, "price_cents": 6997}
        if slug == "brain-boost" else None,
    )

    resp = client.post(
        "/api/practitioner/chat",
        json={"message": "something for focus", "history": []},
    )
    assert resp.status_code == 200
    data = resp.get_json()
    assert data["ok"] is True
    assert "Brain Boost" in data["reply"]
    assert len(data["suggestions"]) == 1
    s = data["suggestions"][0]
    assert s["slug"] == "brain-boost"
    assert s["price_cents"] == 8400
    assert s["name"] == "Brain Boost"


# ── /api/client/<code>/chat ───────────────────────────────────────────────────

def test_client_chat_unknown_code_returns_404(client, monkeypatch):
    """Unknown dispensary code → 404."""
    monkeypatch.setattr(appmod._pp, "practitioner_id_by_dispensary_code", lambda code: None)
    resp = client.post("/api/client/BADCODE/chat", json={"email": "p@x.com", "message": "hi"})
    assert resp.status_code == 404
    data = resp.get_json()
    assert data["ok"] is False


def test_client_chat_not_member_returns_403(client, monkeypatch):
    """Patient who hasn't opted in gets 403 need_optin."""
    monkeypatch.setattr(appmod._pp, "practitioner_id_by_dispensary_code", lambda code: "p1")
    monkeypatch.setattr(appmod, "is_member", lambda session_id, email: False)
    resp = client.post(
        "/api/client/ABC123/chat",
        json={"email": "patient@example.com", "message": "hi"},
    )
    assert resp.status_code == 403
    data = resp.get_json()
    assert data["ok"] is False
    assert data.get("need_optin") is True


def test_client_chat_happy_path(client, monkeypatch):
    """Member patient gets reply + suggestions priced at practitioner price."""
    monkeypatch.setattr(appmod._pp, "practitioner_id_by_dispensary_code", lambda code: "p1")
    monkeypatch.setattr(appmod, "is_member", lambda session_id, email: True)
    monkeypatch.setattr(
        appmod._chat, "scoped_reply",
        lambda message, history, catalog: {
            "reply": "Brain Boost is a great fit.",
            "suggested_slugs": ["brain-boost"],
        },
    )
    monkeypatch.setattr(
        appmod._dropship, "practitioner_price_for",
        lambda pid, slug: 8400,
    )
    monkeypatch.setattr(
        appmod, "_get_product",
        lambda slug: {"name": "Brain Boost", "slug": slug, "price_cents": 6997}
        if slug == "brain-boost" else None,
    )

    resp = client.post(
        "/api/client/ABC123/chat",
        json={"email": "patient@example.com", "message": "focus help", "history": []},
    )
    assert resp.status_code == 200
    data = resp.get_json()
    assert data["ok"] is True
    assert "Brain Boost" in data["reply"]
    assert len(data["suggestions"]) == 1
    s = data["suggestions"][0]
    assert s["slug"] == "brain-boost"
    assert s["price_cents"] == 8400
