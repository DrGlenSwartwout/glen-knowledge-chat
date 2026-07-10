"""The two GHL hooks the C2 nurture sequence depends on.

P1. Completing a founding reservation must tag the contact `reserved:<slug>`.
    Without it the workflow has no exit condition, so a customer who reserved on
    day one keeps being asked to reserve for another thirteen days.

P2. The quiz opt-in must hand GHL the guide URL as a custom field, so email one
    has something to deliver. Two bugs blocked this: the token was minted AFTER
    the onboarding thread started, and ghl_onboard_contact dropped custom_fields
    on the floor even though ghl_upsert_contact accepts them.
"""
import importlib
import inspect
import sys
from pathlib import Path

import pytest

repo = Path(__file__).resolve().parent.parent
if str(repo) not in sys.path:
    sys.path.insert(0, str(repo))


def _app(tmp_path, monkeypatch):
    monkeypatch.setenv("DATA_DIR", str(tmp_path))
    monkeypatch.delenv("CONSOLE_SECRET", raising=False)
    try:
        import app as appmod
        importlib.reload(appmod)
    except Exception as e:
        pytest.skip(f"app not importable: {e}")
    return appmod


# ── P2: the guide URL must be able to reach GHL ──────────────────────────────

def test_onboard_forwards_custom_fields_to_upsert(tmp_path, monkeypatch):
    """ghl_onboard_contact used to silently drop custom_fields."""
    appmod = _app(tmp_path, monkeypatch)
    seen = {}

    def _spy_upsert(email, first_name="", last_name="", phone="", source_tag="",
                    extra_tags=None, custom_fields=None):
        seen["custom_fields"] = custom_fields
        return "cid-1", True, None

    monkeypatch.setattr(appmod, "ghl_upsert_contact", _spy_upsert)
    monkeypatch.setattr(appmod, "ghl_add_to_pipeline", lambda *a, **k: ("opp-1", None))
    monkeypatch.setattr(appmod, "ghl_enroll_workflow", lambda *a, **k: (None, None))

    appmod.ghl_onboard_contact("a@b.com", "A", custom_fields={"guide_url": "https://x/y"})
    assert seen["custom_fields"] == {"guide_url": "https://x/y"}


def test_onboard_accepts_custom_fields_kwarg(tmp_path, monkeypatch):
    """Signature guard: the kwarg must exist, not just be tolerated by **kwargs."""
    appmod = _app(tmp_path, monkeypatch)
    params = inspect.signature(appmod.ghl_onboard_contact).parameters
    assert "custom_fields" in params


def test_quiz_optin_mints_the_guide_token_before_onboarding(tmp_path, monkeypatch):
    """The onboarding thread cannot send a token that has not been minted yet.
    Read the source: the mint must precede the Thread(...).start() call."""
    appmod = _app(tmp_path, monkeypatch)
    src = inspect.getsource(appmod.begin_quiz_optin)
    mint_at = src.index("_mint_lead_magnet_guide_link(")
    start_at = src.index(".start()")
    assert mint_at < start_at, (
        "guide token is minted after the onboarding thread starts; "
        "the thread cannot see it")


def test_quiz_optin_sends_guide_url_as_a_custom_field(tmp_path, monkeypatch):
    """End to end through the route: GHL receives guide_url."""
    appmod = _app(tmp_path, monkeypatch)
    captured = {}

    def _spy_onboard(email, first_name="", last_name="", phone="",
                     source_tag="", extra_tags=None, custom_fields=None):
        captured["email"] = email
        captured["custom_fields"] = custom_fields or {}
        captured["tags"] = extra_tags or []
        return {}

    monkeypatch.setattr(appmod, "ghl_onboard_contact", _spy_onboard)
    c = appmod.app.test_client()
    r = c.post("/begin/quiz/opt-in", json={
        "email": "lead@example.com", "name": "Lead", "tos": True, "quiz_id": "eye-brain"})
    assert r.status_code == 200, r.get_data(as_text=True)[:200]

    import time
    for _ in range(50):          # the onboard call runs on a daemon thread
        if captured:
            break
        time.sleep(0.02)

    assert captured.get("email") == "lead@example.com"
    url = (captured.get("custom_fields") or {}).get("guide_url", "")
    assert "/begin/quiz/guide?token=" in url, f"no guide_url handed to GHL: {captured}"
    token = url.split("token=", 1)[1]
    assert appmod._validate_lead_magnet_guide_link(token) == "lead@example.com", (
        "the URL handed to GHL does not carry a token that actually validates")


# ── P1: a completed reservation must tag the contact ─────────────────────────

def test_reservation_tags_the_contact_so_the_sequence_can_exit(tmp_path, monkeypatch):
    """Without this tag the nurture workflow keeps asking a paying customer to buy."""
    appmod = _app(tmp_path, monkeypatch)
    tagged = {}

    def _spy_upsert(email, first_name="", last_name="", phone="", source_tag="",
                    extra_tags=None, custom_fields=None):
        tagged["email"] = email
        tagged["tags"] = list(extra_tags or [])
        return "cid-1", False, None

    monkeypatch.setattr(appmod, "ghl_upsert_contact", _spy_upsert)
    appmod._tag_founding_reserved("buyer@example.com", "neuro-magnesium")

    import time
    for _ in range(50):
        if tagged:
            break
        time.sleep(0.02)

    assert tagged.get("email") == "buyer@example.com"
    assert "reserved:neuro-magnesium" in tagged.get("tags", [])


def test_reservation_tagging_never_raises(tmp_path, monkeypatch):
    """It runs inside the checkout-return path. A GHL outage must not break a
    reservation that has already been paid for and written to the database."""
    appmod = _app(tmp_path, monkeypatch)

    def _boom(*a, **k):
        raise RuntimeError("ghl down")

    monkeypatch.setattr(appmod, "ghl_upsert_contact", _boom)
    appmod._tag_founding_reserved("buyer@example.com", "neuro-magnesium")  # must not raise
