"""PR1 gate: _is_paid_member is now FULL-members-only (membership_category=='full').

Trial members (the $1-trial free first month) price at regular and accrue the
missed discount as upgrade credit, so the volume-pricing gate must reject them.
The real DB->category mapping is covered by test_membership_category.py; here we
verify the app-level gate maps category -> bool correctly and stays fail-closed.
"""
import importlib
import sys
from pathlib import Path

import pytest


def _app():
    repo = Path(__file__).resolve().parent.parent
    if str(repo) not in sys.path:
        sys.path.insert(0, str(repo))
    try:
        return importlib.import_module("app")
    except Exception as e:  # missing env in some CI
        pytest.skip(f"app not importable: {e}")


def _has_grant(monkeypatch, appmod, present=True):
    """Stub the membership-grant lookup (the cheap first check in the gate)."""
    monkeypatch.setattr(appmod, "_active_membership_for_email",
                        lambda e: {"id": "m"} if present else None)


# A member of ANY kind except an unconverted trial gets the discount. With an
# active grant present, every non-'trial' category passes (full, paused, and
# 'none' = a founding/studio/coaching grant with no $99 sub).
@pytest.mark.parametrize("cat", ["full", "paused", "none"])
def test_real_members_pass_gate(monkeypatch, cat):
    appmod = _app()
    _has_grant(monkeypatch, appmod, present=True)
    monkeypatch.setattr(appmod, "membership_category", lambda e: cat)
    assert appmod._is_paid_member("a@x.com") is True


def test_trial_with_grant_fails_gate(monkeypatch):
    """A $1-trial buyer HAS an active grant but category 'trial' -> regular price."""
    appmod = _app()
    _has_grant(monkeypatch, appmod, present=True)
    monkeypatch.setattr(appmod, "membership_category", lambda e: "trial")
    assert appmod._is_paid_member("a@x.com") is False


def test_no_active_grant_fails_gate(monkeypatch):
    """No active membership grant -> not a member -> regular, regardless of category."""
    appmod = _app()
    _has_grant(monkeypatch, appmod, present=False)
    monkeypatch.setattr(appmod, "membership_category", lambda e: "full")
    assert appmod._is_paid_member("a@x.com") is False


def test_empty_email_fails_gate():
    appmod = _app()
    assert appmod._is_paid_member("") is False
    assert appmod._is_paid_member(None) is False


def test_gate_is_fail_closed_on_error(monkeypatch):
    appmod = _app()
    def _boom(_e):
        raise RuntimeError("db down")
    monkeypatch.setattr(appmod, "_active_membership_for_email", _boom)
    assert appmod._is_paid_member("a@x.com") is False


def test_trial_email_prices_regular_in_preview(monkeypatch):
    """End-to-end: a trial customer gets regular (list) FF pricing, not the volume rate."""
    appmod = _app()
    FF = {"slug": "brain", "qty_pricing": True, "price_cents": 6997, "name": "Brain Boost"}
    monkeypatch.setattr(appmod, "_get_product", {"brain": FF}.get)
    monkeypatch.setattr(appmod, "_active_membership_for_email",
                        lambda e: {"id": "m"} if e == "trial@x.com" else None)  # trial HAS a grant
    monkeypatch.setattr(appmod, "membership_category",
                        lambda e: "trial" if e == "trial@x.com" else "none")
    client = appmod.app.test_client()
    key = appmod.dashboard.CONSOLE_SECRET or ""
    r = client.post("/api/orders/price-preview",
                    json={"email": "trial@x.com", "lines": [{"slug": "brain", "qty": 6}]},
                    headers={"X-Console-Key": key})
    assert r.status_code == 200
    j = r.get_json()
    assert j["ok"]
    # trial -> not a full member -> regular list price, NO volume rate
    assert j["lines"][0]["effective_unit_cents"] == 6997
    assert j["lines"][0]["vol_pct"] == 0


def test_full_email_prices_volume_in_preview(monkeypatch):
    """Contrast: a full member gets the order-wide volume rate on the same cart."""
    appmod = _app()
    FF = {"slug": "brain", "qty_pricing": True, "price_cents": 6997, "name": "Brain Boost"}
    monkeypatch.setattr(appmod, "_get_product", {"brain": FF}.get)
    monkeypatch.setattr(appmod, "_active_membership_for_email",
                        lambda e: {"id": "m"} if e == "full@x.com" else None)
    monkeypatch.setattr(appmod, "membership_category",
                        lambda e: "full" if e == "full@x.com" else "none")
    client = appmod.app.test_client()
    key = appmod.dashboard.CONSOLE_SECRET or ""
    r = client.post("/api/orders/price-preview",
                    json={"email": "full@x.com", "lines": [{"slug": "brain", "qty": 6}]},
                    headers={"X-Console-Key": key})
    assert r.status_code == 200
    j = r.get_json()
    assert j["ok"]
    # full member -> volume rate at total FF qty 6 (5629, per existing #394 tests)
    assert j["lines"][0]["effective_unit_cents"] == 5629
    assert j["lines"][0]["vol_pct"] > 0
