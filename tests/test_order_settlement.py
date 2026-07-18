from types import SimpleNamespace
from dashboard import order_settlement as osx

class _Deps:
    def __init__(self, raise_on=None):
        self.calls = []
        self._raise_on = raise_on or set()
    def _rec(self, name):
        self.calls.append(name)
        if name in self._raise_on:
            raise RuntimeError(f"boom:{name}")
    def settle_points(self, order, order_ref): self._rec("points")
    def settle_referral(self, order, order_ref): self._rec("referral")
    def ensure_subscription(self, md, pi_id): self._rec("subscription")
    def grant_group_bundle(self, md, pi_id): self._rec("group_bundle")
    def settle_client(self, md): self._rec("client")
    def settle_biofield(self, md, sid): self._rec("biofield")
    def grant_membership_line(self, order): self._rec("membership_line")

_ORDER = {"id": 1, "email": "a@b.com"}
_MD = {"invoice_id": "tok1", "kind": "retail"}

def _run(kind, deps, order=_ORDER, md=None):
    return osx.settle_paid_order_effects(
        kind=kind, order=order, md=md or {"invoice_id": "tok1", "kind": kind},
        pi_id="pi_1", sid="sess_1", deps=deps)

def test_retail_settles_points_and_referral_only():
    d = _Deps(); out = _run("retail", d)
    assert d.calls == ["points", "referral"]
    assert set(out["settled"]) == {"points", "referral"}

def test_subscribe_without_grant_group_months_skips_group_bundle():
    # subscribe never carries grant_group_months today -- group_bundle must NOT
    # fire just because kind=='subscribe'; only ensure_subscription runs.
    d = _Deps(); _run("subscribe", d)
    assert d.calls == ["points", "referral", "subscription"]

def test_subscribe_with_grant_group_months_adds_group_bundle():
    d = _Deps()
    _run("subscribe", d, md={"invoice_id": "tok1", "kind": "subscribe",
                              "grant_group_months": "3"})
    assert d.calls == ["points", "referral", "subscription", "group_bundle"]

def test_non_subscribe_kind_with_grant_group_months_calls_group_bundle():
    # Dispatch is kind-agnostic: any kind carrying grant_group_months (today
    # only retail program orders do) must fire group_bundle via the
    # orchestrator, not just subscribe.
    d = _Deps()
    out = _run("retail", d, md={"invoice_id": "tok1", "kind": "retail",
                                 "grant_group_months": "1"})
    assert d.calls == ["points", "referral", "group_bundle"]
    assert set(out["settled"]) == {"points", "referral", "group_bundle"}

def test_client_settles_common_points_and_client():
    # Behavior-preserving: client goes through the shared gate today, so it gets
    # common points+referral AND its own client settlement (dispensary-scope).
    d = _Deps(); out = _run("client", d)
    assert d.calls == ["points", "referral", "client"]
    assert set(out["settled"]) == {"points", "referral", "client"}

def test_biofield_settles_common_plus_biofield():
    d = _Deps(); _run("biofield", d)
    assert d.calls == ["points", "referral", "biofield"]

def test_reorder_and_portal_reorder_like_retail():
    for k in ("reorder", "portal-reorder"):
        d = _Deps(); _run(k, d)
        assert d.calls == ["points", "referral"]

def test_one_settler_raising_is_recorded_and_others_continue():
    d = _Deps(raise_on={"points"})
    out = _run("subscribe", d, md={"invoice_id": "tok1", "kind": "subscribe",
                                    "grant_group_months": "3"})
    # points raises but referral/subscription/group_bundle still run
    assert d.calls == ["points", "referral", "subscription", "group_bundle"]
    assert "points" in out["skipped"]
    assert "referral" in out["settled"]

def test_no_order_skips_common_points_referral():
    d = _Deps(); out = _run("retail", d, order=None)
    assert d.calls == []
    assert out["settled"] == []

def test_unknown_kind_noop():
    d = _Deps(); out = _run("membership_product", d)
    assert d.calls == []

_MEMBERSHIP_ORDER = {"id": 2, "email": "m@b.com",
                     "items": [{"slug": "membership:month", "kind": "membership",
                                "tier": "month"}]}

def test_membership_line_dispatched_only_when_line_present():
    # No membership line on the order -> the grant dep is NOT dispatched (existing
    # non-membership orders are untouched).
    d = _Deps(); _run("retail", d)
    assert "membership_line" not in d.calls
    # Order carries a membership line -> the grant dep fires (kind-agnostic).
    d2 = _Deps(); out = _run("retail", d2, order=_MEMBERSHIP_ORDER)
    assert d2.calls == ["points", "referral", "membership_line"]
    assert "membership_line" in out["settled"]

def test_membership_grant_that_raises_lands_in_skipped():
    d = _Deps(raise_on={"membership_line"})
    out = _run("retail", d, order=_MEMBERSHIP_ORDER)
    assert "membership_line" in out["skipped"]
    assert "membership_line" not in out["settled"]

def test_group_bundle_grant_that_raises_lands_in_skipped():
    # Orchestrator-level guard: if the injected grant_group_bundle dep raises
    # (rather than swallowing internally and returning None), `_do`'s own
    # try/except must catch it and record "group_bundle" in `skipped` -- that's
    # what lets the settlement-todo surface a failed grant instead of it
    # silently looking settled forever.
    d = _Deps(raise_on={"group_bundle"})
    out = _run("retail", d, md={"invoice_id": "tok1", "kind": "retail",
                                 "grant_group_months": "1"})
    assert "group_bundle" in out["skipped"]
    assert "group_bundle" not in out["settled"]

def test_real_grant_group_bundle_failure_reaches_orchestrator_skipped(monkeypatch, tmp_path):
    # True end-to-end: wire the REAL app._grant_group_bundle (not a mock) in as
    # the orchestrator's grant_group_bundle dep, make its internal
    # create_membership call blow up, and confirm the failure surfaces as
    # "group_bundle" in `skipped` -- proving _grant_group_bundle now propagates
    # instead of swallowing, so a real production failure is never invisible.
    import sqlite3
    import app as appmod
    from dashboard import stripe_pay as _stripe_pay_mod
    from dashboard import subscriptions as subs

    db = str(tmp_path / "log.db")
    monkeypatch.setattr(appmod, "LOG_DB", db)
    monkeypatch.setenv("GROUP_BUNDLE_ENABLED", "1")
    monkeypatch.setattr(
        _stripe_pay_mod, "get_payment_intent",
        lambda pi: {"customer": "cus_1", "payment_method": "pm_1"})

    def _boom(cx, **kwargs):
        raise RuntimeError("boom: create_membership")

    monkeypatch.setattr(subs, "create_membership", _boom)

    class _RealGrantDeps(_Deps):
        def grant_group_bundle(self, md, pi_id):
            appmod._grant_group_bundle(md, pi_id)

    d = _RealGrantDeps()
    md = {"invoice_id": "tok1", "kind": "retail", "grant_group_months": "1",
          "email": "a@b.com"}
    out = osx.settle_paid_order_effects(
        kind="retail", order=_ORDER, md=md, pi_id="pi_1", sid="sess_1", deps=d)

    assert "group_bundle" in out["skipped"]
    assert "group_bundle" not in out["settled"]
