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

def test_subscribe_adds_subscription_and_group_bundle():
    d = _Deps(); _run("subscribe", d)
    assert d.calls == ["points", "referral", "subscription", "group_bundle"]

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
    d = _Deps(raise_on={"points"}); out = _run("subscribe", d)
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
