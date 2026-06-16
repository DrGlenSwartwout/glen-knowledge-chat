"""Personal ordering path for cert participants (Task 2).

A near-clone of the wholesale checkout with two crucial differences:
  (a) NO wholesale_unlocked gate, and
  (b) NEVER resale-exempt (resale_ok=False) — personal purchases are taxed.

On a fee-free (zelle/wise) order the participant earns 3.5% Wellness Credit
(``wallet.personal_earn_cents``). To avoid double-crediting, ``build_order`` is
called with ``method=None`` (which suppresses its internal 3% fee-free earn) and
the route credits the full 3.5% itself via the explicit-amount, invoice-
idempotent wallet credit primitive.

The heavy deps (portal data, the checkout engine, order ingestion, the wallet)
are stubbed by monkeypatching the imported names in ``app``.
"""

import pytest


# ── fixtures / stubs ───────────────────────────────────────────────────────────

PORTAL = {
    "cart": [{"slug": "x", "qty": 2}],
    "modules_completed": 3,
    "email": "c@x.com",
    "name": "C",
    "wholesale_unlocked": False,       # crux: personal path must NOT require this
    "resale_license_number": None,
    "wallet_balance_cents": 0,
    "quote": {"total_bottles": 2, "subtotal_cents": 12000},
}


class _Recorder:
    """Records the kwargs/args of every call so tests can assert on them."""

    def __init__(self, ret=None):
        self.calls = []          # list of (args, kwargs)
        self._ret = ret

    def __call__(self, *args, **kwargs):
        self.calls.append((args, kwargs))
        return self._ret

    @property
    def called(self):
        return bool(self.calls)

    def last_kwargs(self):
        return self.calls[-1][1]

    def last_args(self):
        return self.calls[-1][0]


@pytest.fixture
def client(monkeypatch):
    import app as appmod
    appmod.app.config["TESTING"] = True

    # signed in by default (override per-test for the 401 case)
    monkeypatch.setattr(appmod, "_practitioner_session_pid", lambda: "pid1")
    monkeypatch.setattr(appmod._pp, "portal_data", lambda pid: dict(PORTAL))

    build_order = _Recorder(ret={
        "ok": True, "invoice_id": "INV1", "doc_number": "1001",
        "total": 120.0, "get_cents": 500, "credit_redeemed_cents": 0,
    })
    monkeypatch.setattr(appmod._wc, "build_order", build_order)

    ingest = _Recorder(ret=None)
    monkeypatch.setattr(appmod, "_ingest_order", ingest)

    # explicit-amount, invoice-idempotent earned-credit primitive
    earn = _Recorder(ret=420)
    monkeypatch.setattr(appmod._wallet, "earn_dropship_margin", earn)

    monkeypatch.setattr(appmod._pp, "cart_clear", lambda pid: None)
    monkeypatch.setattr(appmod._pp, "record_order", lambda *a, **k: None)

    return appmod.app.test_client(), appmod, build_order, ingest, earn


# ── 1. not signed in → 401 on quote + checkout ─────────────────────────────────

def test_quote_not_signed_in(client):
    c, appmod, *_ = client
    import app as _a
    _a._practitioner_session_pid = lambda: None
    r = c.post("/api/practitioner/personal/quote", json={})
    assert r.status_code == 401


def test_checkout_not_signed_in(client):
    c, appmod, *_ = client
    import app as _a
    _a._practitioner_session_pid = lambda: None
    r = c.post("/api/practitioner/personal/checkout", json={})
    assert r.status_code == 401


# ── 2. checkout succeeds WITHOUT wholesale_unlocked (no 403) ───────────────────

def test_checkout_ok_without_wholesale_unlock(client):
    c, appmod, build_order, ingest, earn = client
    r = c.post("/api/practitioner/personal/checkout", json={"method": "zelle"})
    assert r.status_code == 200
    body = r.get_json()
    assert body["ok"] is True
    assert body["invoice_id"] == "INV1"


# ── 3. build_order called with resale_ok=False ─────────────────────────────────

def test_build_order_resale_ok_false(client):
    c, appmod, build_order, ingest, earn = client
    c.post("/api/practitioner/personal/checkout", json={"method": "zelle"})
    assert build_order.called
    assert build_order.last_kwargs().get("resale_ok") is False


# ── 4. _ingest_order called with channel="personal" ───────────────────────────

def test_ingest_channel_personal(client):
    c, appmod, build_order, ingest, earn = client
    c.post("/api/practitioner/personal/checkout", json={"method": "zelle"})
    assert ingest.called
    assert ingest.last_kwargs().get("channel") == "personal"


# ── 5. empty cart → 400 ────────────────────────────────────────────────────────

def test_empty_cart_400(client, monkeypatch):
    c, appmod, *_ = client
    empty = dict(PORTAL)
    empty["cart"] = []
    monkeypatch.setattr(appmod._pp, "portal_data", lambda pid: empty)
    r = c.post("/api/practitioner/personal/checkout", json={"method": "zelle"})
    assert r.status_code == 400


# ── 6. fee-free earn: zelle credits 3.5% (420), card credits 0 ─────────────────

def test_zelle_credits_personal_earn(client):
    c, appmod, build_order, ingest, earn = client
    c.post("/api/practitioner/personal/checkout", json={"method": "zelle"})
    # build_order must be called with method=None so it does NOT also earn 3%
    assert build_order.last_kwargs().get("method") is None
    assert earn.called
    # charged_cents = round(total*100) = 12000 ; 3.5% -> 420
    args, kwargs = earn.calls[-1]
    flat = list(args) + list(kwargs.values())
    assert 420 in flat


def test_wise_credits_personal_earn(client):
    c, appmod, build_order, ingest, earn = client
    c.post("/api/practitioner/personal/checkout", json={"method": "wise"})
    assert earn.called
    args, kwargs = earn.calls[-1]
    flat = list(args) + list(kwargs.values())
    assert 420 in flat


def test_card_credits_zero(client, monkeypatch):
    c, appmod, build_order, ingest, earn = client
    # force card to be honored (Stripe active) so we exercise the card branch
    monkeypatch.setattr(appmod, "_STRIPE_ACTIVE", True, raising=False)
    monkeypatch.setattr(appmod, "_stripe_checkout_url_for_order",
                        lambda *a, **k: "https://stripe/x", raising=False)
    c.post("/api/practitioner/personal/checkout", json={"method": "card"})
    # card => personal_earn_cents == 0 => credit fn not called (or called with 0)
    if earn.called:
        args, kwargs = earn.calls[-1]
        flat = list(args) + list(kwargs.values())
        assert all(v != 420 for v in flat)
        assert 0 in flat or not any(isinstance(v, int) and v > 0 for v in flat)
    else:
        assert not earn.called


# ── quote happy path ───────────────────────────────────────────────────────────

def test_quote_ok(client):
    c, appmod, *_ = client
    r = c.post("/api/practitioner/personal/quote", json={})
    assert r.status_code == 200
    body = r.get_json()
    assert body["ok"] is True
    assert body["quote"] == PORTAL["quote"]
    assert body["wallet_balance_cents"] == 0
