# Spec 2b-1 — Referral Codes Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Every customer gets a stable referral code; a referee enters it at checkout for 10% off (once ever, no self-referral), resolved through the existing pricing engine. Single-sided (referrer reward = 2b-2).

**Architecture:** A `referrals` store (code per email + redemptions keyed by referee email) + a resolve/record API; a checkout helper that turns a referral code into a `coupon_pct = max(10, daily)`; the funnel begin-buy + reorder checkouts use it and record one redemption per referee at order creation. Flag-gated behind `REFERRALS`.

**Tech Stack:** Python 3.11, Flask, SQLite (`chat_log.db`/`LOG_DB`), the existing pricing engine (`_price_cart`/`pricing.compute`), `secrets`, pytest.

## Global Constraints

- **Discount = 10%** for the referee (env `REFERRAL_PCT`, default 10), resolved as `coupon_pct = max(referral_pct, _active_coupon_pct() or 0)`; flows into `pricing.compute` unchanged (best-of-one with volume, no subscriber stacking, per-SKU floors are the hard cap).
- **One redemption per referee, ever** — `referral_redemptions.referee_email` PRIMARY KEY enforces it; `resolve` also returns None once `has_redeemed`.
- **Self-referral blocked** (referee email ≠ code owner, case-insensitive).
- **Evergreen codes** (one per person, no expiry). **Single-sided** (no referrer reward this spec).
- **Flag `REFERRALS`** (default off): fully inert — no code minted, no referral discount, no recording.
- All referral reads/writes wrapped so a referral failure never blocks checkout/order creation (degrade to the normal coupon).
- NO emoji; no em dashes in generated text.
- **Test command (every task):** `doppler run -p remedy-match -c prd -- env DATA_DIR=$(mktemp -d) ~/.venvs/deploy-chat311/bin/python -m pytest tests/test_referrals_spec2b1.py -v`
- Tests reload `app` via `importlib` with flags; emails lowercased everywhere.

---

### Task 1: `dashboard/referrals.py` store

**Files:**
- Create: `dashboard/referrals.py`
- Test: `tests/test_referrals_spec2b1.py` (create)

**Interfaces:**
- Produces:
  - `init_tables(cx)`
  - `get_or_create_code(cx, email) -> str` (stable per email; unique uppercase code)
  - `owner_of(cx, code) -> str|None`
  - `has_redeemed(cx, referee_email) -> bool`
  - `resolve(cx, code, referee_email, *, pct) -> {"owner_email": str, "coupon_pct": int} | None`
  - `record_redemption(cx, code, owner_email, referee_email, order_ref) -> bool`

- [ ] **Step 1: Write the failing tests**

Create `tests/test_referrals_spec2b1.py`:

```python
import sqlite3
from dashboard import referrals as rf


def _cx():
    return sqlite3.connect(":memory:")


def test_get_or_create_code_stable_and_unique():
    cx = _cx()
    c1 = rf.get_or_create_code(cx, "Owner@X.com")
    c2 = rf.get_or_create_code(cx, "owner@x.com")          # same person (lowercased) -> same code
    assert c1 == c2 and c1
    c3 = rf.get_or_create_code(cx, "other@x.com")
    assert c3 != c1
    assert rf.owner_of(cx, c1) == "owner@x.com"
    assert rf.owner_of(cx, "NOPE") is None


def test_resolve_valid_and_guards():
    cx = _cx()
    code = rf.get_or_create_code(cx, "owner@x.com")
    # valid referee
    assert rf.resolve(cx, code, "friend@x.com", pct=10) == {"owner_email": "owner@x.com", "coupon_pct": 10}
    # self-referral blocked (case-insensitive)
    assert rf.resolve(cx, code, "OWNER@x.com", pct=10) is None
    # unknown code
    assert rf.resolve(cx, "NOPE", "friend@x.com", pct=10) is None


def test_one_redemption_per_referee_ever():
    cx = _cx()
    code = rf.get_or_create_code(cx, "owner@x.com")
    assert rf.has_redeemed(cx, "friend@x.com") is False
    assert rf.record_redemption(cx, code, "owner@x.com", "Friend@x.com", "INV-1") is True
    assert rf.has_redeemed(cx, "friend@x.com") is True       # lowercased
    # a second redemption by the same referee is a no-op insert, and resolve now blocks
    assert rf.record_redemption(cx, code, "owner@x.com", "friend@x.com", "INV-2") is False
    assert rf.resolve(cx, code, "friend@x.com", pct=10) is None
```

- [ ] **Step 2: Run to verify they fail**

Run the test command. Expected: FAIL (`ModuleNotFoundError: dashboard.referrals`).

- [ ] **Step 3: Implement `dashboard/referrals.py`**

```python
import datetime
import secrets


def _now():
    return datetime.datetime.now(datetime.timezone.utc).isoformat()


def _norm(email):
    return (email or "").strip().lower()


def init_tables(cx):
    cx.execute("CREATE TABLE IF NOT EXISTS referral_codes ("
               "email TEXT PRIMARY KEY, code TEXT UNIQUE, created_at TEXT)")
    cx.execute("CREATE TABLE IF NOT EXISTS referral_redemptions ("
               "referee_email TEXT PRIMARY KEY, code TEXT, owner_email TEXT, "
               "order_ref TEXT, created_at TEXT)")
    cx.commit()


def get_or_create_code(cx, email):
    init_tables(cx)
    e = _norm(email)
    row = cx.execute("SELECT code FROM referral_codes WHERE email=?", (e,)).fetchone()
    if row:
        return row[0]
    for _ in range(10):
        code = secrets.token_urlsafe(6).replace("_", "").replace("-", "")[:8].upper()
        try:
            cx.execute("INSERT INTO referral_codes (email, code, created_at) VALUES (?,?,?)",
                       (e, code, _now()))
            cx.commit()
            return code
        except Exception:  # noqa: BLE001 - UNIQUE collision, retry
            continue
    raise RuntimeError("could not mint a unique referral code")


def owner_of(cx, code):
    init_tables(cx)
    row = cx.execute("SELECT email FROM referral_codes WHERE code=?", (code,)).fetchone()
    return row[0] if row else None


def has_redeemed(cx, referee_email):
    init_tables(cx)
    return cx.execute("SELECT 1 FROM referral_redemptions WHERE referee_email=?",
                      (_norm(referee_email),)).fetchone() is not None


def resolve(cx, code, referee_email, *, pct):
    init_tables(cx)
    owner = owner_of(cx, code)
    ref = _norm(referee_email)
    if not owner or owner == ref or has_redeemed(cx, ref):
        return None
    return {"owner_email": owner, "coupon_pct": int(pct)}


def record_redemption(cx, code, owner_email, referee_email, order_ref):
    init_tables(cx)
    cur = cx.execute(
        "INSERT OR IGNORE INTO referral_redemptions (referee_email, code, owner_email, order_ref, created_at) "
        "VALUES (?,?,?,?,?)",
        (_norm(referee_email), code, _norm(owner_email), order_ref or "", _now()))
    cx.commit()
    return cur.rowcount > 0
```

- [ ] **Step 4: Run to verify they pass**

Run the test command. Expected: 3 passed.

- [ ] **Step 5: Commit**

```bash
git add dashboard/referrals.py tests/test_referrals_spec2b1.py
git commit -m "feat(referrals-2b1): referral code store + resolve/record (one-per-referee, no self-referral)"
```

---

### Task 2: `REFERRALS` flag + resolve helper + `/api/referral/my-code`

**Files:**
- Modify: `app.py`
- Test: `tests/test_referrals_spec2b1.py` (append)

**Interfaces:**
- Consumes: Task 1 `referrals.*`; existing `_active_coupon_pct`, `get_authenticated_user`, `_reorder_email_from_cookie`, `LOG_DB`.
- Produces: `_REFERRALS` flag; `_referral_pct() -> int`; `_resolve_checkout_coupon_pct(referral_code, referee_email) -> (pct, ctx)`; `GET /api/referral/my-code`.

- [ ] **Step 1: Write the failing tests**

Append:

```python
import importlib


def _reload_ref_app(monkeypatch, tmp_path, referrals="true"):
    monkeypatch.setenv("DATA_DIR", str(tmp_path))
    monkeypatch.setenv("REFERRALS", referrals)
    monkeypatch.setenv("REFERRAL_PCT", "10")
    import app as appmod
    importlib.reload(appmod)
    return appmod


def test_resolve_checkout_coupon_pct(monkeypatch, tmp_path):
    appmod = _reload_ref_app(monkeypatch, tmp_path)
    monkeypatch.setattr(appmod, "_active_coupon_pct", lambda: None)
    import sqlite3
    from dashboard import referrals as rf
    with sqlite3.connect(appmod.LOG_DB) as cx:
        code = rf.get_or_create_code(cx, "owner@x.com")
    # valid referee -> referral pct + ctx
    pct, ctx = appmod._resolve_checkout_coupon_pct(code, "friend@x.com")
    assert pct == 10 and ctx == {"code": code, "owner_email": "owner@x.com"}
    # self-referral -> falls back to daily (None here), no ctx
    pct, ctx = appmod._resolve_checkout_coupon_pct(code, "owner@x.com")
    assert pct is None and ctx is None
    # daily coupon beats a smaller referral -> max wins
    monkeypatch.setattr(appmod, "_active_coupon_pct", lambda: 15)
    pct, ctx = appmod._resolve_checkout_coupon_pct(code, "friend@x.com")
    assert pct == 15 and ctx == {"code": code, "owner_email": "owner@x.com"}


def test_resolve_flag_off(monkeypatch, tmp_path):
    appmod = _reload_ref_app(monkeypatch, tmp_path, referrals="false")
    monkeypatch.setattr(appmod, "_active_coupon_pct", lambda: 5)
    pct, ctx = appmod._resolve_checkout_coupon_pct("ANYCODE", "friend@x.com")
    assert pct == 5 and ctx is None          # referral ignored when flag off


def test_my_code_endpoint(monkeypatch, tmp_path):
    appmod = _reload_ref_app(monkeypatch, tmp_path)
    c = appmod.app.test_client()
    c.set_cookie("rm_reorder_email", "owner@x.com")     # _reorder_email_from_cookie source
    r1 = c.get("/api/referral/my-code").get_json()
    r2 = c.get("/api/referral/my-code").get_json()
    assert r1["code"] and r1["code"] == r2["code"]       # stable


def test_my_code_404_when_flag_off(monkeypatch, tmp_path):
    appmod = _reload_ref_app(monkeypatch, tmp_path, referrals="false")
    c = appmod.app.test_client(); c.set_cookie("rm_reorder_email", "owner@x.com")
    assert c.get("/api/referral/my-code").status_code == 404
```

- [ ] **Step 2: Run to verify they fail**

Run the test command. Expected: FAIL (`_resolve_checkout_coupon_pct` undefined / route 404).

- [ ] **Step 3: Implement in `app.py`**

Add the flag near the other reviews/feature flags (e.g. after `_REVIEWS_GIFTS`):

```python
_REFERRALS = os.environ.get("REFERRALS", "").strip().lower() in ("1", "true", "yes")


def _referral_pct():
    try:
        return int(os.environ.get("REFERRAL_PCT", "10"))
    except (TypeError, ValueError):
        return 10
```

Add the resolve helper + endpoint (place after `_active_coupon_pct`, app.py ~7656):

```python
def _resolve_checkout_coupon_pct(referral_code, referee_email):
    """Return (effective_coupon_pct, referral_ctx|None). A valid referral code yields
    max(referral_pct, daily); otherwise the daily coupon with no referral context."""
    daily = _active_coupon_pct()
    code = (referral_code or "").strip()
    if not _REFERRALS or not code or not (referee_email or "").strip():
        return daily, None
    try:
        from dashboard import referrals as _rf
        with sqlite3.connect(LOG_DB) as cx:
            res = _rf.resolve(cx, code, referee_email, pct=_referral_pct())
        if not res:
            return daily, None
        eff = max(res["coupon_pct"], daily or 0)
        return eff, {"code": code, "owner_email": res["owner_email"]}
    except Exception as e:  # noqa: BLE001 - referral never blocks checkout
        print(f"[referrals] resolve failed: {e}", flush=True)
        return daily, None


@app.route("/api/referral/my-code", methods=["GET"])
def api_referral_my_code():
    if not _REFERRALS:
        return ("", 404)
    au = get_authenticated_user(request) or {}
    email = (au.get("email") or _reorder_email_from_cookie() or "").strip().lower()
    if not email:
        return jsonify({"ok": False, "error": "no email"}), 400
    from dashboard import referrals as _rf
    with sqlite3.connect(LOG_DB) as cx:
        code = _rf.get_or_create_code(cx, email)
    return jsonify({"ok": True, "code": code,
                    "share_text": f"Use my code {code} for 10% off at illtowell.com"})
```

- [ ] **Step 4: Run to verify they pass**

Run the test command. Expected: all Task-1 + Task-2 tests pass.

- [ ] **Step 5: Commit**

```bash
git add app.py tests/test_referrals_spec2b1.py
git commit -m "feat(referrals-2b1): REFERRALS flag, checkout coupon resolver, my-code endpoint"
```

---

### Task 3: Checkout integration + redemption recording (begin-buy + reorder)

**Files:**
- Modify: `app.py` (the funnel begin-buy checkout ~3645 and `reorder_checkout` ~9516; a `_record_referral_if_any` helper)
- Test: `tests/test_referrals_spec2b1.py` (append)

**Interfaces:**
- Consumes: Task 1 `referrals.record_redemption`; Task 2 `_resolve_checkout_coupon_pct`.
- Produces: both checkouts resolve the referral code into `coupon_pct` and record one redemption at order creation; `_record_referral_if_any(ctx, referee_email, order_ref) -> bool`.

- [ ] **Step 1: Write the failing test**

Append (unit-test the record helper directly — the full QBO checkout routes are covered by their existing tests and a manual smoke):

```python
def test_record_referral_if_any(monkeypatch, tmp_path):
    appmod = _reload_ref_app(monkeypatch, tmp_path)
    import sqlite3
    from dashboard import referrals as rf
    with sqlite3.connect(appmod.LOG_DB) as cx:
        code = rf.get_or_create_code(cx, "owner@x.com")
    ctx = {"code": code, "owner_email": "owner@x.com"}
    assert appmod._record_referral_if_any(ctx, "friend@x.com", "INV-1") is True
    with sqlite3.connect(appmod.LOG_DB) as cx:
        assert rf.has_redeemed(cx, "friend@x.com") is True
    # None ctx -> no-op; second time for same referee -> False (already redeemed)
    assert appmod._record_referral_if_any(None, "friend@x.com", "INV-2") is False
    assert appmod._record_referral_if_any(ctx, "friend@x.com", "INV-2") is False


def test_record_referral_flag_off(monkeypatch, tmp_path):
    appmod = _reload_ref_app(monkeypatch, tmp_path, referrals="false")
    ctx = {"code": "X", "owner_email": "owner@x.com"}
    assert appmod._record_referral_if_any(ctx, "friend@x.com", "INV-1") is False
```

- [ ] **Step 2: Run to verify they fail**

Run the test command. Expected: FAIL (`_record_referral_if_any` undefined).

- [ ] **Step 3: Implement in `app.py`**

Add the record helper near `_resolve_checkout_coupon_pct`:

```python
def _record_referral_if_any(referral_ctx, referee_email, order_ref):
    """Record a single referral redemption for this order, best-effort. Returns True if recorded."""
    if not _REFERRALS or not referral_ctx:
        return False
    try:
        from dashboard import referrals as _rf
        with sqlite3.connect(LOG_DB) as cx:
            return _rf.record_redemption(cx, referral_ctx["code"], referral_ctx["owner_email"],
                                         referee_email, order_ref)
    except Exception as e:  # noqa: BLE001 - referral never blocks order creation
        print(f"[referrals] record failed: {e}", flush=True)
        return False
```

In the **funnel begin-buy** checkout (app.py ~3644), replace the `coupon_pct=_active_coupon_pct()` call. Before the `pc = _price_cart(...)`:

```python
        _ref_pct, _ref_ctx = _resolve_checkout_coupon_pct(data.get("referral_code"), email)
        try:
            pc = _price_cart([{"slug": slug, "qty": qty}], ship=ship,
                             coupon_pct=_ref_pct,
                             points_to_redeem_cents=redeem)
        except CheckoutError as ce:
            return jsonify({"ok": False, "error": str(ce)}), 400
```

After the invoice is created (`inv = qb.create_invoice(...)`), record the redemption:

```python
        _record_referral_if_any(_ref_ctx, email, inv.get("Id"))
```

In **`reorder_checkout`** (app.py ~9515), do the same — replace `coupon_pct=_active_coupon_pct()`:

```python
            _ref_pct, _ref_ctx = _resolve_checkout_coupon_pct(body.get("referral_code"), email)
            try:
                pc = _price_cart(cart, ship=ship, coupon_pct=_ref_pct,
                                 points_to_redeem_cents=requested_redeem)
            except CheckoutError as e:
                return jsonify({"ok": False, "error": str(e)}), 400
```

After `_ingest_order(source="reorder", external_ref=inv.get("Id"), ...)`:

```python
            _record_referral_if_any(_ref_ctx, email, inv.get("Id"))
```

(Confirm the request-body variable name at each site: begin-buy reads `data`, reorder reads `body`. Read both routes to confirm the cart/email locals are in scope; do not change any other behavior.)

- [ ] **Step 4: Run to verify they pass**

Run the test command, then the broader sweep:
`doppler run -p remedy-match -c prd -- env DATA_DIR=$(mktemp -d) ~/.venvs/deploy-chat311/bin/python -m pytest tests/ -k "referral or reorder or begin" -q`
Expected: all pass, no regressions in the existing checkout tests.

- [ ] **Step 5: Commit**

```bash
git add app.py tests/test_referrals_spec2b1.py
git commit -m "feat(referrals-2b1): wire referral code into begin-buy + reorder checkout + record redemption"
```

---

## Manual Verification (after Task 3)

With `REFERRALS=true` locally: `GET /api/referral/my-code` (as a signed-in email) returns a stable code. A referee checkout passing `referral_code` prices at `max(10%, daily)` (capped by floors), records one redemption; a second checkout by the same referee gets no referral discount; a self-referral checkout gets the normal price. Front-end: show the code on the reorder/account page + highlight it on review approval. NO emoji.

## Self-Review (plan author)

- **Spec coverage:** store + resolve/record (T1) → spec Store; flag + pct + resolve helper + my-code (T2) → spec Discount/Retrieval; checkout integration + recording (T3) → spec Checkout integration. Flag `REFERRALS` woven through T2/T3.
- **Decisions honored:** 10% via `_referral_pct` (T2); `max(referral, daily)` best-of-one into `pricing.compute` (T2/T3, no pricing change); one-per-referee via PK + `has_redeemed` (T1); self-referral block (T1); evergreen codes (T1); flag-off inert (T2/T3); single-sided (no referrer reward); referral never blocks checkout (wrapped, T2/T3).
- **Type consistency:** `referrals.*` signatures, `_resolve_checkout_coupon_pct -> (pct, ctx)`, `ctx = {code, owner_email}`, `_record_referral_if_any(ctx, referee_email, order_ref)` — used identically across tasks.
- **Confirm-then-use flagged in-task:** the begin-buy vs reorder request-body var (`data` vs `body`) + local scope (T3); both sites already pass `coupon_pct=_active_coupon_pct()` so the swap is mechanical.
