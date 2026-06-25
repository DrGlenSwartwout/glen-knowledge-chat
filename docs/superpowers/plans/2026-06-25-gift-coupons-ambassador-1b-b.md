# Gift Coupons & Ambassador Gifting (1b-B) — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Give each collected remedy a per-product **gift coupon** a visitor can unlock (instant self-serve "gifting" activation) and share, so a friend gets 15% off that product and the redemption is attributed back to the gifter via the existing referral rails.

**Architecture:** Extend the 1b-A `dashboard/coupons.py` registry with `kind='gift'` minting + validation (redeemable by a *different* email than the owner). Add an instant gifting-activation endpoint that sets `affiliate_signups.gifting_activated_at` with `status='pending'` (commission stays gated on the unchanged `_is_ambassador`/`status='approved'`). A friend's gift redemption applies via the existing floored `coupon_pct` checkout path and records a `referrals.record_redemption`. Ships dark behind a new flag.

**Tech Stack:** Python 3 / Flask 3.1, sqlite, vanilla JS, pytest. Reused: `dashboard/coupons.py` (1b-A), `dashboard/referrals.py`, `begin_funnel` (email/TOS gate), the 1b-A injected shell + wallet, `pricing.compute` floor.

## Global Constraints

- New flag **`REWARDS_1B_GIFT_ENABLED`** (Doppler), **dark by default**, independent of `REWARDS_1B_ENABLED` and `JOURNEY_SHELL_ENABLED`.
- Gift coupon: **15% off**, **single-use**, **30-day** window, **`kind='gift'`**, code prefix `GIFT-`. **One gift coupon per (owner email, product_slug)** for life (mirrors self earn-once).
- Gift coupons are **owned by the gifter** (`email`=owner) but **redeemable by any *other* email** — `validate_gift` rejects `referee_email == owner` (no self-gift).
- Discount applies ONLY through the existing `pricing.compute(coupon_pct=)` path (57% wholesale floor); non-stacking `max()` with referral/self.
- Activation requires the funnel **email+TOS** gate. It must set `status='pending'` **explicitly** (the `affiliate_signups` table defaults `status` to `'approved'`) so **`_is_ambassador` stays False** — gifting ≠ commission.
- **Do NOT modify** `_is_ambassador`, the referral resolution path, `points.py`, or the `begin_funnel` engine.
- Gift coupons share the `coupons` table + owner email with self-coupons → `wallet()` and self auto-apply MUST be **kind-aware** so gifts never leak into self-coupon logic.
- Local test command (app-importing tests): `mkdir -p /tmp/jshell-test && doppler run -p remedy-match -c prd -- env DATA_DIR=/tmp/jshell-test python3 -m pytest …` (work in the worktree, substitute its path for `~/deploy-chat`).

---

### Task 1: Gift coupons + kind-aware wallet (`dashboard/coupons.py`)

**Files:**
- Modify: `dashboard/coupons.py`
- Test: `tests/test_coupons_gift.py`

**Interfaces:**
- Consumes: existing `_SEL`, `_row`, `_now` in `coupons.py`.
- Produces:
  - `mint_gift(cx, *, email, product_slug, pct=15, days=30) -> dict`
  - `validate_gift(cx, code, *, referee_email) -> dict | None`
  - `wallet(cx, *, email, kind=None) -> list[dict]` (adds optional `kind` filter; `kind=None` = all kinds, backward-compatible)

- [ ] **Step 1: Write the failing test**

```python
# tests/test_coupons_gift.py
import sqlite3
import pytest
from dashboard import coupons


@pytest.fixture
def cx(tmp_path):
    c = sqlite3.connect(str(tmp_path / "t.db"))
    coupons.init_coupons_table(c)
    return c


def test_mint_gift_idempotent_and_prefixed(cx):
    a = coupons.mint_gift(cx, email="owner@x.com", product_slug="terrain-restore")
    b = coupons.mint_gift(cx, email="owner@x.com", product_slug="terrain-restore")
    assert a["code"] == b["code"]
    assert a["kind"] == "gift" and a["code"].startswith("GIFT-") and a["pct"] == 15


def test_validate_gift_ok_for_other_email(cx):
    c = coupons.mint_gift(cx, email="owner@x.com", product_slug="terrain-restore")
    assert coupons.validate_gift(cx, c["code"], referee_email="friend@y.com")


def test_validate_gift_blocks_self_gift(cx):
    c = coupons.mint_gift(cx, email="owner@x.com", product_slug="terrain-restore")
    assert coupons.validate_gift(cx, c["code"], referee_email="OWNER@x.com") is None  # case-insensitive


def test_validate_gift_rejects_self_kind_code(cx):
    s = coupons.mint_self(cx, email="owner@x.com", product_slug="terrain-restore")
    assert coupons.validate_gift(cx, s["code"], referee_email="friend@y.com") is None


def test_validate_gift_expired_and_redeemed(cx):
    dead = coupons.mint_gift(cx, email="o@x.com", product_slug="p", days=-1)
    assert coupons.validate_gift(cx, dead["code"], referee_email="f@y.com") is None
    live = coupons.mint_gift(cx, email="o@x.com", product_slug="q")
    coupons.mark_redeemed(cx, live["code"], order_ref="INV-1")
    assert coupons.validate_gift(cx, live["code"], referee_email="f@y.com") is None


def test_wallet_kind_filter(cx):
    coupons.mint_self(cx, email="o@x.com", product_slug="a")
    coupons.mint_gift(cx, email="o@x.com", product_slug="a")
    assert {c["kind"] for c in coupons.wallet(cx, email="o@x.com")} == {"self", "gift"}
    assert {c["kind"] for c in coupons.wallet(cx, email="o@x.com", kind="self")} == {"self"}
    assert {c["kind"] for c in coupons.wallet(cx, email="o@x.com", kind="gift")} == {"gift"}
```

- [ ] **Step 2: Run to verify it fails**

Run: `cd /tmp/wt-deploy-chat-6a686b75 && python3 -m pytest tests/test_coupons_gift.py -q`
Expected: FAIL — `mint_gift`/`validate_gift` undefined; `wallet` has no `kind` kwarg.

- [ ] **Step 3: Add `mint_gift` + `validate_gift`, and a `kind` param to `wallet`**

Append after `mark_redeemed` in `dashboard/coupons.py`:

```python
def mint_gift(cx, *, email, product_slug, pct=15, days=30):
    """One giftable coupon per (owner email, product_slug) for life. Owned by the
    gifter; redeemable by a different email. Idempotent."""
    now = _now()
    existing = _row(cx.execute(
        f"SELECT {_SEL} FROM coupons WHERE email=? AND product_slug=? AND kind='gift' "
        "ORDER BY minted_at DESC LIMIT 1", (email, product_slug)).fetchone())
    if existing:
        return existing
    code = "GIFT-" + uuid.uuid4().hex[:8].upper()
    expires = (datetime.strptime(now, "%Y-%m-%d %H:%M:%S")
               + timedelta(days=days)).strftime("%Y-%m-%d %H:%M:%S")
    cx.execute("INSERT INTO coupons(code,product_slug,pct,kind,email,minted_at,expires_at) "
               "VALUES (?,?,?,?,?,?,?)",
               (code, product_slug, int(pct), "gift", email, now, expires))
    cx.commit()
    return _row(cx.execute(f"SELECT {_SEL} FROM coupons WHERE code=?", (code,)).fetchone())


def validate_gift(cx, code, *, referee_email):
    """Valid gift coupon redeemable by referee_email (not the owner). None otherwise."""
    now = _now()
    r = _row(cx.execute(
        f"SELECT {_SEL} FROM coupons WHERE code=? AND kind='gift' AND redeemed_at IS NULL "
        "AND expires_at > ?", ((code or "").strip(), now)).fetchone())
    if not r:
        return None
    if (referee_email or "").strip().lower() == (r["email"] or "").strip().lower():
        return None  # no self-gift
    return r
```

Replace the existing `wallet` function with a kind-aware version:

```python
def wallet(cx, *, email, kind=None):
    now = _now()
    q = f"SELECT {_SEL} FROM coupons WHERE email=? AND redeemed_at IS NULL AND expires_at > ? "
    params = [email, now]
    if kind:
        q += "AND kind=? "
        params.append(kind)
    q += "ORDER BY expires_at ASC"
    rows = cx.execute(q, params).fetchall()
    return [_row(r) for r in rows]
```

- [ ] **Step 4: Run to verify pass**

Run: `cd /tmp/wt-deploy-chat-6a686b75 && python3 -m pytest tests/test_coupons_gift.py tests/test_coupons.py -q`
Expected: PASS (new gift tests + the existing self-coupon tests still pass — `wallet(email=…)` with no `kind` is unchanged).

- [ ] **Step 5: Commit**

```bash
git add dashboard/coupons.py tests/test_coupons_gift.py
git commit -m "feat(rewards): gift coupons (mint_gift/validate_gift) + kind-aware wallet (1b-B)"
```

---

### Task 2: Gifting activation — schema + endpoint + flag

**Files:**
- Modify: `app.py` (add `gifting_activated_at` migration to the `affiliate_signups` init; `REWARDS_1B_GIFT_ENABLED` flag; `shell_nav.inject_shell_html` rewardsGift flag; `POST /api/journey/activate-gifting`; helper `_gifting_active`)
- Modify: `shell_nav.py` (`inject_shell_html` carries `rewards_gift`)
- Test: `tests/test_gifting_activate.py`

**Interfaces:**
- Consumes: `begin_funnel.get_state`; `coupons.mint_gift`, `coupons.wallet`; `coupons.init_coupons_table`.
- Produces: `POST /api/journey/activate-gifting` → `{ok, gifting:true, gifts:[…]}` or `409 {needs:"email_tos"}`; `_gifting_active(cx, email) -> bool`; `window.__SHELL__.rewardsGift` boolean.

- [ ] **Step 1: Add the `gifting_activated_at` column** to the `affiliate_signups` init in `app.py` (find the `CREATE TABLE IF NOT EXISTS affiliate_signups` block near line 6765; after it, add an idempotent migration alongside the other `ALTER TABLE` migrations in that init function):

```python
            try:
                cx.execute("ALTER TABLE affiliate_signups ADD COLUMN gifting_activated_at TEXT")
            except Exception:
                pass  # already present
```

- [ ] **Step 2: Add the flag + carry it in the shell injection**

In `app.py`, beside `REWARDS_1B_ENABLED`:

```python
REWARDS_1B_GIFT_ENABLED = os.environ.get("REWARDS_1B_GIFT_ENABLED", "").strip().lower() in ("1", "true", "yes", "on")
```

In `shell_nav.py`, extend `inject_shell_html` to carry a third flag:

```python
def inject_shell_html(html: str, mode: str, rewards1b: bool = False, rewards_gift: bool = False) -> str:
    """Insert the shell <link>+<script> tags before </head>. Idempotent; no-op when no </head>."""
    if _MARKER in (html or ""):
        return html
    if "</head>" not in html:
        return html
    mode = "member" if mode == "member" else "funnel"
    r1 = "true" if rewards1b else "false"
    rg = "true" if rewards_gift else "false"
    tags = (
        f'<link {_MARKER} rel="stylesheet" href="/static/shell.css">'
        f'<script>window.__SHELL__={{"mode":"{mode}","rewards1b":{r1},"rewardsGift":{rg}}};</script>'
        f'<script defer src="/static/shell.js"></script>'
    )
    return html.replace("</head>", tags + "\n</head>", 1)
```

And in `app.py`'s `_inject_journey_shell`, pass it:

```python
        response.set_data(shell_nav.inject_shell_html(html, mode, REWARDS_1B_ENABLED, REWARDS_1B_GIFT_ENABLED))
```

(The existing 1a inject test checks the `"mode":"funnel"` substring, still present.)

- [ ] **Step 3: Write the failing endpoint test**

```python
# tests/test_gifting_activate.py
import sqlite3, json
import pytest


@pytest.fixture
def client(monkeypatch, tmp_path):
    import app as appmod
    monkeypatch.setattr(appmod, "LOG_DB", str(tmp_path / "chat_log.db"))
    monkeypatch.setattr(appmod, "REWARDS_1B_GIFT_ENABLED", True)
    import begin_funnel
    from dashboard import coupons
    with sqlite3.connect(appmod.LOG_DB) as cx:
        begin_funnel.init_journey_tables(cx)
        coupons.init_coupons_table(cx)
        appmod._init_affiliate_tables() if hasattr(appmod, "_init_affiliate_tables") else None
    appmod.app.config["TESTING"] = True
    return appmod.app.test_client(), appmod, begin_funnel


def _seed_session(bf, log_db, sid, *, email=None, tos=False, gates=()):
    with sqlite3.connect(log_db) as cx:
        cx.execute(
            "INSERT INTO journey_state(session_id,email,unlocked_gates,tos_agreed_at,created_at,updated_at) "
            "VALUES (?,?,?,?,?,?)",
            (sid, email or "", json.dumps(sorted(gates)),
             "2026-06-25T00:00:00" if tos else None, "2026-06-25T00:00:00", "2026-06-25T00:00:00"))
        cx.commit()


def test_activate_requires_email_tos(client):
    c, appmod, bf = client
    _seed_session(bf, appmod.LOG_DB, "g1", email=None, tos=False)
    c.set_cookie("amg_session", "g1")
    r = c.post("/api/journey/activate-gifting", json={})
    assert r.status_code == 409 and r.get_json()["needs"] == "email_tos"


def test_activate_sets_gifting_but_not_ambassador(client):
    c, appmod, bf = client
    _seed_session(bf, appmod.LOG_DB, "g2", email="me@x.com", tos=True)
    c.set_cookie("amg_session", "g2")
    r = c.post("/api/journey/activate-gifting", json={})
    assert r.status_code == 200 and r.get_json()["gifting"] is True
    with sqlite3.connect(appmod.LOG_DB) as cx:
        row = cx.execute("SELECT status, gifting_activated_at FROM affiliate_signups WHERE email='me@x.com'").fetchone()
        assert row and row[0] == "pending" and row[1]  # pending + activated timestamp
        assert appmod._is_ambassador(cx, "me@x.com") is False  # commission still gated


def test_activate_flag_off_404(client):
    c, appmod, bf = client
    appmod.REWARDS_1B_GIFT_ENABLED = False
    _seed_session(bf, appmod.LOG_DB, "g3", email="me@x.com", tos=True)
    c.set_cookie("amg_session", "g3")
    assert c.post("/api/journey/activate-gifting", json={}).status_code == 404
```

- [ ] **Step 4: Run to verify it fails**

Run: `mkdir -p /tmp/jshell-test && cd /tmp/wt-deploy-chat-6a686b75 && doppler run -p remedy-match -c prd -- env DATA_DIR=/tmp/jshell-test python3 -m pytest tests/test_gifting_activate.py -q`
Expected: FAIL — route 404 (not yet defined).

- [ ] **Step 5: Implement the helper + endpoint in `app.py`** (place the route after `journey_wallet`, ~line 2238):

```python
def _gifting_active(cx, email):
    if not email:
        return False
    try:
        row = cx.execute("SELECT gifting_activated_at FROM affiliate_signups WHERE LOWER(email)=?",
                         (email.lower(),)).fetchone()
        return bool(row and row[0])
    except Exception:
        return False


@app.route("/api/journey/activate-gifting", methods=["POST"])
def journey_activate_gifting():
    if not REWARDS_1B_GIFT_ENABLED:
        return ("", 404)
    session_id = (request.cookies.get("amg_session") or "").strip()
    au = get_authenticated_user(request)
    email = (au["email"] if au else "").strip()
    with _db_lock, sqlite3.connect(LOG_DB) as cx:
        state = begin_funnel.get_state(cx, session_id=session_id, email=email)
        email = (state.get("email") or email or "").strip().lower()
        if not email or not state.get("tos_agreed_at"):
            return jsonify({"ok": False, "needs": "email_tos"}), 409
        now = begin_funnel._now()
        row = cx.execute("SELECT id FROM affiliate_signups WHERE LOWER(email)=?", (email,)).fetchone()
        if row:
            cx.execute("UPDATE affiliate_signups SET gifting_activated_at=? WHERE id=?", (now, row[0]))
        else:
            nm = ((state.get("first_name") or "") + " " + (state.get("last_name") or "")).strip() or email
            base = email.split("@")[0]
            slug = f"gift-{uuid.uuid4().hex[:8]}"
            token = uuid.uuid4().hex
            cx.execute(
                "INSERT INTO affiliate_signups (created_at,name,email,slug,token,status,gifting_activated_at) "
                "VALUES (?,?,?,?,?,?,?)", (now, nm, email, slug, token, "pending", now))
        cx.commit()
        # mint gift twins for every remedy the visitor has already collected (self-coupons)
        from dashboard import coupons as _coupons
        _coupons.init_coupons_table(cx)
        gifts = []
        for sc in _coupons.wallet(cx, email=email, kind="self"):
            gifts.append(_coupons.mint_gift(cx, email=email, product_slug=sc["product_slug"]))
    return jsonify({"ok": True, "gifting": True, "gifts": gifts})
```

> If `uuid` isn't imported at app.py top, it is (used elsewhere). Confirm with `grep -n "^import uuid" app.py`.

- [ ] **Step 6: Run to verify pass**

Run: `cd /tmp/wt-deploy-chat-6a686b75 && doppler run -p remedy-match -c prd -- env DATA_DIR=/tmp/jshell-test python3 -m pytest tests/test_gifting_activate.py tests/test_journey_shell_inject.py -q`
Expected: PASS (activation gating + the 1a inject tests unaffected).

- [ ] **Step 7: Commit**

```bash
git add app.py shell_nav.py tests/test_gifting_activate.py
git commit -m "feat(rewards): instant gifting activation (pending, not approved) + gift flag (1b-B)"
```

---

### Task 3: Wallet (gift) + checkout gift redemption + attribution

**Files:**
- Modify: `app.py` (`journey_wallet` returns gift coupons + share URLs when gifting active; `_best_active_self_coupon_code` filters `kind='self'`; `begin_checkout` resolves a gift code → pct + records redemption)
- Test: `tests/test_gift_checkout.py`

**Interfaces:**
- Consumes: `coupons.validate_gift`, `coupons.mark_redeemed`, `coupons.wallet(kind=)`; `referrals.record_redemption`.
- Produces: `_resolve_gift_coupon_pct(code, referee_email) -> (pct:int, coupon:dict|None)`.

- [ ] **Step 1: Write the failing test**

```python
# tests/test_gift_checkout.py
import sqlite3
import pytest


@pytest.fixture
def appmod(monkeypatch, tmp_path):
    import app as appmod
    monkeypatch.setattr(appmod, "LOG_DB", str(tmp_path / "chat_log.db"))
    monkeypatch.setattr(appmod, "REWARDS_1B_GIFT_ENABLED", True)
    from dashboard import coupons
    with sqlite3.connect(appmod.LOG_DB) as cx:
        coupons.init_coupons_table(cx)
    return appmod


def test_resolve_gift_pct_and_self_block(appmod):
    from dashboard import coupons
    with sqlite3.connect(appmod.LOG_DB) as cx:
        g = coupons.mint_gift(cx, email="owner@x.com", product_slug="terrain-restore")
    pct, found = appmod._resolve_gift_coupon_pct(g["code"], "friend@y.com")
    assert pct == 15 and found and found["code"] == g["code"]
    # owner cannot redeem their own gift
    assert appmod._resolve_gift_coupon_pct(g["code"], "owner@x.com")[0] == 0
    assert appmod._resolve_gift_coupon_pct("nope", "friend@y.com")[0] == 0


def test_self_autoapply_ignores_gift_coupons(appmod):
    """A gift coupon owned by the same email must NOT be auto-applied as a self discount."""
    from dashboard import coupons
    with sqlite3.connect(appmod.LOG_DB) as cx:
        coupons.mint_gift(cx, email="me@x.com", product_slug="terrain-restore")  # only a gift, no self
    assert appmod._best_active_self_coupon_code("me@x.com", "terrain-restore") == ""
```

- [ ] **Step 2: Run to verify it fails**

Run: `cd /tmp/wt-deploy-chat-6a686b75 && doppler run -p remedy-match -c prd -- env DATA_DIR=/tmp/jshell-test python3 -m pytest tests/test_gift_checkout.py -q`
Expected: FAIL — `_resolve_gift_coupon_pct` undefined; `_best_active_self_coupon_code` currently returns the gift code (leak).

- [ ] **Step 3: Fix the self-leak + add the gift resolver in `app.py`**

In `_best_active_self_coupon_code` (~line 10216), make the wallet lookup kind-aware:

```python
            actives = [c for c in _coupons.wallet(cx, email=email, kind="self")
                       if c.get("product_slug") == product_slug]
```

Add the gift resolver next to `_resolve_self_coupon_pct` (~line 10233):

```python
def _resolve_gift_coupon_pct(code, referee_email):
    """A friend's gift coupon → (pct, coupon|None). Product-agnostic at the code level
    (the code is product-bound); never raises; flag-gated."""
    code = (code or "").strip()
    if not REWARDS_1B_GIFT_ENABLED or not code:
        return 0, None
    try:
        from dashboard import coupons as _coupons
        with sqlite3.connect(LOG_DB) as cx:
            _coupons.init_coupons_table(cx)
            found = _coupons.validate_gift(cx, code, referee_email=referee_email)
        return (int(found["pct"]), found) if found else (0, None)
    except Exception as e:  # noqa: BLE001 — never block checkout
        print(f"[coupons] gift resolve failed: {e!r}", flush=True)
        return 0, None
```

- [ ] **Step 4: Thread the gift code into `begin_checkout`** (~line 5213). Replace the self-coupon resolution + combine block with one that also resolves a gift code and combines all three via `max()`:

```python
        _coupon_code = (data.get("coupon_code") or "").strip() or _best_active_self_coupon_code(email, slug)
        _self_pct, _self_coupon = _resolve_self_coupon_pct(_coupon_code, slug)
        _gift_pct, _gift_coupon = _resolve_gift_coupon_pct((data.get("gift") or "").strip(), email)
        # a gift code is for ONE product — only honor it when buying that product
        if _gift_coupon and _gift_coupon.get("product_slug") != slug:
            _gift_pct, _gift_coupon = 0, None
        _eff_pct = max(_ref_pct or 0, _self_pct or 0, _gift_pct or 0)
        try:
            pc = _price_cart([{"slug": slug, "qty": qty}], ship=ship,
                             coupon_pct=_eff_pct,
                             points_to_redeem_cents=redeem)
        except CheckoutError as ce:
            return jsonify({"ok": False, "error": str(ce)}), 400
```

Then, after the invoice exists and the existing self-coupon redemption block (~line 5232), add the gift redemption + attribution:

```python
        if _gift_coupon and _gift_pct >= max(_ref_pct or 0, _self_pct or 0):
            try:
                from dashboard import coupons as _coupons
                from dashboard import referrals as _rf
                with _db_lock, sqlite3.connect(LOG_DB) as _gcx:
                    _coupons.mark_redeemed(_gcx, _gift_coupon["code"], order_ref=inv.get("Id"))
                    _rf.record_redemption(_gcx, _gift_coupon["code"], _gift_coupon["email"], email, inv.get("Id"))
            except Exception as e:  # noqa: BLE001
                print(f"[coupons] gift redeem/attrib failed: {e!r}", flush=True)
```

- [ ] **Step 5: Extend `journey_wallet`** (~line 2238) to add gift coupons + share URLs when gifting is active. After it builds the self `items`, add:

```python
        gifts = []
        if _gifting_active(cx, email):
            for g in _coupons.wallet(cx, email=email, kind="gift"):
                g = dict(g)
                g["share_url"] = f"/begin/buy/{g['product_slug']}?gift={g['code']}"
                gifts.append(g)
    return jsonify({"ok": True, "coupons": items, "gifts": gifts, "gifting": _gifting_active_flag})
```

> Adjust to the real structure of `journey_wallet`: compute `_gifting_active_flag = _gifting_active(cx, email)` inside the `with` block, build `gifts` there, and return `coupons` + `gifts` + `gifting`. Keep the existing `coupons` (self) key unchanged so 1b-A's wallet rendering is unaffected.

- [ ] **Step 6: Run the tests to verify pass + no regression**

Run: `cd /tmp/wt-deploy-chat-6a686b75 && doppler run -p remedy-match -c prd -- env DATA_DIR=/tmp/jshell-test python3 -m pytest tests/test_gift_checkout.py tests/test_rewards_checkout.py tests/test_rewards_claim.py -q`
Expected: PASS — gift resolver + self-leak fix work; the 1b-A self-coupon checkout/claim/wallet tests still pass.

- [ ] **Step 7: Commit**

```bash
git add app.py tests/test_gift_checkout.py
git commit -m "feat(rewards): gift redemption at checkout + attribution, wallet gifts, self-leak fix (1b-B)"
```

---

### Task 4: Client — Beacon CTA, wallet gifts + Share, `?gift=` capture

**Files:**
- Modify: `static/shell.js` (gift UI gated on `window.__SHELL__.rewardsGift`)
- Modify: `static/begin-buy.html` (capture `?gift=` → POST as `gift`)
- Test: manual/visual QA (controller runs the live render).

**Interfaces:**
- Consumes: `window.__SHELL__.rewardsGift`; `POST /api/journey/activate-gifting`; `GET /api/journey/wallet` (now `{coupons, gifts, gifting}`).

- [ ] **Step 1: Add gift logic to `static/shell.js`** — add `var REWARDS_GIFT = !!(window.__SHELL__ && window.__SHELL__.rewardsGift);` near the existing `REWARDS` const. In `buildOverlay`, in the `give` land card (the one with no `meta.featured`), when `REWARDS_GIFT`, add an **"Unlock gifting"** button that POSTs `/api/journey/activate-gifting` (on `409 needs:email_tos` → `location.href="/begin/match"`; on 200 → `refreshWallet()`):

```javascript
      if (REWARDS_GIFT && card.key === "give") {
        var gb = el("button", "js-claim", "Unlock gifting");
        gb.onclick = function () {
          gb.disabled = true; gb.textContent = "Activating…";
          fetch("/api/journey/activate-gifting", {method: "POST", credentials: "same-origin",
            headers: {"Content-Type": "application/json"}, body: "{}"})
            .then(function (r) { return r.json().then(function (j) { return {s: r.status, j: j}; }); })
            .then(function (res) {
              if (res.s === 200) { gb.textContent = "✓ Gifting unlocked"; refreshWallet(); }
              else if (res.j && res.j.needs === "email_tos") { location.href = "/begin/match"; }
              else { gb.disabled = false; gb.textContent = "Unlock gifting"; }
            }).catch(function () { gb.disabled = false; gb.textContent = "Unlock gifting"; });
        };
        box.appendChild(gb);
      }
```

In `refreshWallet`, after rendering self `coupons`, render `gifts` (each with a Share button copying `c.share_url` as an absolute link):

```javascript
        var gifts = (j && j.gifts) || [];
        gifts.forEach(function (g) {
          var row = el("div", "js-wallet-coupon",
            "<b>Gift 15% off</b> " + g.product_slug + " <span class='js-exp'>to a friend</span>");
          var share = el("button", "js-claim", "Share");
          share.onclick = function () {
            var url = location.origin + g.share_url;
            (navigator.clipboard ? navigator.clipboard.writeText(url) : Promise.reject())
              .then(function () { share.textContent = "✓ Link copied"; })
              .catch(function () { share.textContent = url; });
          };
          row.appendChild(share);
          panel.appendChild(row);
        });
```

- [ ] **Step 2: Capture `?gift=` in `static/begin-buy.html`** — find where the checkout payload is built (the `fetch`/POST to `/begin/checkout/<slug>`; grep `referral_code` in the file) and add the gift code from the URL to the payload:

```javascript
      // near where the payload object is built (alongside referral_code):
      var _gift = new URLSearchParams(location.search).get("gift");
      if (_gift) payload.gift = _gift;
```

- [ ] **Step 3: Verify**

Run: `cd /tmp/wt-deploy-chat-6a686b75 && node --check static/shell.js && echo "shell.js OK"`
Then the controller starts the app with `JOURNEY_SHELL_ENABLED=1 REWARDS_1B_ENABLED=1 REWARDS_1B_GIFT_ENABLED=1` and confirms: the Beacon land shows "Unlock gifting"; after activating (with a seeded email+TOS session) the wallet shows gift coupons with Share; a `?gift=<code>` link is captured into the buy payload.

- [ ] **Step 4: Commit**

```bash
git add static/shell.js static/begin-buy.html
git commit -m "feat(rewards): client — unlock-gifting CTA, gift wallet + share, ?gift= capture (1b-B)"
```

---

### Task 5: Integration smoke + full suite (PR by controller)

**Files:** none (verification only).

- [ ] **Step 1: Full 1b-A + 1b-B + 1a suite green**

Run: `cd /tmp/wt-deploy-chat-6a686b75 && mkdir -p /tmp/jshell-test && doppler run -p remedy-match -c prd -- env DATA_DIR=/tmp/jshell-test python3 -m pytest tests/test_coupons.py tests/test_coupons_gift.py tests/test_gifting_activate.py tests/test_gift_checkout.py tests/test_rewards_claim.py tests/test_rewards_checkout.py tests/test_rewards_console.py tests/test_shell_map_config.py tests/test_journey_shell_inject.py -q`
Expected: all green.

- [ ] **Step 2: Dark-by-default check**

With `REWARDS_1B_GIFT_ENABLED` unset: `POST /api/journey/activate-gifting` → 404; `window.__SHELL__.rewardsGift` is `false`; no gift UI; `GET /api/journey/wallet` returns `gifts: []` (or omits gifts) and `gifting:false`. 1b-A self-coupons + 1a shell unaffected.

- [ ] **Step 3: Console** — confirm the existing `GET /api/console/coupons` lists gift coupons too (it SELECTs all kinds; the `kind` column distinguishes them). No code change needed.

- [ ] **Step 4: Report** to the controller — the controller runs the final whole-branch review and opens the PR (do NOT push/PR from this task).

---

## Self-Review

**Spec coverage:**
- Gift coupon mint/validate (kind='gift', redeemable by other email, no self-gift) → Task 1. ✓
- Kind-aware wallet so gifts don't leak into self logic → Task 1 (`wallet(kind=)`) + Task 3 (`_best_active_self_coupon_code` fix + test). ✓
- Instant gifting activation, `status='pending'`, `_is_ambassador` unchanged → Task 2. ✓
- `REWARDS_1B_GIFT_ENABLED` flag, dark, independent; client `rewardsGift` → Task 2. ✓
- Wallet shows gift coupons + share URL when gifting active → Task 3 (endpoint) + Task 4 (render). ✓
- Checkout: gift code → floored pct, mark redeemed, **record referral_redemption** (attribution) → Task 3. ✓
- Beacon "Unlock gifting" CTA + `?gift=` capture → Task 4. ✓
- Single-use / 30-day / one-per-(owner,product) / 15% → Task 1 (`mint_gift` defaults + idempotency). ✓
- Don't touch `_is_ambassador`/referral path/`points.py`/`begin_funnel` → reuse only; referral path unchanged (gift adds a parallel resolver + a `record_redemption` call). ✓
- Console lists gifts → Task 5 (existing route, no change). ✓

**Placeholder scan:** none — concrete code/commands throughout. Two "adjust to real structure" notes (the `journey_wallet` return shape in Task 3 Step 5; the `begin-buy.html` payload object in Task 4 Step 2) name the exact grep/anchor and the contract to satisfy.

**Type consistency:** `mint_gift`/`validate_gift`/`wallet(kind=)` signatures consistent Task 1↔2↔3. `_resolve_gift_coupon_pct(code, referee) -> (int, dict|None)` consistent Task 3. `inject_shell_html(html, mode, rewards1b=False, rewards_gift=False)` updated in Task 2 and its only caller (`after_request`) updated same task. Client reads `window.__SHELL__.rewardsGift` (Task 2 emits, Task 4 consumes); wallet `{coupons, gifts, gifting}` produced Task 3, consumed Task 4.
