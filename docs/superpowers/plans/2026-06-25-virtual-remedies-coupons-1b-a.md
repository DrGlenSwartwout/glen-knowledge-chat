# Virtual Remedies & Coupon Wallet (1b-A) — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make each journey land reveal a real "virtual remedy" (preframe), and let a visitor who completes a stage claim a 15%/10-day product **coupon** (email+TOS-gated) into a wallet, redeemable at checkout — with a biofield-orb avatar that brightens per coupon.

**Architecture:** One net-new subsystem — a `coupons` registry (`dashboard/coupons.py`) mirroring `points.py`/`referrals.py`. Everything else reuses existing rails: `begin_funnel` state (email/TOS gate + land status), `data/products.json`, the checkout `coupon_pct` path (`_resolve_checkout_coupon_pct` → `_price_cart` → `pricing.compute` with its 57% wholesale floor), and the injected shell from 1a. Ships dark behind a new flag.

**Tech Stack:** Python 3 / Flask 3.1, sqlite, vanilla JS, pytest. Reused: `begin_funnel.journey_map/get_state`, `dashboard/pricing.py`, `dashboard/products.py`, `shell_nav.py`, `static/shell.js`.

## Global Constraints

- New flag **`REWARDS_1B_ENABLED`** (Doppler `remedy-match/prd`), **dark by default**; independent of `JOURNEY_SHELL_ENABLED`.
- Coupon = **15% off, 10-day** window, **one product** per code, `kind='self'` for 1b-A.
- Discount is applied through the **existing** `pricing.compute(coupon_pct=…)` path — never below the **57% wholesale floor** (already enforced there). Do not add a second discount path.
- **Claiming requires the funnel email+TOS gate** (`get_state` → `email` and `tos_agreed_at` both present). No new account system.
- **Do NOT touch** `points.py`, the `begin_funnel` engine, or the referral path. `_resolve_checkout_coupon_pct` is *extended*, not rewritten.
- 1b-A covers lands **scan / find / heal** only (3 self-coupons). **give** (gift coupon + Ambassador) = 1b-B, out of scope.
- Local test command (app-importing tests): `doppler run -p remedy-match -c prd -- env DATA_DIR=/tmp/jshell-test python3 -m pytest …` (mkdir the dir first). Pure-module tests run with plain `python3 -m pytest`.

---

### Task 1: Coupon registry (`dashboard/coupons.py`)

**Files:**
- Create: `dashboard/coupons.py`
- Test: `tests/test_coupons.py`

**Interfaces:**
- Produces (all take an open sqlite `cx`; dict-returning, row_factory-independent):
  - `init_coupons_table(cx)`
  - `mint_self(cx, *, email, product_slug, pct=15, days=10) -> dict`
  - `validate(cx, code, *, product_slug=None) -> dict | None`
  - `mark_redeemed(cx, code, *, order_ref) -> bool`
  - `wallet(cx, *, email) -> list[dict]`

- [ ] **Step 1: Write the failing test**

```python
# tests/test_coupons.py
import sqlite3
import pytest
from dashboard import coupons


@pytest.fixture
def cx(tmp_path):
    c = sqlite3.connect(str(tmp_path / "t.db"))
    coupons.init_coupons_table(c)
    return c


def test_mint_self_is_idempotent_while_active(cx):
    a = coupons.mint_self(cx, email="m@x.com", product_slug="terrain-restore")
    b = coupons.mint_self(cx, email="m@x.com", product_slug="terrain-restore")
    assert a["code"] == b["code"]
    assert a["pct"] == 15 and a["kind"] == "self"


def test_validate_ok_and_product_mismatch(cx):
    c = coupons.mint_self(cx, email="m@x.com", product_slug="terrain-restore")
    assert coupons.validate(cx, c["code"], product_slug="terrain-restore")
    assert coupons.validate(cx, c["code"], product_slug="other-slug") is None


def test_validate_expired_is_none(cx):
    c = coupons.mint_self(cx, email="m@x.com", product_slug="x", days=-1)  # already expired
    assert coupons.validate(cx, c["code"]) is None


def test_mark_redeemed_then_invalid(cx):
    c = coupons.mint_self(cx, email="m@x.com", product_slug="x")
    assert coupons.mark_redeemed(cx, c["code"], order_ref="INV-1") is True
    assert coupons.mark_redeemed(cx, c["code"], order_ref="INV-1") is False  # idempotent
    assert coupons.validate(cx, c["code"]) is None


def test_wallet_lists_active_only(cx):
    coupons.mint_self(cx, email="m@x.com", product_slug="a")
    dead = coupons.mint_self(cx, email="m@x.com", product_slug="b", days=-1)
    w = coupons.wallet(cx, email="m@x.com")
    slugs = {r["product_slug"] for r in w}
    assert slugs == {"a"} and dead["product_slug"] == "b"
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cd ~/deploy-chat && python3 -m pytest tests/test_coupons.py -q`
Expected: FAIL — `ModuleNotFoundError: No module named 'dashboard.coupons'`.

- [ ] **Step 3: Write `dashboard/coupons.py`**

```python
# dashboard/coupons.py
"""1b product savings coupons. A coupon = a single % discount on ONE product,
time-limited. Self-coupons are minted on journey-stage completion and applied at
checkout via the EXISTING coupon_pct path (clamped to the wholesale floor in
dashboard.pricing.compute). Idempotent; safe under the app's single sqlite conn.
Reads build dicts by hand so they don't depend on the connection's row_factory."""
import uuid
from datetime import datetime, timedelta

_COLS = ["code", "product_slug", "pct", "kind", "email", "session_id",
         "minted_at", "expires_at", "redeemed_at", "order_ref"]
_SEL = ",".join(_COLS)


def _now():
    return datetime.utcnow().strftime("%Y-%m-%d %H:%M:%S")


def _row(r):
    return dict(zip(_COLS, r)) if r else None


def init_coupons_table(cx):
    cx.execute("""
        CREATE TABLE IF NOT EXISTS coupons (
            code TEXT PRIMARY KEY,
            product_slug TEXT NOT NULL,
            pct INTEGER NOT NULL,
            kind TEXT NOT NULL DEFAULT 'self',
            email TEXT,
            session_id TEXT,
            minted_at TEXT NOT NULL,
            expires_at TEXT NOT NULL,
            redeemed_at TEXT,
            order_ref TEXT
        )""")
    cx.execute("CREATE INDEX IF NOT EXISTS ix_coupons_email ON coupons(email)")
    cx.commit()


def mint_self(cx, *, email, product_slug, pct=15, days=10):
    now = _now()
    existing = _row(cx.execute(
        f"SELECT {_SEL} FROM coupons WHERE email=? AND product_slug=? AND kind='self' "
        "AND redeemed_at IS NULL AND expires_at > ? ORDER BY minted_at DESC LIMIT 1",
        (email, product_slug, now)).fetchone())
    if existing:
        return existing
    code = "SELF-" + uuid.uuid4().hex[:8].upper()
    expires = (datetime.strptime(now, "%Y-%m-%d %H:%M:%S")
               + timedelta(days=days)).strftime("%Y-%m-%d %H:%M:%S")
    cx.execute("INSERT INTO coupons(code,product_slug,pct,kind,email,minted_at,expires_at) "
               "VALUES (?,?,?,?,?,?,?)",
               (code, product_slug, int(pct), "self", email, now, expires))
    cx.commit()
    return _row(cx.execute(f"SELECT {_SEL} FROM coupons WHERE code=?", (code,)).fetchone())


def validate(cx, code, *, product_slug=None):
    now = _now()
    r = _row(cx.execute(
        f"SELECT {_SEL} FROM coupons WHERE code=? AND redeemed_at IS NULL AND expires_at > ?",
        ((code or "").strip(), now)).fetchone())
    if not r:
        return None
    if product_slug is not None and r["product_slug"] != product_slug:
        return None
    return r


def mark_redeemed(cx, code, *, order_ref):
    cur = cx.execute(
        "UPDATE coupons SET redeemed_at=?, order_ref=? WHERE code=? AND redeemed_at IS NULL",
        (_now(), str(order_ref), (code or "").strip()))
    cx.commit()
    return cur.rowcount > 0


def wallet(cx, *, email):
    now = _now()
    rows = cx.execute(
        f"SELECT {_SEL} FROM coupons WHERE email=? AND redeemed_at IS NULL AND expires_at > ? "
        "ORDER BY expires_at ASC", (email, now)).fetchall()
    return [_row(r) for r in rows]
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `cd ~/deploy-chat && python3 -m pytest tests/test_coupons.py -q`
Expected: PASS (5 tests).

- [ ] **Step 5: Commit**

```bash
git add dashboard/coupons.py tests/test_coupons.py
git commit -m "feat(rewards): coupon registry — mint/validate/redeem/wallet (1b-A)"
```

---

### Task 2: Featured config + claim/wallet endpoints + flag

**Files:**
- Modify: `static/shell-map.json` (add `featured` per land)
- Modify: `shell_nav.py` (`inject_shell_html` carries the `rewards1b` flag)
- Modify: `app.py` (flag `REWARDS_1B_ENABLED`; after_request passes `rewards1b`; two new routes)
- Test: `tests/test_rewards_claim.py`, and extend `tests/test_shell_map_config.py`

**Interfaces:**
- Consumes: `coupons.mint_self/wallet` (Task 1); `begin_funnel.get_state`, `begin_funnel.journey_map`.
- Produces: `POST /api/journey/claim-coupon` `{land}` → coupon JSON or `409 {needs:"email_tos"}`; `GET /api/journey/wallet` → `{coupons:[…]}`; `window.__SHELL__.rewards1b` boolean.

- [ ] **Step 1: Add `featured` to `static/shell-map.json`**

Edit each 1b-A land to add a `featured` object (give is left without one):

```json
{
  "lands": {
    "scan": {"name": "The Listening Pool", "category": "scan", "intrigue": "Your body is already speaking. Step in and listen.",
             "featured": {"product_slug": "terrain-restore", "product_name": "Terrain Restore", "healing_power": "supports your foundational terrain"}},
    "find": {"name": "The Hall of Mirrors", "category": "find", "intrigue": "See the one remedy your body is asking for.",
             "featured": {"product_slug": "nous-energy", "product_name": "Nous Energy", "healing_power": "supports clear, energized focus"}},
    "heal": {"name": "The Sanctuary", "category": "heal", "intrigue": "Where the root causes finally settle.",
             "featured": {"product_slug": "microbiome", "product_name": "Microbiome", "healing_power": "supports your gut and inner terrain"}},
    "give": {"name": "The Beacon", "category": "give", "intrigue": "the gift of healing"}
  },
  "categories": {
    "scan": {"icon": "🌀", "hue": "#4aa3a2"},
    "find": {"icon": "🔮", "hue": "#7a6cc4"},
    "heal": {"icon": "🌿", "hue": "#5aa36a"},
    "give": {"icon": "✨", "hue": "#caa64a"}
  }
}
```

> The 3 slugs are drafts. Before shipping, confirm each `product_slug` exists in `data/products.json` (the test below enforces it). If a slug differs, fix it here.

- [ ] **Step 2: Write the failing test for the featured-slug guard**

Append to `tests/test_shell_map_config.py`:

```python
def test_featured_slugs_exist_in_catalog():
    import json
    from dashboard import products as _products
    cfg = json.loads(CFG.read_text())
    catalog = set(_products.load_products().keys())
    missing = []
    for key, land in cfg["lands"].items():
        f = land.get("featured")
        if f and f["product_slug"] not in catalog:
            missing.append(f["product_slug"])
    assert missing == [], f"featured slugs not in products.json: {missing}"
```

- [ ] **Step 3: Run it — fix any slug mismatches**

Run: `cd ~/deploy-chat && python3 -m pytest tests/test_shell_map_config.py::test_featured_slugs_exist_in_catalog -q`
Expected: PASS once the three slugs match real catalog keys. If it FAILS, print candidates with
`python3 -c "from dashboard import products as p; print([k for k in p.load_products() if 'terrain' in k.lower() or 'magnesium' in k.lower() or 'endocrine' in k.lower()])"` and correct `shell-map.json`.

- [ ] **Step 4: Carry the `rewards1b` flag in `shell_nav.inject_shell_html`**

Replace the body of `inject_shell_html` in `shell_nav.py` so it accepts and emits the flag:

```python
def inject_shell_html(html: str, mode: str, rewards1b: bool = False) -> str:
    """Insert the shell <link>+<script> tags before </head>. Idempotent;
    no-op when there is no </head>."""
    if _MARKER in (html or ""):
        return html
    if "</head>" not in html:
        return html
    mode = "member" if mode == "member" else "funnel"
    flag = "true" if rewards1b else "false"
    tags = (
        f'<link {_MARKER} rel="stylesheet" href="/static/shell.css">'
        f'<script>window.__SHELL__={{"mode":"{mode}","rewards1b":{flag}}};</script>'
        f'<script defer src="/static/shell.js"></script>'
    )
    return html.replace("</head>", tags + "\n</head>", 1)
```

The existing test `test_inject_adds_assets_before_head_close` still passes (it checks the `"mode":"funnel"` substring, which remains).

- [ ] **Step 5: Add the flag + pass it from `after_request` in `app.py`**

Add beside `JOURNEY_SHELL_ENABLED`:

```python
REWARDS_1B_ENABLED = os.environ.get("REWARDS_1B_ENABLED", "").strip().lower() in ("1", "true", "yes", "on")
```

In `_inject_journey_shell`, change the inject call to pass the flag:

```python
        response.set_data(shell_nav.inject_shell_html(html, mode, REWARDS_1B_ENABLED))
```

- [ ] **Step 6: Write the failing endpoint tests**

```python
# tests/test_rewards_claim.py
"""1b-A claim + wallet endpoints. App-importing → run under doppler+DATA_DIR."""
import sqlite3
import pytest


@pytest.fixture
def client(monkeypatch, tmp_path):
    import app as appmod
    monkeypatch.setattr(appmod, "LOG_DB", str(tmp_path / "chat_log.db"))
    monkeypatch.setattr(appmod, "REWARDS_1B_ENABLED", True)
    import begin_funnel
    from dashboard import coupons
    with sqlite3.connect(appmod.LOG_DB) as cx:
        begin_funnel.init_journey_tables(cx)
        coupons.init_coupons_table(cx)
    appmod.app.config["TESTING"] = True
    return appmod.app.test_client(), appmod, begin_funnel


def _seed_session(begin_funnel, log_db, sid, *, email=None, tos=False, gates=()):
    """Seed a journey_state row directly (mirrors record_unlock's columns)."""
    import json
    with sqlite3.connect(log_db) as cx:
        cx.execute(
            "INSERT INTO journey_state(session_id,email,unlocked_gates,tos_agreed_at) VALUES (?,?,?,?)",
            (sid, email or "", json.dumps(sorted(gates)), "2026-06-25T00:00:00" if tos else None))
        cx.commit()


def test_claim_requires_email_and_tos(client):
    c, appmod, bf = client
    _seed_session(bf, appmod.LOG_DB, "s1", email=None, tos=False, gates=["scan", "course_ww"])
    c.set_cookie("amg_session", "s1")
    r = c.post("/api/journey/claim-coupon", json={"land": "scan"})
    assert r.status_code == 409 and r.get_json()["needs"] == "email_tos"


def test_claim_rejects_incomplete_stage(client):
    c, appmod, bf = client
    _seed_session(bf, appmod.LOG_DB, "s2", email="m@x.com", tos=True, gates=[])  # nothing done
    c.set_cookie("amg_session", "s2")
    r = c.post("/api/journey/claim-coupon", json={"land": "scan"})
    assert r.status_code == 409 and r.get_json()["needs"] == "complete_stage"


def test_claim_mints_when_eligible_and_wallet_lists_it(client):
    c, appmod, bf = client
    _seed_session(bf, appmod.LOG_DB, "s3", email="m@x.com", tos=True, gates=["scan", "course_ww"])
    c.set_cookie("amg_session", "s3")
    r = c.post("/api/journey/claim-coupon", json={"land": "scan"})
    assert r.status_code == 200
    body = r.get_json()
    assert body["coupon"]["pct"] == 15 and body["coupon"]["product_slug"]
    w = c.get("/api/journey/wallet").get_json()
    assert any(x["code"] == body["coupon"]["code"] for x in w["coupons"])
```

- [ ] **Step 7: Run the tests to verify they fail**

Run: `cd ~/deploy-chat && doppler run -p remedy-match -c prd -- env DATA_DIR=/tmp/jshell-test python3 -m pytest tests/test_rewards_claim.py -q` (mkdir `/tmp/jshell-test` first)
Expected: FAIL — routes return 404 (not yet defined).

- [ ] **Step 8: Implement the two routes in `app.py`**

Add near the other `/api/journey`-ish funnel routes (after `begin_card_click`). Helpers + routes:

```python
import json as _json_mod  # already imported as json at top; reuse the top-level json

def _featured_for_land(land):
    """Return (product_slug, product_name) for a land from shell-map.json, or (None, None)."""
    try:
        cfg = json.loads((STATIC / "shell-map.json").read_text())
        f = ((cfg.get("lands") or {}).get(land) or {}).get("featured") or {}
        return f.get("product_slug"), f.get("product_name")
    except Exception:
        return None, None


def _land_is_done(state, land):
    """True when the land's journey card is fully complete (fill>=1.0)."""
    for card in begin_funnel.journey_map(state, "", {}):
        if card["key"] == land:
            return card["status"] == "done"
    return False


@app.route("/api/journey/claim-coupon", methods=["POST"])
def journey_claim_coupon():
    if not REWARDS_1B_ENABLED:
        return ("", 404)
    data = request.get_json(silent=True) or {}
    land = (data.get("land") or "").strip()
    slug, _name = _featured_for_land(land)
    if not slug:
        return jsonify({"ok": False, "error": "no featured product for land"}), 400
    session_id = (request.cookies.get("amg_session") or "").strip()
    au = get_authenticated_user(request)
    email = (au["email"] if au else "").strip()
    with _db_lock, sqlite3.connect(LOG_DB) as cx:
        state = begin_funnel.get_state(cx, session_id=session_id, email=email)
        email = (state.get("email") or email or "").strip().lower()
        if not email or not state.get("tos_agreed_at"):
            return jsonify({"ok": False, "needs": "email_tos"}), 409
        if not _land_is_done(state, land):
            return jsonify({"ok": False, "needs": "complete_stage"}), 409
        from dashboard import coupons as _coupons
        _coupons.init_coupons_table(cx)
        coupon = _coupons.mint_self(cx, email=email, product_slug=slug)
    return jsonify({"ok": True, "coupon": coupon})


@app.route("/api/journey/wallet", methods=["GET"])
def journey_wallet():
    if not REWARDS_1B_ENABLED:
        return ("", 404)
    session_id = (request.cookies.get("amg_session") or "").strip()
    au = get_authenticated_user(request)
    email = (au["email"] if au else "").strip()
    with _db_lock, sqlite3.connect(LOG_DB) as cx:
        state = begin_funnel.get_state(cx, session_id=session_id, email=email)
        email = (state.get("email") or email or "").strip().lower()
        if not email:
            return jsonify({"ok": True, "coupons": []})
        from dashboard import coupons as _coupons
        _coupons.init_coupons_table(cx)
        items = _coupons.wallet(cx, email=email)
    return jsonify({"ok": True, "coupons": items})
```

- [ ] **Step 9: Run the endpoint + config tests to verify they pass**

Run: `cd ~/deploy-chat && doppler run -p remedy-match -c prd -- env DATA_DIR=/tmp/jshell-test python3 -m pytest tests/test_rewards_claim.py tests/test_shell_map_config.py tests/test_journey_shell_inject.py -q`
Expected: PASS (claim/wallet + featured-slug guard + 1a inject tests unaffected).

- [ ] **Step 10: Commit**

```bash
git add static/shell-map.json shell_nav.py app.py tests/test_rewards_claim.py tests/test_shell_map_config.py
git commit -m "feat(rewards): claim-coupon + wallet endpoints, featured config, rewards1b flag (1b-A)"
```

---

### Task 3: Checkout wiring — apply + redeem the self-coupon

**Files:**
- Modify: `app.py` (new `_resolve_self_coupon_pct` helper; thread it into `begin_checkout`)
- Test: `tests/test_rewards_checkout.py`

**Interfaces:**
- Consumes: `coupons.validate/mark_redeemed` (Task 1); `dashboard/pricing.py`.
- Produces: `_resolve_self_coupon_pct(code, slug) -> (pct:int, coupon:dict|None)`.

- [ ] **Step 1: Write the failing test**

```python
# tests/test_rewards_checkout.py
"""Self-coupon resolution + that the existing pricing path applies it under the floor."""
import sqlite3
import pytest


@pytest.fixture
def appmod(monkeypatch, tmp_path):
    import app as appmod
    monkeypatch.setattr(appmod, "LOG_DB", str(tmp_path / "chat_log.db"))
    from dashboard import coupons
    with sqlite3.connect(appmod.LOG_DB) as cx:
        coupons.init_coupons_table(cx)
    return appmod


def test_resolve_self_coupon_pct(appmod):
    from dashboard import coupons
    with sqlite3.connect(appmod.LOG_DB) as cx:
        c = coupons.mint_self(cx, email="m@x.com", product_slug="terrain-restore")
    pct, found = appmod._resolve_self_coupon_pct(c["code"], "terrain-restore")
    assert pct == 15 and found and found["code"] == c["code"]
    # wrong product → no match
    assert appmod._resolve_self_coupon_pct(c["code"], "other")[0] == 0
    # junk code → no match
    assert appmod._resolve_self_coupon_pct("nope", "terrain-restore")[0] == 0


def test_pricing_applies_coupon_clamped_to_floor(appmod):
    from dashboard import pricing
    # 90% off would blow past the 57% wholesale floor → clamp up to the floor
    p = {"name": "X", "price_cents": 10000, "sku_discount_floor_pct": 0.57}
    out = pricing.compute([{"product": p, "qty": 1, "unit_list_cents": 10000}],
                          settings=pricing.DEFAULTS, coupon_pct=90)
    assert out["lines"][0]["unit_cents"] >= 5700  # never below the 57% floor
```

> If `pricing.compute`'s item shape differs, adjust the item dict to match `_price_cart`'s builder (`app.py:3417` `_pricing_item`). The assertion (never below floor) is the contract that matters.

- [ ] **Step 2: Run it to verify it fails**

Run: `cd ~/deploy-chat && doppler run -p remedy-match -c prd -- env DATA_DIR=/tmp/jshell-test python3 -m pytest tests/test_rewards_checkout.py -q`
Expected: FAIL — `_resolve_self_coupon_pct` undefined.

- [ ] **Step 3: Add `_resolve_self_coupon_pct` near `_resolve_checkout_coupon_pct` (app.py ~10091)**

```python
def _resolve_self_coupon_pct(code, product_slug):
    """1b self-coupon → (pct, coupon|None). Product-bound; never raises."""
    code = (code or "").strip()
    if not REWARDS_1B_ENABLED or not code or not product_slug:
        return 0, None
    try:
        from dashboard import coupons as _coupons
        with sqlite3.connect(LOG_DB) as cx:
            _coupons.init_coupons_table(cx)
            found = _coupons.validate(cx, code, product_slug=product_slug)
        return (int(found["pct"]), found) if found else (0, None)
    except Exception as e:  # noqa: BLE001 — a coupon never blocks checkout
        print(f"[coupons] resolve failed: {e!r}", flush=True)
        return 0, None
```

- [ ] **Step 4: Thread it into `begin_checkout` (app.py ~5117)**

Replace the coupon-resolution + price block so the self-coupon combines with the existing referral/daily pct (non-stacking: take the max), and mark it redeemed after the invoice exists:

```python
        _ref_pct, _ref_ctx = _resolve_checkout_coupon_pct(data.get("referral_code"), email)
        _self_pct, _self_coupon = _resolve_self_coupon_pct(data.get("coupon_code"), slug)
        _eff_pct = max(_ref_pct or 0, _self_pct or 0)
        try:
            pc = _price_cart([{"slug": slug, "qty": qty}], ship=ship,
                             coupon_pct=_eff_pct,
                             points_to_redeem_cents=redeem)
        except CheckoutError as ce:
            return jsonify({"ok": False, "error": str(ce)}), 400
```

Then, right after `_record_referral_if_any(_ref_ctx, email, inv.get("Id"))`, add:

```python
        if _self_coupon and _self_pct >= (_ref_pct or 0):
            try:
                from dashboard import coupons as _coupons
                with _db_lock, sqlite3.connect(LOG_DB) as _ccx:
                    _coupons.mark_redeemed(_ccx, _self_coupon["code"], order_ref=inv.get("Id"))
            except Exception as e:  # noqa: BLE001
                print(f"[coupons] redeem-mark failed: {e!r}", flush=True)
```

- [ ] **Step 5: Run the tests to verify they pass**

Run: `cd ~/deploy-chat && doppler run -p remedy-match -c prd -- env DATA_DIR=/tmp/jshell-test python3 -m pytest tests/test_rewards_checkout.py -q`
Expected: PASS.

- [ ] **Step 6: Commit**

```bash
git add app.py tests/test_rewards_checkout.py
git commit -m "feat(rewards): resolve + redeem self-coupon at begin_checkout, clamped to floor (1b-A)"
```

---

### Task 4: Client — see (overlay), claim, wallet, avatar orb

**Files:**
- Modify: `static/shell.js` (gate on `rewards1b`; featured remedy in overlay; claim flow; wallet panel; avatar orb)
- Modify: `static/shell.css` (claim button, wallet, avatar orb)
- Test: manual/visual QA via the playwright screenshot workflow.

**Interfaces:**
- Consumes: `window.__SHELL__.rewards1b`; `GET /api/journey/wallet`; `POST /api/journey/claim-coupon`; the journey `card.status` + `shell-map.json` `featured`.

- [ ] **Step 1: Add the avatar + claim + wallet CSS to `static/shell.css`**

Append:

```css
/* 1b reward layer */
#journey-shell .js-orb { width: 26px; height: 26px; border-radius: 50%; margin-left: 4px;
  background: radial-gradient(circle at 50% 40%, var(--gold,#d4a843), transparent 70%);
  box-shadow: 0 0 0 0 var(--gold,#d4a843); opacity: .4; transition: opacity .4s, box-shadow .4s; }
#journey-shell .js-orb[data-lit="1"] { opacity: .7; box-shadow: 0 0 8px 1px var(--gold,#d4a843); }
#journey-shell .js-orb[data-lit="2"] { opacity: .85; box-shadow: 0 0 12px 2px var(--gold,#d4a843); }
#journey-shell .js-orb[data-lit="3"] { opacity: 1; box-shadow: 0 0 18px 3px var(--gold,#d4a843); }
.js-pav-featured { margin: 4px 0 8px; padding: 8px; border-radius: 8px;
  background: var(--surface-2,#162318); }
.js-pav-featured .js-fname { font-weight: 600; color: var(--gold,#d4a843); }
.js-pav-featured .js-fpower { font-size: 12px; color: var(--muted,#a89870); }
.js-claim { margin-top: 6px; background: var(--gold,#d4a843); color: #12130f; border: none;
  border-radius: 8px; padding: 5px 10px; font-size: 12px; font-weight: 600; cursor: pointer; }
.js-claim[disabled] { opacity: .5; cursor: default; }
.js-wallet-coupon { padding: 6px 8px; border: 1px dashed var(--gold,#d4a843); border-radius: 8px;
  margin: 6px 0; font-size: 13px; }
.js-wallet-coupon .js-exp { color: var(--muted,#a89870); font-size: 11px; }
```

- [ ] **Step 2: Add the reward logic to `static/shell.js`**

Inside the IIFE, add a `REWARDS = !!(window.__SHELL__ && window.__SHELL__.rewards1b)` near `MODE`. In `buildOverlay`, after the intrigue line, insert the featured block + claim button when `REWARDS`:

```javascript
      if (REWARDS && meta.featured) {
        var fb = el("div", "js-pav-featured",
          '<div class="js-fname">' + meta.featured.product_name + '</div>' +
          '<div class="js-fpower">' + meta.featured.healing_power + '</div>');
        var claim = el("button", "js-claim", "Claim 15% off");
        if (card.status !== "done") { claim.disabled = true; claim.textContent = "Complete this step to unlock"; }
        claim.onclick = function () { claimCoupon(card.key, claim); };
        fb.appendChild(claim);
        box.appendChild(fb);
      }
```

Add these functions inside the IIFE (before `boot`):

```javascript
  function claimCoupon(land, btn) {
    btn.disabled = true; btn.textContent = "Claiming…";
    fetch("/api/journey/claim-coupon", {
      method: "POST", credentials: "same-origin",
      headers: {"Content-Type": "application/json"}, body: JSON.stringify({land: land})
    }).then(function (r) { return r.json().then(function (j) { return {s: r.status, j: j}; }); })
      .then(function (res) {
        if (res.s === 200) { btn.textContent = "✓ In your wallet"; refreshWallet(); }
        else if (res.j && res.j.needs === "email_tos") { btn.disabled = false; location.href = "/begin/match"; }
        else { btn.disabled = false; btn.textContent = "Complete this step to unlock"; }
      }).catch(function () { btn.disabled = false; btn.textContent = "Claim 15% off"; });
  }

  function refreshWallet() {
    if (!REWARDS) return;
    fetch("/api/journey/wallet", {credentials: "same-origin"})
      .then(function (r) { return r.json(); })
      .then(function (j) {
        var coupons = (j && j.coupons) || [];
        var orb = document.querySelector("#journey-shell .js-orb");
        if (orb) orb.setAttribute("data-lit", String(Math.min(coupons.length, 3)));
        var panel = document.getElementById("js-wallet-body");
        if (panel) {
          panel.innerHTML = coupons.length ? "" : "<p class='js-fpower'>No offers yet — complete a step to earn one.</p>";
          coupons.forEach(function (c) {
            panel.appendChild(el("div", "js-wallet-coupon",
              "<b>15% off</b> " + c.product_slug +
              "<div class='js-exp'>expires " + (c.expires_at || "").slice(0, 10) + "</div>"));
          });
        }
      }).catch(function () {});
  }
```

In `buildRibbon`, when `REWARDS`, add the orb next to the map button and a "Wallet" button + panel:

```javascript
    if (REWARDS) {
      var orb = el("span", "js-orb"); orb.setAttribute("data-lit", "0"); orb.title = "Your biofield";
      bar.appendChild(orb);
      var walletBtn = el("button", "js-mypath-btn", "Wallet");
      bar.appendChild(walletBtn);
      var wp = el("div", "js-mypath"); wp.id = "js-wallet";
      wp.appendChild(el("h4", null, "Your offers"));
      var body = el("div"); body.id = "js-wallet-body"; wp.appendChild(body);
      document.body.appendChild(wp);
      walletBtn.onclick = function () { wp.classList.toggle("open"); };
    }
```

In `boot`, after the overlay is built, call `refreshWallet();`.

- [ ] **Step 3: Syntax check + visual verify**

Run: `cd ~/deploy-chat && node --check static/shell.js && echo OK`
Then start the app with both flags on and screenshot the overlay + wallet (reuse the workflow):
`doppler run -p remedy-match -c prd -- env DATA_DIR=/tmp/jshell-test JOURNEY_SHELL_ENABLED=1 REWARDS_1B_ENABLED=1 PORT=5078 python3 app.py`
Expected: overlay land cards show the featured remedy + "Complete this step to unlock" (fresh session); the ribbon shows the orb (dim) + Wallet button; Wallet panel says "No offers yet".

- [ ] **Step 4: Commit**

```bash
git add static/shell.js static/shell.css
git commit -m "feat(rewards): client — featured remedy, claim flow, wallet, biofield orb (1b-A)"
```

---

### Task 5: Console view + integration smoke + PR

**Files:**
- Modify: `app.py` (read-only `GET /api/console/coupons` behind the console key)
- Test: `tests/test_rewards_console.py`

**Interfaces:**
- Consumes: `coupons.wallet`-style query; the existing console auth (`X-Console-Key` / `_auth`).

- [ ] **Step 1: Write the failing test**

```python
# tests/test_rewards_console.py
import sqlite3, pytest

@pytest.fixture
def client(monkeypatch, tmp_path):
    import app as appmod
    monkeypatch.setattr(appmod, "LOG_DB", str(tmp_path / "chat_log.db"))
    monkeypatch.setattr(appmod, "CONSOLE_SECRET", "test-secret")
    from dashboard import coupons
    with sqlite3.connect(appmod.LOG_DB) as cx:
        coupons.init_coupons_table(cx)
        coupons.mint_self(cx, email="m@x.com", product_slug="terrain-restore")
    appmod.app.config["TESTING"] = True
    return appmod.app.test_client()

def test_console_lists_coupons_authed(client):
    r = client.get("/api/console/coupons", headers={"X-Console-Key": "test-secret"})
    assert r.status_code == 200 and len(r.get_json()["coupons"]) == 1

def test_console_coupons_requires_auth(client):
    assert client.get("/api/console/coupons").status_code in (401, 403)
```

- [ ] **Step 2: Run it to verify it fails**

Run: `cd ~/deploy-chat && doppler run -p remedy-match -c prd -- env DATA_DIR=/tmp/jshell-test python3 -m pytest tests/test_rewards_console.py -q`
Expected: FAIL (404).

- [ ] **Step 3: Add the console route in `app.py`** (use the same auth guard other `/api/console/*` routes use)

```python
@app.route("/api/console/coupons", methods=["GET"])
def api_console_coupons():
    if not _console_authed(request):   # use whatever guard the neighbouring console routes use
        return ("", 401)
    with sqlite3.connect(LOG_DB) as cx:
        from dashboard import coupons as _coupons
        _coupons.init_coupons_table(cx)
        rows = cx.execute(
            "SELECT code,product_slug,pct,kind,email,minted_at,expires_at,redeemed_at "
            "FROM coupons ORDER BY minted_at DESC LIMIT 500").fetchall()
    keys = ["code", "product_slug", "pct", "kind", "email", "minted_at", "expires_at", "redeemed_at"]
    return jsonify({"coupons": [dict(zip(keys, r)) for r in rows]})
```

> Match the exact console-auth guard used by adjacent `/api/console/*` routes (grep `def api_console_` to copy the pattern — e.g. `_auth(request)` returning a role, or an `X-Console-Key`/`CONSOLE_SECRET` check). Replace `_console_authed(request)` accordingly so the two tests pass.

- [ ] **Step 4: Run all 1b tests + the 1a suite (no regressions)**

Run: `cd ~/deploy-chat && doppler run -p remedy-match -c prd -- env DATA_DIR=/tmp/jshell-test python3 -m pytest tests/test_coupons.py tests/test_rewards_claim.py tests/test_rewards_checkout.py tests/test_rewards_console.py tests/test_shell_map_config.py tests/test_journey_shell_inject.py tests/test_begin_journey_map.py -q`
Expected: PASS, all green.

- [ ] **Step 5: Two-flag manual smoke + dark-by-default**

With `JOURNEY_SHELL_ENABLED=1 REWARDS_1B_ENABLED=1`: overlay shows featured remedies + claim; complete a stage (seed gates or use the funnel) → claim → coupon appears in Wallet, orb brightens. Restart with `REWARDS_1B_ENABLED` unset → no claim buttons / no orb / `/api/journey/claim-coupon` returns 404; 1a shell still works.

- [ ] **Step 6: Commit + push + PR**

```bash
git add app.py tests/test_rewards_console.py
git commit -m "feat(rewards): read-only console coupons view (1b-A)"
git push
gh pr create --title "Virtual Remedies & Coupon Wallet (1b-A) — dark" --body "$(cat <<'EOF'
Per-land virtual remedies + a 15%/10-day self-coupon, claimed via the funnel email+TOS gate, redeemed through the existing checkout coupon_pct path (floored at 57% wholesale). New coupons registry; biofield-orb avatar + wallet. Reuses begin_funnel, pricing.compute, products.json. Ships dark behind REWARDS_1B_ENABLED. 1b-B (gift coupon + Ambassador) is a later increment.

🤖 Generated with [Claude Code](https://claude.com/claude-code)
EOF
)"
```

---

## Self-Review

**Spec coverage:**
- Coupon registry (mint/validate/redeem/wallet) → Task 1. ✓
- See (featured remedy in overlay) → Task 2 (config) + Task 4 (render). ✓
- Earn / claim gated by email+TOS + stage-done → Task 2 (`/api/journey/claim-coupon`). ✓
- Use / checkout applies + redeems, floored → Task 3. ✓
- Wallet → Task 2 (endpoint) + Task 4 (panel). ✓
- Biofield-orb avatar (not body) → Task 4. ✓
- `REWARDS_1B_ENABLED` flag, dark, independent of 1a → Task 2. ✓
- Featured slugs exist in catalog → Task 2 guard test. ✓
- Console view → Task 5. ✓
- No points/begin_funnel/referral changes → Tasks reuse only; referral path untouched (Task 3 adds a parallel resolver). ✓

**Placeholder scan:** none — every step has concrete code/commands. Two spots flagged for the implementer to match existing patterns (the console-auth guard in Task 5; the `pricing.compute` item shape in Task 3) — both include the exact grep to copy from and the contract to satisfy.

**Type consistency:** `mint_self/validate/mark_redeemed/wallet` signatures match between Task 1, the endpoints (Task 2), and checkout (Task 3). `_resolve_self_coupon_pct(code, slug) -> (int, dict|None)` consistent Task 3. `inject_shell_html(html, mode, rewards1b=False)` updated in Task 2 and matches its only caller (`after_request`). Client reads `window.__SHELL__.rewards1b` (Task 2 emits it, Task 4 consumes).
