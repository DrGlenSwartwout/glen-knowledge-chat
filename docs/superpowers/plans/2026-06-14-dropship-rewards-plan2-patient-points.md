# Drop-ship Rewards Plan 2 — Patient Channel-Locked Points

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development. Steps use checkbox (`- [ ]`) syntax.

**Goal:** A patient earns RM loyalty points on a practitioner client-page order, **scoped to that practitioner** (redeemable only on their client page, never RM-direct), **RM-absorbed**, and redeemable at checkout up to a safe floor. Earn rate reuses `points_earn_pct` (5%). Ships dark behind a `CLIENT_POINTS_ENABLED` flag.

**Architecture:** The points ledger gains a `scope` column (default `"rm"` = ordinary retail points; channel points use `"dispensary:<pid>"`). The client checkout passes the patient's scoped balance + requested redemption into `build_client_order`, which caps redemption at the order's **total service fee** (so RM's per-unit keep never drops below its product base — practitioner margin stays full, RM absorbs the discount) and applies it as a QBO `discount_cents`. On the Stripe return (`kind=="client"`), the patient's points are recorded: a redeem (idempotent) if points were used, otherwise an earn on the product subtotal (full-price-only, mirroring retail). The client page shows the patient's channel balance and an "apply points" control. Card payments only (the path that has a settle-on-paid return), matching where the practitioner margin is credited.

**Tech Stack:** Python 3.11, Flask, sqlite (points ledger), QBO + Stripe, pytest.

**Spec:** `docs/superpowers/specs/2026-06-14-dropship-rewards-design.md` (Piece 2).

**Redemption rule (decided):** redeemed = `min(requested, scoped_balance, total_service_fee_cents)`. RM forgoes at most its service fee; it never sells a bottle below its blended base cost; the practitioner always receives full margin. (A more generous "subsidize below base" option is deferred to a future toggle.)

**Test invocation:** pure points module → `~/.venvs/deploy-chat311/bin/python -m pytest <path> -q`. App/route tests → `doppler run -p remedy-match -c prd -- env DATA_DIR="$HOME/deploy-chat" ~/.venvs/deploy-chat311/bin/python -m pytest <path> -q` (from inside the worktree; ignore the 2 known pre-existing failures).

---

### Task 1: scoped points ledger

**Files:**
- Modify: `dashboard/points.py`
- Test: `tests/test_points_scope.py` (new)

- [ ] **Step 1: Write the failing test** — `tests/test_points_scope.py`:

```python
import sqlite3
from dashboard import points


def _cx():
    cx = sqlite3.connect(":memory:")
    cx.row_factory = sqlite3.Row
    points.init_points_table(cx)
    return cx


def test_scopes_are_isolated():
    cx = _cx()
    points.credit(cx, "p@x.com", value_cents=500, reason="earn:dispensary",
                  order_ref="O1", scope="dispensary:42")
    points.credit(cx, "p@x.com", value_cents=200, reason="earn", order_ref="O2")  # default rm
    assert points.balance(cx, "p@x.com", scope="dispensary:42") == 500
    assert points.balance(cx, "p@x.com") == 200            # rm scope unaffected
    assert points.balance(cx, "p@x.com", scope="dispensary:99") == 0


def test_redeem_scoped():
    cx = _cx()
    points.credit(cx, "p@x.com", value_cents=500, reason="earn:dispensary",
                  order_ref="O1", scope="dispensary:42")
    points.redeem(cx, "p@x.com", value_cents=300, order_ref="O3", scope="dispensary:42")
    assert points.balance(cx, "p@x.com", scope="dispensary:42") == 200
    # cannot overdraw the scope even though another scope might have funds
    points.credit(cx, "p@x.com", value_cents=1000, reason="earn", order_ref="O4")  # rm
    import pytest
    with pytest.raises(ValueError):
        points.redeem(cx, "p@x.com", value_cents=999, order_ref="O5", scope="dispensary:42")


def test_has_entry_is_scoped():
    cx = _cx()
    points.credit(cx, "p@x.com", value_cents=500, reason="earn:dispensary",
                  order_ref="O1", scope="dispensary:42")
    assert points.has_entry(cx, order_ref="O1", reason="earn:dispensary", scope="dispensary:42")
    assert not points.has_entry(cx, order_ref="O1", reason="earn:dispensary", scope="rm")


def test_default_scope_is_rm_backward_compatible():
    cx = _cx()
    points.earn(cx, "p@x.com", full_price_cents=10000, earn_pct=0.05, order_ref="O1")
    assert points.balance(cx, "p@x.com") == 500            # default rm, old signature works
```

- [ ] **Step 2: Run → fail.**
Run: `~/.venvs/deploy-chat311/bin/python -m pytest tests/test_points_scope.py -q`

- [ ] **Step 3: Implement** — edit `dashboard/points.py`.

`init_points_table`: add the column to the CREATE and migrate older tables:
```python
def init_points_table(cx):
    cx.execute("""
        CREATE TABLE IF NOT EXISTS points_ledger (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            email TEXT NOT NULL,
            delta_cents INTEGER NOT NULL,
            reason TEXT,
            order_ref TEXT,
            balance_after INTEGER NOT NULL,
            created_at TEXT DEFAULT (datetime('now')),
            scope TEXT NOT NULL DEFAULT 'rm'
        )""")
    cols = [r[1] for r in cx.execute("PRAGMA table_info(points_ledger)").fetchall()]
    if "scope" not in cols:
        cx.execute("ALTER TABLE points_ledger ADD COLUMN scope TEXT NOT NULL DEFAULT 'rm'")
    cx.execute("CREATE INDEX IF NOT EXISTS ix_points_email ON points_ledger(email)")
    cx.commit()
```

`balance` — scope filter:
```python
def balance(cx, email, *, scope="rm"):
    row = cx.execute("SELECT COALESCE(SUM(delta_cents),0) FROM points_ledger "
                     "WHERE email=? AND scope=?", (email, scope)).fetchone()
    return int(row[0] or 0)
```

`_add` — write scope, balance_after within scope:
```python
def _add(cx, email, delta_cents, reason, order_ref, scope="rm"):
    bal = balance(cx, email, scope=scope) + int(delta_cents)
    cx.execute("""INSERT INTO points_ledger(email,delta_cents,reason,order_ref,balance_after,scope)
                  VALUES (?,?,?,?,?,?)""",
               (email, int(delta_cents), reason, order_ref, bal, scope))
    cx.commit()
    return bal
```

`earn`, `redeem`, `credit`, `has_entry` — thread `scope="rm"`:
```python
def earn(cx, email, *, full_price_cents, earn_pct, order_ref, scope="rm"):
    delta = int(round(int(full_price_cents) * float(earn_pct)))
    return _add(cx, email, delta, "earn", order_ref, scope=scope)


def redeem(cx, email, *, value_cents, order_ref, scope="rm"):
    value_cents = int(value_cents)
    if value_cents > balance(cx, email, scope=scope):
        raise ValueError("redeem exceeds balance")
    return _add(cx, email, -value_cents, "redeem", order_ref, scope=scope)


def has_entry(cx, *, order_ref, reason, scope="rm"):
    row = cx.execute("SELECT 1 FROM points_ledger WHERE order_ref=? AND reason=? AND scope=? LIMIT 1",
                     (order_ref, reason, scope)).fetchone()
    return row is not None


def credit(cx, email, *, value_cents, reason, order_ref, scope="rm"):
    if not has_entry(cx, order_ref=order_ref, reason=reason, scope=scope):
        _add(cx, email, value_cents, reason, order_ref, scope=scope)
```

- [ ] **Step 4: Run → pass.** Also run the existing points tests to confirm backward-compat:
`doppler run -p remedy-match -c prd -- env DATA_DIR="$HOME/deploy-chat" ~/.venvs/deploy-chat311/bin/python -m pytest tests/test_points_scope.py tests/test_points_settlement.py tests/test_referral_settlement.py -q`
Expected: PASS (existing callers use the default `scope="rm"`).

- [ ] **Step 5: Commit** — `feat(client-points): scope column on points ledger`

---

### Task 2: redemption cap in `build_client_order`

**Files:**
- Modify: `dashboard/dropship_checkout.py`
- Test: `tests/test_client_points_order.py` (new)

- [ ] **Step 1: Write the failing test** — `tests/test_client_points_order.py`. Mirror the existing `build_client_order` tests (see `tests/test_*client*`/`test_dropship*` for the qb monkeypatch pattern; stub `qb.find_or_create_customer`/`qb.create_invoice` and `practitioner_price_for`). Assert:
  - With `points_to_redeem_cents=0`, behavior unchanged: `create_invoice` called with `discount_cents=0`, result `points_redeemed_cents == 0`, `subtotal_cents` returned, `margin_cents` unchanged.
  - With a large `points_to_redeem_cents` + ample `points_balance_cents`, redemption is capped at the order's total `fee_cents` (sum of per-line fee × qty); `create_invoice` receives `discount_cents == that cap`; `margin_cents` is unchanged (full margin); `points_redeemed_cents == cap`.
  - Redemption is also capped by `points_balance_cents` when the balance is the binding limit.

```python
import dashboard.dropship_checkout as dc


def _stub(monkeypatch, captured):
    monkeypatch.setattr(dc, "practitioner_price_for", lambda pid, slug: 8000)  # $80 selling
    monkeypatch.setattr(dc.qb, "find_or_create_customer", lambda email, name="": {"Id": "C1"})
    def _ci(cust, lines, **kw):
        captured["discount_cents"] = kw.get("discount_cents")
        captured["lines"] = lines
        return {"Id": "INV1", "TotalAmt": sum(l["amount"] * l["qty"] for l in lines)
                - (kw.get("discount_cents", 0) / 100.0)}
    monkeypatch.setattr(dc.qb, "create_invoice", _ci)
    monkeypatch.setattr(dc, "_settings", lambda: {"discount_floor_pct": 0.57})  # if used

PRAC = {"id": "42", "modules_completed": 0}
PATIENT = {"email": "p@x.com", "name": "Pat", "ship": {"state": "HI", "country": "US"}}


def test_no_redemption_unchanged(monkeypatch):
    cap = {}; _stub(monkeypatch, cap)
    out = dc.build_client_order([{"slug": "a", "qty": 2}], PRAC, patient=PATIENT,
                                points_to_redeem_cents=0, points_balance_cents=0)
    assert out["ok"] and out["points_redeemed_cents"] == 0
    assert cap["discount_cents"] == 0
    assert out["subtotal_cents"] == 16000
    margin_full = out["margin_cents"]
    assert margin_full > 0


def test_redemption_capped_at_total_fee(monkeypatch):
    cap = {}; _stub(monkeypatch, cap)
    base = dc.build_client_order([{"slug": "a", "qty": 2}], PRAC, patient=PATIENT,
                                 points_to_redeem_cents=0, points_balance_cents=0)
    out = dc.build_client_order([{"slug": "a", "qty": 2}], PRAC, patient=PATIENT,
                                points_to_redeem_cents=999999, points_balance_cents=999999)
    # cap == total service fee on the order (== subtotal - margin - base summed); margin unchanged
    assert out["points_redeemed_cents"] == cap["discount_cents"]
    assert out["points_redeemed_cents"] > 0
    assert out["margin_cents"] == base["margin_cents"]      # practitioner keeps full margin


def test_redemption_capped_by_balance(monkeypatch):
    cap = {}; _stub(monkeypatch, cap)
    out = dc.build_client_order([{"slug": "a", "qty": 2}], PRAC, patient=PATIENT,
                                points_to_redeem_cents=999999, points_balance_cents=150)
    assert out["points_redeemed_cents"] == 150
    assert cap["discount_cents"] == 150
```

> Adjust the stub to match the real qb monkeypatch shape used in the existing dropship/client tests if it differs. Keep the three assertions' intent.

- [ ] **Step 2: Run → fail.**

- [ ] **Step 3: Implement** — modify `build_client_order` in `dashboard/dropship_checkout.py`:
  - Add keyword params `points_to_redeem_cents=0, points_balance_cents=0`.
  - While looping lines, also accumulate `total_fee_cents += q["fee_cents"] * line_qty`.
  - After the loop compute:
    ```python
    redeem_cents = max(0, min(int(points_to_redeem_cents or 0),
                              int(points_balance_cents or 0),
                              total_fee_cents))
    ```
  - Pass `discount_cents=redeem_cents` to `qb.create_invoice(cust, lines, email_to=..., discount_cents=redeem_cents)`.
  - Add to the returned dict: `"subtotal_cents": subtotal_cents` and `"points_redeemed_cents": redeem_cents`. **Do NOT change `margin_cents`** (still the full `total_margin_cents` — RM absorbs the redemption, the practitioner keeps full margin).

- [ ] **Step 4: Run → pass.**
- [ ] **Step 5: Commit** — `feat(client-points): fee-capped redemption in build_client_order`

---

### Task 3: wire earn + redeem into the client checkout + return

**Files:**
- Modify: `app.py` (the `/api/client/<code>/checkout` route + the `kind=="client"` checkout-return branch)
- Test: `tests/test_client_points_routes.py` (new)

- [ ] **Step 1: Write the failing test** — exercise the `kind=="client"` return branch directly (call the same internal the return handler uses, or post to `/begin/checkout-return` with a stubbed Stripe session whose metadata carries `kind=client`, `patient_email`, `subtotal_cents`, `points_redeemed_cents`, `practitioner_id`, `invoice_id`). Set `CLIENT_POINTS_ENABLED=1`. Assert:
  - earn path (`points_redeemed_cents=0`, `subtotal_cents=8000`) → patient scope `dispensary:<pid>` balance == round(8000 × points_earn_pct) == 400; idempotent on a second call.
  - redeem path (`points_redeemed_cents=300`, after seeding a 500 scoped balance) → scope balance == 200; idempotent.
  - with `CLIENT_POINTS_ENABLED` unset → no points written.

  Use the existing checkout-return test harness in the repo (search `tests/` for `checkout-return` / `kind` stubs) and follow it. Stub Stripe session retrieval + `wallet.earn_dropship_margin` + `_pp.record_dispensary_order` so only the points behavior is asserted.

- [ ] **Step 2: Run → fail.**

- [ ] **Step 3: Implement.**

(a) In the `/api/client/<code>/checkout` route, BEFORE building the order, look up the scoped balance and requested redemption (card + flag only):
```python
    redeem_req = 0
    bal_cents = 0
    scope = f"dispensary:{pid}"
    if os.environ.get("CLIENT_POINTS_ENABLED") and method == "card":
        try:
            redeem_req = max(0, int(body.get("points_to_redeem_cents") or 0))
        except (TypeError, ValueError):
            redeem_req = 0
        with sqlite3.connect(LOG_DB) as _bcx:
            _bcx.row_factory = sqlite3.Row
            _points_mod = __import__("dashboard.points", fromlist=["points"])
            _points_mod.init_points_table(_bcx)
            bal_cents = _points_mod.balance(_bcx, email, scope=scope)
```
(Prefer a top-of-function `from dashboard import points as _points_mod` import to the `__import__` — match the file's local-import convention.)

Pass them into the build:
```python
        out = _dropship.build_client_order(items, prac, patient=patient, method=method,
                                           points_to_redeem_cents=redeem_req,
                                           points_balance_cents=bal_cents)
```

Add to the Stripe metadata (so the return can settle points):
```python
                    metadata={
                        "kind": "client",
                        "practitioner_id": pid,
                        "margin_cents": str(out.get("margin_cents", 0)),
                        "invoice_id": str(out.get("invoice_id") or ""),
                        "customer_id": str(out.get("customer_id") or ""),
                        "patient_email": email,
                        "subtotal_cents": str(out.get("subtotal_cents", 0)),
                        "points_redeemed_cents": str(out.get("points_redeemed_cents", 0)),
                    },
```

(b) In the `kind=="client"` checkout-return branch, AFTER the existing margin credit, add the points settle (flag-gated, best-effort):
```python
                        # ── Patient channel points (flag-gated, card path) ──
                        if os.environ.get("CLIENT_POINTS_ENABLED"):
                            try:
                                from dashboard import points as _points_ret
                                p_email = (md.get("patient_email") or "").strip().lower()
                                redeemed = int(md.get("points_redeemed_cents") or 0)
                                subtotal = int(md.get("subtotal_cents") or 0)
                                scope = f"dispensary:{_pid}"
                                if p_email:
                                    with sqlite3.connect(LOG_DB) as _pcx:
                                        _pcx.row_factory = sqlite3.Row
                                        _points_ret.init_points_table(_pcx)
                                        if redeemed > 0:
                                            if not _points_ret.has_entry(_pcx, order_ref=_inv,
                                                    reason="redeem:dispensary", scope=scope):
                                                take = min(redeemed,
                                                           _points_ret.balance(_pcx, p_email, scope=scope))
                                                if take > 0:
                                                    _points_ret.redeem(_pcx, p_email, value_cents=take,
                                                                       order_ref=_inv, scope=scope)
                                        elif subtotal > 0:
                                            earn_pct = float(_pricing_settings().get("points_earn_pct", 0.05))
                                            _points_ret.credit(_pcx, p_email,
                                                value_cents=round(subtotal * earn_pct),
                                                reason="earn:dispensary", order_ref=_inv, scope=scope)
                            except Exception as _pe:
                                app.logger.exception("client points settle failed: %s", _pe)
```

- [ ] **Step 4: Run → pass.** Re-run the dispensary/client route tests to confirm the margin path is unbroken:
`doppler run -p remedy-match -c prd -- env DATA_DIR="$HOME/deploy-chat" ~/.venvs/deploy-chat311/bin/python -m pytest tests/test_client_points_routes.py tests/test_client_routes.py -q` (adapt to the actual client-route test filename).

- [ ] **Step 5: Commit** — `feat(client-points): earn + redeem on client checkout-return`

---

### Task 4: client page balance + apply-points control

**Files:**
- Modify: `app.py` (catalog returns the flag; new `/api/client/<code>/points`), `static/practitioner-client.html`
- Test: `tests/test_client_points_routes.py` (append)

- [ ] **Step 1: Write the failing test** — `/api/client/<code>/points` (POST `{email}`, consent-gated like checkout): 404 unknown code; 403 `need_optin` if not a member; returns `{ok, balance_cents, client_points_enabled}` for a member, where balance is the `dispensary:<pid>` scope. And `/api/client/<code>/catalog` now includes `client_points_enabled`.

- [ ] **Step 2: Run → fail.**

- [ ] **Step 3: Implement.**
  - In `api_client_catalog`, add `"client_points_enabled": bool(os.environ.get("CLIENT_POINTS_ENABLED"))` to the JSON.
  - Add the balance endpoint:
    ```python
    @app.route("/api/client/<code>/points", methods=["POST"])
    def api_client_points(code):
        pid = _pp.practitioner_id_by_dispensary_code(code)
        if not pid:
            return jsonify({"ok": False, "error": "unknown dispensary code"}), 404
        body = request.get_json(silent=True) or {}
        email = (body.get("email") or "").strip().lower()
        if not email:
            return jsonify({"ok": False, "error": "email required"}), 400
        _sid = request.cookies.get("amg_session", "")
        if not is_member(_sid, email):
            return jsonify({"ok": False, "need_optin": True}), 403
        enabled = bool(os.environ.get("CLIENT_POINTS_ENABLED"))
        bal = 0
        if enabled:
            from dashboard import points as _points
            with sqlite3.connect(LOG_DB) as cx:
                cx.row_factory = sqlite3.Row
                _points.init_points_table(cx)
                bal = _points.balance(cx, email, scope=f"dispensary:{pid}")
        return jsonify({"ok": True, "balance_cents": bal, "client_points_enabled": enabled})
    ```
  - In `static/practitioner-client.html`: when `client_points_enabled`, after the patient identifies (has entered email + agreed), call `/api/client/<code>/points` and show "You have $X in <practice name> rewards." Add an "Apply my points" checkbox (or amount input) near checkout; when checked, send `points_to_redeem_cents` (the patient's balance, the server caps it) in the `/api/client/<code>/checkout` body. Show a note: "Points apply to card orders and come off your total; your practitioner is unaffected." No em dashes / ALL CAPS in copy.

- [ ] **Step 4: Run → pass** + page parses (`html.parser` check) + route tests green.
- [ ] **Step 5: Commit** — `feat(client-points): client-page balance + apply-points control`

---

### Task 5: doc + suite

**Files:**
- Modify: `docs/console-settings.md` (or a short `docs/client-points.md`)

- [ ] **Step 1:** Document: patient channel points — earn `points_earn_pct` on client-page orders, scoped `dispensary:<pid>`, redeemable only on that practitioner's client page; **RM-absorbed, redemption capped at the order's total service fee** (RM never sells below its base; practitioner keeps full margin); card payments only; full-price-only earn (no earn when points are redeemed); gated by `CLIENT_POINTS_ENABLED` (default off). Note the deferred "subsidize below base" generosity toggle and the deferred refund-reversal.
- [ ] **Step 2:** Combined suite — green:
`doppler run -p remedy-match -c prd -- env DATA_DIR="$HOME/deploy-chat" ~/.venvs/deploy-chat311/bin/python -m pytest tests/test_points_scope.py tests/test_client_points_order.py tests/test_client_points_routes.py tests/test_points_settlement.py tests/test_referral_settlement.py tests/test_client_routes.py -q`
- [ ] **Step 3:** Commit — `docs(client-points): patient channel points`

---

## Self-review

- **Spec coverage:** patient earns `points_earn_pct` (Task 3 earn), scoped `dispensary:<pid>` (Task 1), redeemable only on that practitioner's page (Task 3/4 scope), RM-absorbed with full practitioner margin + safe floor (Task 2 fee cap), consent-gated via the existing member gate (Task 3/4 reuse `is_member`), ships dark behind `CLIENT_POINTS_ENABLED` (all tasks).
- **Type consistency:** `points.*(..., scope="rm")` everywhere; scope string `dispensary:<pid>`; reasons `earn:dispensary` / `redeem:dispensary`; `build_client_order(..., points_to_redeem_cents=0, points_balance_cents=0) -> {..., subtotal_cents, points_redeemed_cents}`; metadata keys `patient_email`/`subtotal_cents`/`points_redeemed_cents`.
- **Deferred:** subsidize-below-base generosity toggle; refund-driven reversal of earned/redeemed channel points; alt-pay (zelle/wise) earn/redeem (card-only in v1, matching where margin is credited); cross-scope portability.
- **Risk:** money path. Mitigations — redemption capped at the service fee (RM never below base; practitioner margin untouched), redeem recorded only on PAID and idempotent (`has_entry` on order_ref+reason+scope), earn full-price-only, everything behind a default-off flag. Tail case: balance changes between build and return → settle takes `min(redeemed, current balance)` and logs; bounded by the small per-order earn.

## Done
A patient earns and redeems practitioner-scoped, RM-absorbed loyalty points on the client page, with redemption safely capped at the order's service fee, shipped dark behind `CLIENT_POINTS_ENABLED`.
