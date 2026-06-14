# Begin-Funnel Checkout on the Pricing Engine — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development.

**Goal:** Route `/begin/checkout/<slug>` (the main funnel) through `_price_cart()` behind the existing `PRICING_ENGINE_CHECKOUT` flag, so the funnel gets the volume curve / floors / coupon / shipping and records `discount_cents`/`points_redeemed_cents`/`shipping_cents` — which also closes the Plan 4 gap (a discounted funnel order no longer earns points, since `discount_cents` is now recorded). Legacy `_qty_unit_cents` path stays unchanged when the flag is off.

**Map facts (the current route, app.py `begin_checkout`):** single product `p=_get_product(slug)`; `info_only` short-circuit; consent gate `is_member(_sid, email)` → 403 `need_optin`; legacy pricing `unit=_qty_unit_cents(p,qty)`, one-line `qb.create_invoice` (no discount/shipping), `get_cents` recorded-not-charged; `_ingest_order(source="funnel", channel="retail", get_cents=...)`; journey-event log; dispensary attribution; `allow_online=(method=="card") and _QBO_PAYMENTS_ACTIVE`; card → `_stripe_checkout_url_for_retail(out, email, slug)`; `out` carries `customer_id`. The reorder engine path (Plan 2) already shows the exact shape: `_price_cart(cart, ship=ship, coupon_pct=_active_coupon_pct())` → `qb.create_invoice(pc["qbo_lines"] + _shipping_line(pc["shipping_cents"]), discount_cents=pc["discount_cents"]+pc["points_redeemed_cents"], ...)` → `_ingest_order(..., get_cents=pc["priced"]["get_cents"], discount_cents=..., points_redeemed_cents=..., shipping_cents=...)`.

---

### Task 1: Engine path in `begin_checkout` (flag-gated)

**Files:** Modify `app.py` (`begin_checkout`); Test `tests/test_begin_checkout_engine.py`

- [ ] **Step 1: Write the failing test**

```python
# tests/test_begin_checkout_engine.py
import app as appmod

def _setup(monkeypatch):
    monkeypatch.setattr(appmod, "is_member", lambda sid, email: True)   # consent satisfied
    monkeypatch.setattr(appmod, "_get_product",
        lambda s: {"slug":s,"name":"Brain Boost","price_cents":7000,"qty_pricing":True,"qbo_item_id":"27"} if s=="brain-boost" else None)
    monkeypatch.setattr(appmod._shipping, "quote", lambda b: {"shipping_cents": 2295})
    monkeypatch.setattr(appmod.qb if hasattr(appmod,"qb") else appmod, "find_or_create_customer", lambda *a, **k: {"Id":"C1"}, raising=False)
    cap = {}
    def fake_invoice(cust, lines, **kw):
        cap["lines"] = lines; cap["kw"] = kw
        return {"Id":"INV","TotalAmt":74.0,"DocNumber":"7"}
    # qbo_billing is imported locally in begin_checkout; patch the module it imports
    import dashboard.qbo_billing as _qb
    monkeypatch.setattr(_qb, "find_or_create_customer", lambda *a, **k: {"Id":"C1"})
    monkeypatch.setattr(_qb, "create_invoice", fake_invoice)
    monkeypatch.setattr(_qb, "get_invoice_pay_link", lambda inv: "")
    monkeypatch.setattr(appmod, "_ingest_order", lambda **kw: cap.setdefault("order", kw))
    monkeypatch.setattr(appmod, "_stripe_checkout_url_for_retail", lambda *a, **k: "https://stripe/x")
    monkeypatch.setenv("PRICING_ENGINE_CHECKOUT", "true")
    return cap

def test_begin_checkout_engine_records_discount_and_shipping(monkeypatch):
    cap = _setup(monkeypatch)
    c = appmod.app.test_client()
    r = c.post("/begin/checkout/brain-boost", json={
        "email":"buyer@x.com","name":"B","method":"card","qty":6,
        "address":{"state":"CA","country":"US","name":"B"}})
    assert r.status_code == 200
    # 6 months → 29% volume off each 7000 → discount 6*(7000-4970)=12180 passed to QBO
    assert cap["kw"]["discount_cents"] == 12180
    assert cap["order"]["discount_cents"] == 12180
    assert cap["order"]["shipping_cents"] == 2295
    assert cap["order"]["source"] == "funnel"
    assert r.get_json()["customer_id"] == "C1"

def test_begin_checkout_consent_gate_still_enforced(monkeypatch):
    cap = _setup(monkeypatch)
    monkeypatch.setattr(appmod, "is_member", lambda sid, email: False)
    c = appmod.app.test_client()
    r = c.post("/begin/checkout/brain-boost", json={"email":"b@x.com","method":"card",
               "address":{"state":"CA","country":"US"}})
    assert r.status_code == 403 and r.get_json().get("need_optin") is True
```

- [ ] **Step 2: Run to verify it fails**

Run: `doppler run -p remedy-match -c prd -- env DATA_DIR="$HOME/deploy-chat" ~/.venvs/deploy-chat311/bin/python -m pytest tests/test_begin_checkout_engine.py -v`
Expected: FAIL (engine path not implemented → discount_cents not passed).

- [ ] **Step 3: Implement** — In `begin_checkout`, AFTER the consent gate (keep `info_only`, the email check, and the consent gate exactly as-is), add the engine branch; keep the existing legacy body as the `else`:

```python
    if os.environ.get("PRICING_ENGINE_CHECKOUT", "").strip().lower() in ("1","true","yes","on"):
        from dashboard import qbo_billing as qb
        try:
            redeem = int((data.get("points_to_redeem_cents") or 0))
        except (TypeError, ValueError):
            redeem = 0
        if redeem > 0:
            from dashboard import points as _points
            with sqlite3.connect(LOG_DB) as _bcx:
                _points.init_points_table(_bcx)
                redeem = min(redeem, _points.balance(_bcx, email))
        try:
            pc = _price_cart([{"slug": slug, "qty": qty}], ship=ship,
                             coupon_pct=_active_coupon_pct(),
                             points_to_redeem_cents=redeem)
        except CheckoutError as ce:
            return jsonify({"ok": False, "error": str(ce)}), 400
        cust = qb.find_or_create_customer(email, name)
        allow_online = (method == "card") and _QBO_PAYMENTS_ACTIVE
        inv = qb.create_invoice(cust, pc["qbo_lines"] + _shipping_line(pc["shipping_cents"]),
                                allow_online_pay=allow_online, email_to=email,
                                discount_cents=pc["discount_cents"] + pc["points_redeemed_cents"])
        out = {"ok": True, "invoice_id": inv.get("Id"), "sync_token": inv.get("SyncToken"),
               "doc_number": inv.get("DocNumber"), "total": inv.get("TotalAmt"),
               "method": method, "customer_id": cust.get("Id"),
               "pay_link": qb.get_invoice_pay_link(inv)}
        try:
            with _db_lock, sqlite3.connect(LOG_DB) as cx:
                cx.execute("INSERT INTO journey_events (ts, session_id, email, trigger, detail, rung_before, rung_after) "
                           "VALUES (?,?,?,?,?,?,?)",
                           (begin_funnel._now(), session_id, email, "purchase",
                            f"buy-{slug}-{method}", "", ""))
                cx.commit()
        except Exception:
            pass
        _ingest_order(source="funnel", external_ref=inv.get("Id"), email=email, name=name,
                      items=pc["items_rec"],
                      total_cents=int(round(float(inv.get("TotalAmt") or 0) * 100)),
                      address=ship, channel="retail", get_cents=pc["priced"]["get_cents"],
                      discount_cents=pc["discount_cents"],
                      points_redeemed_cents=pc["points_redeemed_cents"],
                      shipping_cents=pc["shipping_cents"])
        if method in ("zelle", "wise"):
            out["pay_instructions"] = _ALT_PAY.get(method, {})
        elif method == "card" and _STRIPE_ACTIVE:
            out["stripe_url"] = _stripe_checkout_url_for_retail(out, email, slug)
        try:
            disp = (request.cookies.get("rm_dispensary") or "").strip()
            if disp:
                _record_dispensary_sale(disp, email, qty, inv.get("Id"))
        except Exception as e:
            print(f"[dispensary] hook: {e!r}", flush=True)
        return jsonify(out)
    # else: existing legacy body unchanged
```
Wrap the existing legacy code (from `session_id = ...` through `return jsonify(out)`) in the `else`/fall-through so it runs only when the flag is off. Be careful with indentation in the large function.

- [ ] **Step 4: Run to verify it passes** — PASS (2).
- [ ] **Step 5: Commit** — `feat(checkout): begin-funnel checkout prices via engine + shipping behind PRICING_ENGINE_CHECKOUT`

---

### Task 2: Suite + doc note

- [ ] **Step 1:** Run `tests/test_begin_checkout_engine.py tests/test_begin_routes.py tests/test_reorder_checkout_engine.py tests/test_points_settlement.py` — all green (the funnel legacy path + reorder unchanged).
- [ ] **Step 2:** Append to `docs/checkout-pricing-engine.md`: "`/begin/checkout/<slug>` also prices via the engine under `PRICING_ENGINE_CHECKOUT` (list lines + discount + shipping line, US-only, GET absorbed, points redeemable by the funnel email); legacy path runs when off. This makes funnel orders record `discount_cents`, so the Plan 4 points-earn 'full-price only' rule now applies correctly to the funnel."
- [ ] **Step 3:** Commit.

---

## Self-review
- **Spec coverage:** funnel on the engine (volume/floors/coupon/shipping/US-only), discount/points/shipping recorded, consent gate preserved, journey + dispensary + stripe preserved, GET still recorded-not-charged. Flag-gated; legacy path untouched when off.
- **Risk:** changes live funnel prices → behind `PRICING_ENGINE_CHECKOUT` (default off); the consent gate, alt-pay, dispensary, and stripe-return paths are all preserved in the engine branch.
- **Closes:** Plan 4 I2 (funnel orders now record discount_cents → discounted funnel buys correctly don't earn points).
