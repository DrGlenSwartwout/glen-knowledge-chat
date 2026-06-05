# BOS Phase 10: Stripe retail checkout

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development or superpowers:executing-plans. Steps use checkbox (`- [ ]`) syntax.

**Goal:** Let retail funnel customers pay by card via Stripe (today the funnel offers Zelle/Wise + an optional QBO online-pay link; only wholesale uses Stripe). When `method=card` and `STRIPE_ACTIVE`, the buy route creates a Stripe Checkout Session and the browser redirects to it; a new return handler records the QBO payment and captures the PaymentIntent on the order (so retail card refunds work via the Phase 9 path).

**Architecture:** Mirror the proven wholesale Stripe flow exactly. A `_stripe_checkout_url_for_retail` helper + a `GET /begin/checkout-return` handler (copies of `_stripe_checkout_url_for_order` / `practitioner_checkout_return`), retail metadata (`kind=retail`, `slug`), and a one-line redirect in the buy page's confirmation step. Activation is Glen's env: `STRIPE_ACTIVE=1` + `STRIPE_SECRET_KEY`.

**Builds on:** the merged Business OS (Stripe refund + payment_intent capture from Phase 9). New branch `sess/ec0e1f15` off main, worktree `/tmp/wt-deploy-chat-ec0e1f15`.

---

## File Structure

- `app.py` (modify): `_stripe_checkout_url_for_retail` helper; `begin_checkout` adds `customer_id` to `out` + the `stripe_url` for card; the `/begin/checkout-return` route; `begin_product_data` exposes the card option when Stripe is active.
- `static/begin-buy.html` (modify): redirect to `stripe_url` in `renderConfirmation`.

This phase is route/checkout wiring; it imports Pinecone so it is verified under doppler (app import + route registration + the buy page redirect), not local pytest. The actual Stripe session creation runs once `STRIPE_ACTIVE` is set.

---

## Task 1: Backend wiring (`app.py`)

**Files:**
- Modify: `app.py`

- [ ] **Step 1: Add the retail session helper** next to `_stripe_checkout_url_for_order` (search `def _stripe_checkout_url_for_order`). Add:

```python
def _stripe_checkout_url_for_retail(out, email, slug):
    """Create a Stripe Checkout Session for a retail funnel order; returns its URL.
    Captures invoice_id/customer_id/slug in metadata so the return handler records
    the QBO payment and the PaymentIntent (for refunds)."""
    try:
        from dashboard import stripe_pay
        total_cents = int(round(float(out.get("total") or 0) * 100))
        if total_cents <= 0:
            return ""
        success = (f"{PUBLIC_BASE_URL}/begin/checkout-return"
                   f"?session_id={{CHECKOUT_SESSION_ID}}")
        sess = stripe_pay.create_checkout_session(
            total_cents, customer_email=email,
            description=f"Remedy Match order #{out.get('doc_number')}",
            metadata={"invoice_id": out.get("invoice_id"),
                      "customer_id": out.get("customer_id"),
                      "kind": "retail", "slug": slug},
            success_url=success,
            cancel_url=f"{PUBLIC_BASE_URL}/begin/buy/{slug}")
        return sess.get("url") or ""
    except Exception as e:
        print(f"[stripe-retail] session create failed: {e!r}", flush=True)
        return ""
```

- [ ] **Step 2: In `begin_checkout`** (`POST /begin/checkout/<slug>`), expose the customer id + add the Stripe card branch. After the `out = {...}` dict is built (the line with `"pay_link": qb.get_invoice_pay_link(inv)`), add immediately:

```python
        out["customer_id"] = cust.get("Id")
```

Then, after the existing `if method in ("zelle", "wise"):` block (the one that sets `out["pay_instructions"]`), add:

```python
        elif method == "card" and _STRIPE_ACTIVE:
            out["stripe_url"] = _stripe_checkout_url_for_retail(out, email, slug)
```

- [ ] **Step 3: Add the retail return handler** near the buy/checkout routes (e.g. right after `begin_checkout`). It mirrors `practitioner_checkout_return`:

```python
@app.route("/begin/checkout-return")
def begin_checkout_return():
    """Stripe retail return: verify the session, record the QBO payment, capture
    the PaymentIntent on the order, then back to the buy page (paid)."""
    from flask import redirect as _redir
    sid = (request.args.get("session_id") or "").strip()
    slug = ""
    paid = "0"
    if sid:
        try:
            from dashboard import stripe_pay
            sess = stripe_pay.get_session(sid)
            md = sess.get("metadata") or {}
            slug = md.get("slug", "")
            if sess.get("payment_status") == "paid":
                paid = "1"
                inv, cid = md.get("invoice_id"), md.get("customer_id")
                if inv and cid:
                    try:
                        from dashboard import qbo_billing as qb
                        qb.record_payment(cid, int(sess.get("amount_total") or 0), inv)
                    except Exception as e:
                        print(f"[begin-return] qbo payment failed: {e!r}", flush=True)
                    pi = sess.get("payment_intent")
                    if pi:
                        try:
                            _cxo = _sqlite3.connect(LOG_DB); _cxo.row_factory = _sqlite3.Row
                            try:
                                _o = _bos_orders.find_order_by_external_ref(_cxo, inv)
                                if _o:
                                    _bos_orders.set_order_stripe_pi(_cxo, _o["id"], pi)
                            finally:
                                _cxo.close()
                        except Exception as _e:
                            print(f"[begin-return] pi capture: {_e!r}", flush=True)
        except Exception as e:
            print(f"[begin-return] {e!r}", flush=True)
    dest = (f"/begin/buy/{slug}?paid={paid}" if slug else f"/begin?paid={paid}")
    return _redir(dest)
```

- [ ] **Step 4: Show the card option when Stripe is active.** In `begin_product_data` (search `def begin_product_data` / the `/begin/product-data/<slug>` route), find the response field `"payments_active": _QBO_PAYMENTS_ACTIVE` and change it to:

```python
            "payments_active": _QBO_PAYMENTS_ACTIVE or _STRIPE_ACTIVE,
```

- [ ] **Step 5: Compile + verify under doppler**

Run: `python3 -m py_compile app.py` (OK).
Run:
```bash
doppler run -p remedy-match -c prd -- bash -c 'mkdir -p /tmp/bostest && DATA_DIR=/tmp/bostest python3 - <<PY
import app
rules = {r.rule for r in app.app.url_map.iter_rules()}
assert "/begin/checkout-return" in rules, "return route missing"
assert hasattr(app, "_stripe_checkout_url_for_retail"), "retail helper missing"
print("RETAIL_STRIPE_OK")
PY'
rm -rf /tmp/bostest
```
Expected: `RETAIL_STRIPE_OK`.

Run: `python3 -m pytest tests/test_bos_spine.py -q` (green; no BOS logic changed).

- [ ] **Step 6: Commit**

```bash
git add app.py
git commit -m "feat(bos): Stripe retail checkout (session + /begin/checkout-return + pi capture)"
```

---

## Task 2: Frontend redirect (`static/begin-buy.html`)

**Files:**
- Modify: `static/begin-buy.html`

- [ ] **Step 1: Redirect to Stripe** in `renderConfirmation`. Find the `if (data.method === 'card') {` line inside `renderConfirmation`. Insert as the FIRST statement inside that branch (before the existing `var link = (data.pay_link...` logic):

```javascript
    if (data.stripe_url) { window.location.href = data.stripe_url; return; }
```

This sends the browser to Stripe Checkout when a Stripe session was created; otherwise the existing QBO online-pay link / "card coming soon" path is unchanged.

- [ ] **Step 2: (Optional) detect `?paid=1` on the buy page** so a returning paid customer sees a thank-you state. This is a nice-to-have; if the existing page already handles a post-purchase state, skip. If adding: near the page's init, read `new URLSearchParams(location.search).get('paid')` and, when `'1'`, show a brief "Payment received, thank you" message. Keep it minimal and only if it fits the file cleanly.

- [ ] **Step 3: Verify it parses**

Run: `python3 -c "import html.parser; html.parser.HTMLParser().feed(open('static/begin-buy.html').read()); print('parsed OK')"`
Confirm: no em dashes added; the redirect line is inside the card branch and returns before `initConcierge`.

- [ ] **Step 4: Commit**

```bash
git add static/begin-buy.html
git commit -m "feat(bos): redirect retail card checkout to Stripe when a session is created"
```

---

## Self-Review

**Spec coverage:** retail card checkout creates a Stripe session (`method=card` + `STRIPE_ACTIVE`), the browser redirects to it, and `/begin/checkout-return` records the QBO payment + captures the PaymentIntent so retail card refunds work via Phase 9. The card option shows when QBO or Stripe online-pay is active.

**Reuse:** mirrors `_stripe_checkout_url_for_order` + `practitioner_checkout_return` (proven wholesale flow) and the Phase 9 `find_order_by_external_ref` / `set_order_stripe_pi` capture helpers.

**Activation:** gated on `STRIPE_ACTIVE=1` + `STRIPE_SECRET_KEY` in the env (Glen's switch). With Stripe off, behavior is unchanged (Zelle/Wise + the existing QBO card path).

**Safety:** the QBO invoice is created before redirect (an unpaid invoice, like Zelle/Wise pre-payment); the return handler only records payment when Stripe reports `payment_status == paid`; all capture is best-effort and cannot break the return redirect. No funnel checkout path is broken when Stripe is off.

**Placeholder scan:** none (Step 2 of Task 2 is explicitly optional/skippable).

**Type consistency:** `_stripe_checkout_url_for_retail(out, email, slug)`, the `stripe_url` response key, the metadata (`invoice_id`/`customer_id`/`kind`/`slug`), and the return-handler capture (`find_order_by_external_ref` + `set_order_stripe_pi`) match the Phase 9 contracts and the wholesale pattern.
```
