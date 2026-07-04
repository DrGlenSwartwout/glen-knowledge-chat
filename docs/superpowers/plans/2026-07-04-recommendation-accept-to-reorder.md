# Recommendation Accept → Reorder List Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Fix the accept-before-payment dead-end and duplicate-invoice bug by making "accept a recommendation" add the items to the patient's persistent reorder list (no invoice minted at the tap) — the purchase runs through the existing retryable reorder checkout.

**Architecture:** Three small changes: (1) the accept route stops minting a QBO invoice / Stripe session and just marks the recommendation `accepted`; (2) the portal payload merges an `accepted` recommendation's items into `reorder_items` (deduped by slug); (3) light client-portal copy so the accept UX reads as "added to your reorder list." The card already renders only for `status=='sent'` and the client handler already reloads on `{ok}` (no `stripe_url`), so those need no structural change.

**Tech Stack:** Python 3, Flask, SQLite, pytest, vanilla JS (client-portal.html).

## Global Constraints

- Accept mints **no** invoice, order, or Stripe session — the purchase happens only through the normal reorder checkout (`/api/portal/<token>/checkout`).
- Patient identity is resolved from the portal token/session (`_portal_record_for`), NEVER a request-body field (unchanged from #572).
- The `status == 'sent'` guard on accept stays (single accept).
- An `accepted` recommendation's items are merged into `reorder_items` **deduped by slug** (a recommended slug already present in the reorder set is not doubled).
- The recommendation **card** renders only for `status == 'sent'`; an `accepted` rec shows no card (its items are in the reorder module); a `dismissed` rec shows nothing.
- Accepted items **persist** in the ongoing reorder list (Glen's decision).

---

## File Structure

- **Modify** `app.py` — `api_portal_recommendation_accept` (drop the invoice/Stripe path); the portal-data builder where `reorder_items` is assembled (~line 13647) — merge accepted-rec items.
- **Modify** `static/client-portal.html` — accept-handler copy + a brief "added to your reorder list" confirmation.
- **Test** `tests/test_continuity_landing.py` (extend — the existing accept tests live here) + the portal-data test that covers `reorder_items`.

---

### Task 1: Accept mints no invoice — just marks the recommendation `accepted`

**Files:**
- Modify: `app.py` (`api_portal_recommendation_accept`)
- Test: `tests/test_continuity_landing.py`

**Interfaces:**
- Produces: `POST /api/portal/<token>/recommendation/accept` → `{ok:true, accepted:true}` and status `accepted`, creating NO invoice/order/Stripe session.

- [ ] **Step 1: Write the failing test**

Add to `tests/test_continuity_landing.py` (mirror the existing accept test's setup — seed a client portal + a `sent` recommendation for the patient):

```python
def test_accept_mints_no_invoice_and_marks_accepted(...):
    # seed a portal token for patient email + a 'sent' practitioner_recommendation
    # (reuse the existing accept test's fixtures/stubs in this file)
    before = _count_orders_for(email)            # helper the sibling tests use, or query orders directly
    r = client.post(f"/api/portal/{token}/recommendation/accept")
    body = r.get_json()
    assert r.status_code == 200 and body["ok"] is True and body.get("accepted") is True
    assert "stripe_url" not in body or body["stripe_url"] is None
    assert _count_orders_for(email) == before    # NO order/invoice created
    # status flipped to 'accepted'
    with sqlite3.connect(LOG_DB) as cx:
        cx.row_factory = sqlite3.Row
        assert _pr.active_for_patient(cx, email)["status"] == "accepted"
```

Note: read the existing `test_accept_prices_at_member_price_not_zero...` / `test_accept_replay...` in this file to reuse their exact seeding + Stripe stubs, and adapt the order-count assertion to however those tests observe created orders.

- [ ] **Step 2: Run test to verify it fails**

Run: `doppler run -p remedy-match -c dev -- python -m pytest tests/test_continuity_landing.py::test_accept_mints_no_invoice_and_marks_accepted -q`
Expected: FAIL (an order/invoice IS created and/or `stripe_url` is returned by the current code)

- [ ] **Step 3: Rewrite the accept route**

Replace the body of `api_portal_recommendation_accept` (drop the pricing + `_portal_reorder_checkout` + `stripe_url`):

```python
@app.route("/api/portal/<token>/recommendation/accept", methods=["POST"])
def api_portal_recommendation_accept(token):
    """Patient accepts their active practitioner recommendation: mark it 'accepted'
    so its items surface in the patient's reorder list. NO invoice/Stripe session is
    minted here — the purchase runs through the normal reorder checkout, which is
    retryable and mints exactly one invoice. Scope: the authenticated patient's OWN
    active recommendation (identity from the portal token, never a body field)."""
    from dashboard import client_portal as _cp
    from dashboard import practitioner_recommendations as _pr
    with sqlite3.connect(LOG_DB) as cx:
        _cp.init_client_portal_table(cx)
        portal = _portal_record_for(cx, token)
    if not portal:
        return jsonify({"error": "not found"}), 404
    email = (portal.get("email") or "").strip().lower()
    with sqlite3.connect(LOG_DB) as cx:
        cx.row_factory = sqlite3.Row
        _pr.init_table(cx)
        rec = _pr.active_for_patient(cx, email)
    if not rec or rec.get("status") != "sent":
        return jsonify({"error": "No active recommendation to accept."}), 400
    with sqlite3.connect(LOG_DB) as cx:
        _pr.set_status(cx, rec["id"], "accepted")
    return jsonify({"ok": True, "accepted": True})
```

- [ ] **Step 4: Run test to verify it passes**

Run: `doppler run -p remedy-match -c dev -- python -m pytest tests/test_continuity_landing.py -q`
Expected: PASS — the new test passes; update/keep the sibling accept tests: `test_accept_replay...` still holds (second accept → 400 by the `!= 'sent'` guard); the old `test_accept_prices_at_member_price...` / `...builds_live_member_priced_checkout...` tests that asserted an invoice-at-accept must be **retargeted to the reorder-checkout path** (Task 3 of the sibling flow) or removed — the accept route no longer prices/invoices. Adjust those assertions to the new contract (accept → no invoice) in this step; the member-price behavior is re-asserted against the reorder checkout in Task 2's test.

- [ ] **Step 5: Commit**

```bash
git add app.py tests/test_continuity_landing.py
git commit -m "fix(continuity): accept mints no invoice — marks recommendation accepted only"
```

---

### Task 2: Merge accepted-recommendation items into `reorder_items` (deduped)

**Files:**
- Modify: `app.py` (the portal-data builder near the `reorder_items` assembly, ~line 13647)
- Test: `tests/test_continuity_landing.py`

**Interfaces:**
- Consumes: `practitioner_recommendations.active_for_patient` (returns the latest non-dismissed rec with `items`).
- Produces: the client-portal payload's `reorder_items` includes an `accepted` recommendation's items, deduped by slug; a member-priced checkout of that reorder set still mints one invoice at the real member price.

- [ ] **Step 1: Write the failing test**

```python
def test_accepted_recommendation_items_appear_in_reorder_deduped(...):
    # seed portal + an ACCEPTED recommendation with items [{slug:'nerve-repair',qty:1}, {slug:'terrain-restore',qty:1}]
    # AND a reorder set that already contains 'nerve-repair'
    data = _portal_payload(token)                 # however the sibling tests fetch the portal data payload
    slugs = [it["slug"] for it in data["reorder_items"]]
    assert "terrain-restore" in slugs             # the new recommended item is present
    assert slugs.count("nerve-repair") == 1       # not doubled (dedup by slug)

def test_reorder_checkout_of_accepted_items_prices_member_and_one_invoice(...):
    # after the accepted items are in reorder_items, checking out the reorder via
    # /api/portal/<token>/checkout mints exactly ONE invoice at the member price (>0, <6997)
    # (reuse the member-price assertion pattern from the old accept test, now against the checkout route)
    ...
```

Note: read how the sibling tests fetch the portal-data payload (the route/function that returns `reorder_items` + `recommendation`) and how the old member-price test asserted the price; reuse both, retargeted to the reorder path.

- [ ] **Step 2: Run test to verify it fails**

Run: `doppler run -p remedy-match -c dev -- python -m pytest tests/test_continuity_landing.py::test_accepted_recommendation_items_appear_in_reorder_deduped -q`
Expected: FAIL (`terrain-restore` not in `reorder_items` — accepted-rec items aren't merged yet)

- [ ] **Step 3: Merge accepted-rec items into the reorder set**

At the `reorder_items` assembly in the portal-data builder (~`app.py:13647`, where `reorder_src` is computed), after `reorder_src` is resolved, merge an `accepted` recommendation's items in, deduped by slug. Read the exact `reorder_items` element shape first (slug/qty/name/…) and build merged items in that shape:

```python
    # Merge an ACCEPTED practitioner recommendation's items into the reorder set
    # (deduped by slug) so an accepted recommendation lives in the patient's
    # persistent, retryable reorder list. 'sent' recs stay on the card; 'accepted'
    # move here; 'dismissed' show nowhere.
    try:
        from dashboard import practitioner_recommendations as _pr
        with sqlite3.connect(LOG_DB) as _rcx:
            _rcx.row_factory = sqlite3.Row
            _pr.init_table(_rcx)
            _rec = _pr.active_for_patient(_rcx, email)
        if _rec and _rec.get("status") == "accepted":
            _have = { (it.get("slug") or "").strip().lower() for it in (reorder_src or []) }
            for _it in (_rec.get("items") or []):
                _slug = (_it.get("slug") or "").strip().lower()
                if _slug and _slug not in _have:
                    reorder_src = (reorder_src or []) + [_reorder_item_from_slug(_slug, int(_it.get("qty") or 1))]
                    _have.add(_slug)
    except Exception:
        pass  # a recommendation-merge failure must never break the portal render
```

Where `_reorder_item_from_slug` builds a `reorder_items`-shaped dict for a slug+qty using the same product/catalog lookup the reorder list already uses (read the surrounding assembly to reuse its name/price fields; price is authoritative at checkout, so a display placeholder is fine and MUST NOT be a misleading real price — mirror how the reorder list handles price display). Confirm `email` is in scope at this point in the builder; if not, resolve it the same way the surrounding code does.

- [ ] **Step 4: Run test to verify it passes**

Run: `doppler run -p remedy-match -c dev -- python -m pytest tests/test_continuity_landing.py -q`
Expected: PASS (both new tests; member-price checkout of the merged items mints one invoice)

- [ ] **Step 5: Commit**

```bash
git add app.py tests/test_continuity_landing.py
git commit -m "fix(continuity): accepted recommendation items surface in the reorder list (deduped)"
```

---

### Task 3: Client-portal accept UX copy + confirmation

**Files:**
- Modify: `static/client-portal.html` (`acceptRecommendation` + the card copy)

**Interfaces:**
- Consumes: the accept route now always returns `{ok:true}` (no `stripe_url`).

This is UI-only (no pytest cycle). The existing handler already does `if(cj.ok){ await load(); return; }`, so it functionally works after Task 1; this task makes the copy read correctly.

- [ ] **Step 1: Update the accept handler copy**

In `acceptRecommendation` (`static/client-portal.html`): change the in-progress label from "Setting up your order…" to "Adding to your order…", drop the now-dead `cj.stripe_url` redirect branch, and after `await load()` surface a brief non-blocking confirmation (e.g. a transient banner/toast "Added to your reorder list below.") near where the reorder module renders. Keep the one-shot latch and the error path.

- [ ] **Step 2: Update the card blurb**

Where the recommendation card renders (the `d.recommendation` block, ~`static/client-portal.html:680`): keep "Your practitioner recommends" + items + note; change the price line to make clear accepting adds to the reorder list (e.g. "Add these to your reorder list — your member price and shipping are calculated at checkout."). Button label "Add to my order" stays.

- [ ] **Step 3: Verify**

Extract the `<script>` and run `node --check` to confirm no JS syntax error; confirm the accept fetch still targets `/api/portal/<token>/recommendation/accept` and no longer references `stripe_url`. Report that live browser render is pending (the controller will render-verify).

- [ ] **Step 4: Commit**

```bash
git add static/client-portal.html
git commit -m "fix(continuity): accept UX reads as 'added to your reorder list' (no inline checkout)"
```

---

## Self-Review

**Spec coverage:**
- Accept mints no invoice, marks accepted → Task 1. ✓
- Accepted items merge into reorder_items (deduped) → Task 2. ✓
- Purchase via normal reorder checkout, member-priced, one invoice → asserted in Task 2's checkout test (behavior reused, unchanged). ✓
- Card only for `sent`; accepted → items in reorder, no card → already true (card renders only for `sent`); Task 2 ensures accepted items appear in reorder; Task 3 copy. ✓
- Abandon-safe / no dead-end → items persist in `reorder_items` (Task 2), asserted across a fresh payload fetch. ✓
- No duplicate → nothing minted at accept (Task 1) + `!= 'sent'` guard holds. ✓
- Persist in ongoing reorder list → Task 2 merges on every render while `accepted`. ✓

**Placeholder scan:** Task 1's accept route is complete code. Tasks 2/3 instruct the implementer to read the real `reorder_items` element shape + the sibling test fixtures and reuse them (rather than invent the reorder-item shape / portal-data fetch), and to reuse the existing member-price assertion retargeted to the reorder checkout — appropriate for edits into an existing large builder whose exact shape must be read. The retargeting of the two old accept-time member-price tests is called out explicitly in Task 1 Step 4 so no stale test asserts an invoice-at-accept.

**Type consistency:** `active_for_patient(cx, email) -> {status, items, ...}` used consistently in Tasks 1/2. Accept returns `{ok, accepted}` (Task 1), consumed by the client handler (Task 3). `reorder_items` element shape is read from source and reused via `_reorder_item_from_slug` (Task 2).

## Notes / open confirmations
- **The old accept-time member-price tests** (`test_accept_prices_at_member_price...`, `...builds_live_member_priced_checkout...`) asserted an invoice minted AT accept — that contract is intentionally removed. Task 1 Step 4 retargets/removes them; Task 2 re-asserts member pricing against the reorder checkout. Flag if any other test asserts an accept-time invoice.
- `_reorder_item_from_slug` names a small helper to build a reorder-shaped item from a slug+qty; if the surrounding builder already has such a helper, reuse it instead of adding one.
