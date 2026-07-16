# Shopper "Subscribe & Save" CTA — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a "Subscribe & save — save more each time" option to the buy page for every autoship-eligible product, so a paid member can start an autoship subscription (via the existing `POST /reorder/subscribe`), and route non-paid visitors to become a member.

**Architecture:** One small backend addition exposes the flags the page needs (`begin_product_data`); the rest is frontend in `static/begin-buy.html` — a purchase-mode toggle that reuses the page's existing address form and Stripe-redirect pattern, POSTing to the already-built subscribe endpoint. No pricing/ladder/backend logic changes (shipped in #936).

**Tech Stack:** Python/Flask (`app.py`), vanilla JS in a static HTML page, pytest, headless Chrome for frontend verification (no JS unit framework in this repo).

## Global Constraints

- **Paid-member benefit:** the Subscribe flow is for `is_paid_member` viewers only; non-paid → route to `/membership`.
- **Autoship-eligible products** get the CTA: **bundles** with `autoship_eligible: true` (ladder 12→29) and **single-SKU Functional Formulations** — `_qty_eligible(p)` = `qty_pricing` and not `info_only` (ladder 3→25). Device bundles, non-FF single SKUs (ionizers/nightlights), and `info_only` → no CTA.
- **Ladder percentages come only from server data** (`data.autoship.first_pct` / `cap_pct`) — never hardcode 12/29/3/25 in JS.
- **Subscribe is card-only** (Stripe card vaulting). In Subscribe mode the Zelle/Wise method selector does not apply; `POST /reorder/subscribe` never takes a `method`.
- **Subscribe request contract:** `POST /reorder/subscribe` body `{items:[{slug, qty}], address:{name,street,city,state,zip,country}, cadence_months: 1|2|3}`; auth is the signed-in `rm_reorder_email` cookie (401 `not signed in`) + `is_member` ToS (403 `need_optin`); returns `{ok, stripe_url}`.
- **Cadence vocabulary** matches the manage portal exactly: 1 Monthly / 2 Every 2 months / 3 Quarterly.
- **Copy:** "Subscribe & save — save more each time" — no em dashes, no ALL CAPS.
- deploy-chat is merge=deploy, no CI — render-verify live after deploy.

---

## Task 1: Backend — expose autoship fields in product-data

**Files:**
- Modify: `app.py` — `begin_product_data` (the `data` dict at `app.py:6358-6373`)
- Test: `tests/test_product_data_autoship_fields.py`

**Interfaces:**
- Produces: `/begin/product-data/<slug>` response gains `autoship_eligible: bool`, `bundle: bool`, `is_paid_member: bool`, and (when autoship-eligible) `autoship: {first_pct: int, cap_pct: int}`.

- [ ] **Step 1: Write the failing test** (imports `app` → run under Doppler)

```python
# tests/test_product_data_autoship_fields.py
import pytest
app = pytest.importorskip("app")

def _data(slug):
    with app.app.test_client() as c:
        r = c.get(f"/begin/product-data/{slug}")
        assert r.status_code == 200, (slug, r.status_code)
        return r.get_json()

def test_bundle_exposes_bundle_ladder():
    d = _data("crystalline-lens-program")
    assert d["autoship_eligible"] is True
    assert d["bundle"] is True
    assert d["autoship"]["first_pct"] == 12
    assert d["autoship"]["cap_pct"] == 29

def test_ff_single_sku_exposes_standard_ladder():
    d = _data("wholomega")  # an FF single SKU (qty_pricing) -> autoship-eligible
    assert d["autoship_eligible"] is True
    assert d["bundle"] is False
    assert d["autoship"]["first_pct"] == 3
    assert d["autoship"]["cap_pct"] == 25

def test_device_bundle_not_autoship_eligible():
    d = _data("dental-bundle")
    assert d["autoship_eligible"] is False
    assert "autoship" not in d  # no ladder block for non-eligible

def test_non_ff_device_single_sku_not_eligible():
    d = _data("water-ionizer-5plate")  # not an FF (qty_pricing unset) -> excluded
    assert d["autoship_eligible"] is False
    assert "autoship" not in d

def test_is_paid_member_present_and_false_for_anon():
    d = _data("crystalline-lens-program")
    assert d["is_paid_member"] is False  # no auth cookie in the test client
```

- [ ] **Step 2: Run — expect FAIL**

Run: `doppler run --project remedy-match --config dev -- python3 -m pytest tests/test_product_data_autoship_fields.py -q`
Expected: FAIL — `KeyError: 'autoship_eligible'` (fields not added yet). (App-import test; run under Doppler or it silently skips.)

- [ ] **Step 3: Add the fields** in `begin_product_data`, immediately after the `data = { … }` literal closes (after `app.py:6373`, before the `try:` founding block at `:6374`):

```python
    _viewer_email = ((get_authenticated_user(request) or {}).get("email", "")
                     or request.cookies.get("rm_reorder_email", ""))
    # Autoship eligibility: a bundle must carry autoship_eligible (device bundles are
    # false); a single SKU must be a Functional Formulation (_qty_eligible = qty_pricing
    # and not info_only), which excludes devices (ionizers/nightlights) and services.
    _autoship_ok = (bool(p.get("autoship_eligible")) if p.get("bundle")
                    else _qty_eligible(p))
    data["autoship_eligible"] = _autoship_ok
    data["bundle"] = bool(p.get("bundle"))
    data["is_paid_member"] = _is_paid_member(_viewer_email)
    if _autoship_ok:
        from dashboard import subscriptions as _subs
        if p.get("bundle"):
            data["autoship"] = {"first_pct": _subs.tier_for_bundle(0),
                                "cap_pct": _subs.tier_for_bundle(99)}
        else:
            data["autoship"] = {"first_pct": _subs.tier_for(0),
                                "cap_pct": _subs.tier_for(99)}
```

(`get_authenticated_user`, `_is_paid_member`, and `_get_product` are already used in this function/module; `tier_for`/`tier_for_bundle` clamp at their caps, so `(99)` yields 25/29.)

- [ ] **Step 4: Run — expect PASS**

Run: `doppler run --project remedy-match --config dev -- python3 -m pytest tests/test_product_data_autoship_fields.py -q`
Expected: 4 passed.

- [ ] **Step 5: Commit**

```bash
git add app.py tests/test_product_data_autoship_fields.py
git commit -m "feat: expose autoship_eligible/bundle/is_paid_member/ladder in product-data"
```

---

## Task 2: Frontend — Subscribe & Save CTA on begin-buy.html

**Files:**
- Modify: `static/begin-buy.html` (markup near `:341-346`; JS `renderProduct` `:443`, buy-button listener `~:640`, add `placeSubscription`)

**Interfaces:**
- Consumes: the Task 1 product-data fields.
- Produces: three CTA states (paid → subscribe flow; non-paid → join teaser; ineligible → nothing).

Note: no JS unit framework exists in this repo. This task is verified by driving the page in headless Chrome (Step 6), not a pytest.

- [ ] **Step 1: Add the CTA markup** — insert this block immediately BEFORE `<div class="buy-wrap">` (`static/begin-buy.html:343`), so it sits above the Place-order button:

```html
        <!-- Subscribe & Save (populated by renderSubscribeCta from product-data) -->
        <div id="subscribe-cta" class="subscribe-cta hidden">
          <!-- State A: paid member -->
          <div id="sub-modes" class="sub-modes hidden">
            <label class="sub-mode"><input type="radio" name="purchase-mode" value="once" checked /> <span>One-time purchase</span></label>
            <label class="sub-mode sub-mode-hi"><input type="radio" name="purchase-mode" value="subscribe" /> <span id="sub-mode-label">Subscribe &amp; save</span></label>
            <div id="sub-options" class="sub-options hidden">
              <label for="sub-cadence">Delivery</label>
              <select id="sub-cadence">
                <option value="1">Monthly</option>
                <option value="2">Every 2 months</option>
                <option value="3">Quarterly</option>
              </select>
              <p class="sub-ladder" id="sub-ladder"></p>
            </div>
          </div>
          <!-- State B: not a paid member -->
          <div id="sub-join" class="sub-join hidden">
            <p id="sub-join-copy"></p>
            <button type="button" class="buy-btn sub-join-btn" id="sub-join-btn">Become a member</button>
          </div>
        </div>
```

- [ ] **Step 2: Add the render + submit JS** — inside the page script, add these functions (near `placeOrder`, `static/begin-buy.html:~738`). `subscribeMode()` reads the toggle; `renderSubscribeCta(p)` sets the state; `placeSubscription()` mirrors `placeOrder` but posts to the subscribe endpoint:

```js
    // ── Subscribe & Save ───────────────────────────────────────────────────
    function subscribeMode(){
      var r = document.querySelector('input[name="purchase-mode"]:checked');
      return !!r && r.value === 'subscribe';
    }

    function renderSubscribeCta(p){
      var wrap = document.getElementById('subscribe-cta');
      if (!p || !p.autoship_eligible) { wrap.classList.add('hidden'); return; }
      wrap.classList.remove('hidden');
      if (p.is_paid_member) {
        document.getElementById('sub-modes').classList.remove('hidden');
        document.getElementById('sub-join').classList.add('hidden');
        var a = p.autoship || {};
        document.getElementById('sub-ladder').textContent =
          'Save ' + a.first_pct + '% on your first shipment, climbing to ' + a.cap_pct +
          '% — save more each time. Cancel anytime.';
        // Toggle: show cadence when Subscribe is chosen; retitle the buy button.
        document.querySelectorAll('input[name="purchase-mode"]').forEach(function(el){
          el.addEventListener('change', function(){
            document.getElementById('sub-options').classList.toggle('hidden', !subscribeMode());
            var b = document.getElementById('buy-btn');
            b.textContent = subscribeMode() ? 'Start my subscription' : 'Place your order';
          });
        });
      } else {
        document.getElementById('sub-modes').classList.add('hidden');
        var j = document.getElementById('sub-join');
        j.classList.remove('hidden');
        var a2 = p.autoship || {};
        document.getElementById('sub-join-copy').textContent =
          'Members save ' + a2.first_pct + '–' + a2.cap_pct +
          '% with Subscribe & Save on this product.';
        document.getElementById('sub-join-btn').onclick = function(){ window.location.href = '/membership'; };
      }
    }

    function placeSubscription(){
      if (ordering) return;
      var _street = (document.getElementById('ship-street').value || '').trim();
      var _city = (document.getElementById('ship-city').value || '').trim();
      var _state = (document.getElementById('ship-state').value || '').trim();
      var _zip = (document.getElementById('ship-zip').value || '').trim();
      if (!_street || !_city || !_state || !_zip) {
        setInlineMsg('Please add your shipping address so we can send your shipments.');
        document.getElementById('ship-street').focus();
        return;
      }
      ordering = true;
      var buyBtn = document.getElementById('buy-btn');
      buyBtn.disabled = true;
      buyBtn.textContent = 'Setting up your subscription…';
      clearInlineMsg();
      var qty = parseInt(document.getElementById('qty').value, 10); if (!qty || qty < 1) qty = 1;
      var payload = {
        items: [{ slug: slug, qty: qty }],
        cadence_months: parseInt(document.getElementById('sub-cadence').value, 10) || 1,
        address: { name: (document.getElementById('name').value || '').trim(),
                   street: _street, city: _city, state: _state, zip: _zip, country: 'US' }
      };
      fetch(BASE + '/reorder/subscribe', {
        method: 'POST', headers: { 'Content-Type': 'application/json' },
        credentials: 'same-origin', body: JSON.stringify(payload)
      })
      .then(function(r){ return r.json().catch(function(){ return { ok:false, error:'HTTP ' + r.status }; }); })
      .then(function(data){
        ordering = false;
        buyBtn.textContent = 'Start my subscription';
        if (data && data.need_optin && window.OptinGate) {
          OptinGate.show({ base: BASE, onAgree: function(){ placeSubscription(); } });
          return;
        }
        if (data && data.error === 'not signed in') {
          setInlineMsg('Please sign in (check your email for a sign-in link) to set up Subscribe & Save.');
          refreshBuyState();
          return;
        }
        if (!data || !data.ok) {
          setInlineMsg((data && data.error) ? data.error : 'Could not start your subscription. Please try again.');
          refreshBuyState();
          return;
        }
        if (data.stripe_url) { window.location.href = data.stripe_url; return; }
        setInlineMsg('Subscription created, but card setup is unavailable right now. Please try again shortly.');
        refreshBuyState();
      })
      .catch(function(e){
        ordering = false;
        buyBtn.textContent = 'Start my subscription';
        setInlineMsg('Connection hiccup: ' + e.message + '. Please try again.');
        refreshBuyState();
      });
    }
```

- [ ] **Step 3: Call the renderer** — in `renderProduct` (or right after `product = data;` at `static/begin-buy.html:434-435`), add a call so the CTA renders when product-data lands:

```js
        product = data;
        renderProduct(data);
        renderSubscribeCta(data);   // NEW
```

- [ ] **Step 4: Route the buy button** — the buy button currently binds directly to `placeOrder` (`static/begin-buy.html:~640`, `buyBtn.addEventListener('click', placeOrder)`). Replace that binding with a dispatcher so Subscribe mode routes to `placeSubscription`:

```js
      document.getElementById('buy-btn').addEventListener('click', function(){
        if (subscribeMode()) { placeSubscription(); } else { placeOrder(); }
      });
```

(Find the existing `addEventListener('click', placeOrder)` and replace it with the above. If the binding differs, adapt but preserve one-time behavior when not in subscribe mode.)

- [ ] **Step 5: Minimal styles** — add to the page's `<style>` so the block is legible (match existing `.buy-btn`/`.field` styling; keep it simple):

```css
    .subscribe-cta{ margin:14px 0; }
    .subscribe-cta.hidden,.sub-modes.hidden,.sub-options.hidden,.sub-join.hidden{ display:none; }
    .sub-modes .sub-mode{ display:block; margin:6px 0; cursor:pointer; }
    .sub-mode-hi{ font-weight:600; }
    .sub-options{ margin:8px 0 4px; }
    .sub-ladder{ font-size:.9em; opacity:.85; margin:6px 0 0; }
    .sub-join{ border:1px solid rgba(0,0,0,.12); border-radius:10px; padding:12px; }
```

- [ ] **Step 6: Verify headless (3 states)** — use the webapp-testing / render-verify approach to drive `/begin/buy/<slug>` in headless Chrome against a locally-running app (or a stubbed `/begin/product-data` response):
  - Bundle + paid member (stub `is_paid_member:true, autoship_eligible:true, bundle:true, autoship:{first_pct:12,cap_pct:29}`): toggle appears; selecting "Subscribe & save" reveals the cadence select and the ladder line reads "Save 12% … climbing to 29% …"; the buy button relabels to "Start my subscription"; clicking it fires a POST to `/reorder/subscribe` with `{items,address,cadence_months}` (observe the network request).
  - Single SKU + paid (`bundle:false, autoship:{3,25}`): ladder reads 3 → 25.
  - Non-paid (`is_paid_member:false`): no toggle; "Become a member" button navigates to `/membership`.
  - Device bundle (`autoship_eligible:false`): the `#subscribe-cta` block stays hidden; one-time buy unchanged.
  Capture a screenshot/GIF of the paid-member bundle state for the record.

- [ ] **Step 7: Commit**

```bash
git add static/begin-buy.html
git commit -m "feat: Subscribe & Save CTA on begin-buy (paid-member toggle + cadence, join teaser for non-members)"
```

---

## Rollout / verification

- [ ] **Prod flags (check prod, not dev):** confirm `SUBSCRIPTIONS_ENABLED`, `PRICING_ENGINE_CHECKOUT`, Stripe-active are ON, and `MEMBERSHIP_PRODUCTS_ENABLED` is ON (else State B's `/membership` 404s — repoint to `/begin/ascend`).
- [ ] **Live render-verify after deploy:** load `/begin/buy/crystalline-lens-program` as a paid member → the Subscribe toggle + 12→29 ladder shows; as anon → the "Become a member" teaser shows; `/begin/buy/dental-bundle` → no CTA.
- [ ] **End-to-end (one real paid member):** start a subscription → redirected to Stripe → after return, a `subscriptions` row exists priced at the bundle ladder (verify in the manage portal `/subscription` or DB), and the charge-cron will apply 12→29 on renewals.

## Non-goals

- Managing/cancelling subscriptions (exists at `/subscription`).
- Any pricing/ladder/backend change (done in #936).
- A CTA on `static/begin-product.html` (no purchase controls there).
