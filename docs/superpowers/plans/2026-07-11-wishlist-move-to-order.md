# Wishlist → Move-to-Order (fold in quantities) Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Let a portal customer select wishlist items, set a quantity per item, and check them out together in one Stripe checkout at member price.

**Architecture:** Extend the existing portal checkout's entitlement gate to also accept the customer's own wishlist slugs (mirroring the accepted-recommendation union already there), then make the "Your wishlist" card rows orderable — each with a quantity stepper and a select checkbox — with one "Order selected (N)" button that posts all checked `{slug, qty}` items to the existing `/api/portal/<token>/checkout`.

**Tech Stack:** Flask (app.py), SQLite (LOG_DB), vanilla JS (static/client-portal.html), pytest.

## Global Constraints

- Reuse existing `_WISHLIST_ENABLED` flag (`WISHLIST_ENABLED` env). No new flag. Feature is dark wherever the wishlist card is dark.
- Never trust client-posted price. Pricing is server-side via `_portal_priced_lines` only (unchanged).
- Portal writes/reads key off the **portal token's** email (already lowercased as `email` in the checkout endpoint), never begin-side cookies/`amg_session`.
- Every DB access uses a dedicated `sqlite3.connect(LOG_DB)` connection. Do not re-acquire a `_db_lock` a caller already holds (non-reentrant).
- Copy rules: no em dashes, no ALL CAPS, structure/function framing only.
- Module import convention in app.py: `from dashboard import wishlist as _wl` (local import inside the function).

---

### Task 1: Backend — `slugs_for` helper + checkout entitlement union

**Files:**
- Modify: `dashboard/wishlist.py` (add `slugs_for`)
- Modify: `app.py` (`api_client_portal_checkout`, union site near line 18400)
- Test: `tests/test_wishlist.py` (add `slugs_for` cases)
- Test: `tests/test_wishlist_checkout_entitlement.py` (create — endpoint union)

**Interfaces:**
- Produces: `dashboard.wishlist.slugs_for(cx, owner) -> set[str]` — the set of slugs on `owner`'s wishlist (`owner` is a resolved key like `"email:a@x.com"`). Empty set if none.
- Consumes: existing `_portal_entitled_slugs(email)`, `_accepted_recommendation_slugs(_rcx, email)`, `_WISHLIST_ENABLED`, `LOG_DB` in app.py.

- [ ] **Step 1: Write the failing test for `slugs_for`**

Add to `tests/test_wishlist.py`:

```python
def test_slugs_for_returns_set():
    cx = _cx()
    w.toggle(cx, "email:e@x.com", "a"); w.toggle(cx, "email:e@x.com", "b")
    assert w.slugs_for(cx, "email:e@x.com") == {"a", "b"}

def test_slugs_for_empty_when_none():
    cx = _cx()
    assert w.slugs_for(cx, "email:none@x.com") == set()

def test_slugs_for_scopes_by_owner():
    cx = _cx()
    w.toggle(cx, "email:e@x.com", "a"); w.toggle(cx, "sess:s1", "b")
    assert w.slugs_for(cx, "email:e@x.com") == {"a"}
```

- [ ] **Step 2: Run it to verify it fails**

Run: `cd /tmp/wt-deploy-chat-74463de6 && python3 -m pytest tests/test_wishlist.py -q`
Expected: FAIL — `AttributeError: module 'dashboard.wishlist' has no attribute 'slugs_for'`

- [ ] **Step 3: Implement `slugs_for` in `dashboard/wishlist.py`**

Add after `list_for` (mirrors its query, returns a set):

```python
def slugs_for(cx, owner):
    """Set of slugs currently on ``owner``'s wishlist. ``owner`` is a resolved
    key ('email:<addr>' / 'sess:<id>', see resolve_owner). Empty set if none."""
    if not owner:
        return set()
    return {r[0] for r in cx.execute(
        "SELECT slug FROM wishlist WHERE owner=?", (owner,))}
```

- [ ] **Step 4: Run it to verify it passes**

Run: `cd /tmp/wt-deploy-chat-74463de6 && python3 -m pytest tests/test_wishlist.py -q`
Expected: PASS (all wishlist unit tests, including the 3 new ones).

- [ ] **Step 5: Write the failing endpoint-union test**

Create `tests/test_wishlist_checkout_entitlement.py`. This exercises the entitlement branch of `api_client_portal_checkout` without hitting Stripe/QBO by stubbing the pricing boundary and asserting the union admits a wishlist-only slug. Match the existing test style (import `app`, use Flask test client, monkeypatch module globals).

```python
import sqlite3
import app as appmod
from dashboard import wishlist as w


def _client(monkeypatch, tmp_path):
    db = str(tmp_path / "log.db")
    monkeypatch.setattr(appmod, "LOG_DB", db)
    monkeypatch.setattr(appmod, "_WISHLIST_ENABLED", True)
    # a portal token that resolves to a known email
    monkeypatch.setattr(appmod, "_portal_record_for",
                        lambda cx, token: {"email": "buyer@x.com", "content": {}})
    # buyer has NO purchase history and NO accepted recs
    monkeypatch.setattr(appmod, "_portal_entitled_slugs", lambda email: set())
    monkeypatch.setattr(appmod, "_accepted_recommendation_slugs", lambda cx, email: set())
    # stop the flow right after entitlement passes: make pricing return no lines,
    # which yields the well-defined 400 "no longer available" (NOT the
    # "isn't available to reorder" entitlement rejection we are testing against).
    monkeypatch.setattr(appmod, "_portal_priced_lines",
                        lambda items, email=None: ([], [], 0))
    # seed the buyer's wishlist
    with sqlite3.connect(db) as cx:
        w.init_wishlist_table(cx)
        w.toggle(cx, "email:buyer@x.com", "night-vision")
    return appmod.app.test_client()


def test_wishlist_slug_passes_entitlement(monkeypatch, tmp_path):
    c = _client(monkeypatch, tmp_path)
    r = c.post("/api/portal/tok/checkout", json={"items": [{"slug": "night-vision", "qty": 2}]})
    body = r.get_json()
    # entitlement passed (slug was on the wishlist) -> we reached pricing, which
    # returned [] -> the "no longer available" 400, NOT the entitlement 400.
    assert "isn't available to reorder" not in (body.get("error") or "")


def test_non_wishlist_slug_still_rejected(monkeypatch, tmp_path):
    c = _client(monkeypatch, tmp_path)
    r = c.post("/api/portal/tok/checkout", json={"items": [{"slug": "not-on-list", "qty": 1}]})
    assert r.status_code == 400
    assert "isn't available to reorder" in (r.get_json().get("error") or "")
```

- [ ] **Step 6: Run it to verify it fails**

Run: `cd /tmp/wt-deploy-chat-74463de6 && python3 -m pytest tests/test_wishlist_checkout_entitlement.py -q`
Expected: FAIL — `test_wishlist_slug_passes_entitlement` fails because without the union the wishlist-only slug is rejected with "isn't available to reorder".

- [ ] **Step 7: Add the wishlist union in `api_client_portal_checkout`**

In app.py, immediately after the accepted-recommendation union (line ~18400, the `entitled = entitled | _accepted_recommendation_slugs(_rcx, email)` block and its `except`), add:

```python
        if _WISHLIST_ENABLED:
            # The customer's own saved-and-chosen wishlist item is authorization
            # to buy it, same as an accepted recommendation. Union those slugs so
            # a wishlist-only (never-before-purchased) slug is purchasable at the
            # member price. Failure must never break checkout.
            try:
                from dashboard import wishlist as _wl
                with sqlite3.connect(LOG_DB) as _wcx:
                    _wl.init_wishlist_table(_wcx)
                    entitled = entitled | _wl.slugs_for(_wcx, "email:" + email)
            except Exception:
                pass
```

(`email` here is already the token's lowercased email. Place this inside the `if posted:` branch, alongside the existing union — not in the no-body branch.)

- [ ] **Step 8: Run both test files to verify they pass**

Run: `cd /tmp/wt-deploy-chat-74463de6 && python3 -m pytest tests/test_wishlist.py tests/test_wishlist_checkout_entitlement.py -q`
Expected: PASS.

- [ ] **Step 9: Commit**

```bash
cd /tmp/wt-deploy-chat-74463de6
git add dashboard/wishlist.py app.py tests/test_wishlist.py tests/test_wishlist_checkout_entitlement.py
git commit -m "feat: wishlist slugs authorize portal checkout (move-to-order backend)"
```

---

### Task 2: Frontend — orderable wishlist rows + combined checkout

**Files:**
- Modify: `static/client-portal.html` (wishlist card render ~1422-1445; qty-stepper handler ~1785-1795; add an `orderWishlist()` function near `reorderItem` ~3513)

**Interfaces:**
- Consumes: `d.wishlist` array (each item `{slug, name, price, url, image}`), the existing `/api/portal/<token>/checkout` endpoint (accepts `{items:[{slug,qty}]}`), the existing `token` global, `esc()`, the `.qty-ctl`/`.qtybtn`/`.qtyval` markup and CSS.
- Produces: no new server interface. Purely client-side.

- [ ] **Step 1: Make wishlist rows orderable in the render (`render`, ~1427-1443)**

Replace the `d.wishlist` card-build block so each row carries a select checkbox and a qty stepper, and the card gains an "Order selected" button. The remove (×) button stays. Keep the `.wishitem` / `data-slug` / link / image / remove markup; add the controls:

```javascript
  if(Array.isArray(d.wishlist) && d.wishlist.length){
    let wrows = "";
    d.wishlist.forEach(it=>{
      const slug = esc(it.slug||"");
      const img = it.image ? `<img class="wishitem-img" src="${esc(it.image)}" alt="" loading="lazy">` : "";
      wrows += `<div class="wishitem" data-slug="${slug}">
        <label class="wishitem-pick"><input type="checkbox" class="wishSel"></label>
        <a class="wishitem-link" href="${esc(it.url||"#")}">
          ${img}
          <span class="nm-wrap"><span class="nm">${esc(it.name||it.slug||"")}</span></span>
          <span class="pr">${esc(it.price||"")}</span>
        </a>
        <span class="qty-ctl">
          <button type="button" class="qtybtn" data-dir="-1" aria-label="Decrease quantity">−</button>
          <span class="qtyval" data-qty="1">1</span>
          <button type="button" class="qtybtn" data-dir="1" aria-label="Increase quantity">+</button>
        </span>
        <button type="button" class="wishRemBtn" data-slug="${slug}" aria-label="Remove from wishlist">×</button>
      </div>`;
    });
    html += `<div class="card" id="wishlistCard"><h2>Your wishlist</h2>
      <div>${wrows}</div>
      <button class="btn full" id="wishOrderBtn" disabled>Order selected</button>
      <p class="muted small" style="margin:.7rem 0 0">Select items, set how many, and order them together. Member price and shipping are calculated at checkout.</p>
      <p class="small err" id="wishErr" hidden></p></div>`;
  }
```

- [ ] **Step 2: Generalize the qty-stepper handler to cover wishlist rows (~1785)**

The current handler scopes to `.remitem`. Change it to scope to the nearest row that owns a `.qtyval` so it drives both `.remitem` and `.wishitem`:

```javascript
  app.querySelectorAll(".qtybtn").forEach(qb=>{
    qb.addEventListener("click", ()=>{
      const row = qb.closest(".remitem, .wishitem");
      const val = row && row.querySelector(".qtyval");
      if(!val) return;
      let q = (parseInt(val.getAttribute("data-qty"), 10) || 1) + (parseInt(qb.getAttribute("data-dir"), 10) || 0);
      if(q < 1) q = 1;
      val.setAttribute("data-qty", String(q));
      val.textContent = String(q);
    });
  });
```

- [ ] **Step 3: Wire the select checkboxes + "Order selected" button (in the same wiring block, near the `.wishRemBtn` handler ~1811)**

Add, after the `.wishRemBtn` wiring:

```javascript
  // Move-to-order: checking wishlist rows enables the combined "Order selected"
  // button and shows the live count. Clicking it folds every checked row's
  // {slug, qty} into one checkout.
  const wishOrderBtn = document.getElementById("wishOrderBtn");
  if(wishOrderBtn){
    const card = document.getElementById("wishlistCard");
    const refresh = ()=>{
      const n = card.querySelectorAll(".wishSel:checked").length;
      wishOrderBtn.disabled = n === 0;
      wishOrderBtn.textContent = n ? `Order selected (${n})` : "Order selected";
    };
    card.querySelectorAll(".wishSel").forEach(cb=> cb.addEventListener("change", refresh));
    wishOrderBtn.addEventListener("click", ()=> orderWishlist(wishOrderBtn));
    refresh();
  }
```

- [ ] **Step 4: Add the `orderWishlist` function (near `reorderItem`, ~3513)**

Mirrors `reorderItem`'s add-then-confirm + one-shot latch + error surface, but collects every checked wishlist row:

```javascript
async function orderWishlist(btn){
  // Move-to-order: post every checked wishlist row's {slug, qty} as ONE checkout
  // so the selected items are ordered together (folded), each at its chosen qty.
  // Add-then-confirm: the endpoint returns a live Stripe URL; the charge only
  // happens if the client confirms on Stripe's page. One-shot latch (disabled
  // before any await) so a double-click can't double-fire.
  if(btn.disabled) return;
  const card = document.getElementById("wishlistCard");
  const err = document.getElementById("wishErr");
  if(err) err.hidden = true;
  const items = [];
  card.querySelectorAll(".wishitem").forEach(row=>{
    const cb = row.querySelector(".wishSel");
    if(!cb || !cb.checked) return;
    const slug = row.getAttribute("data-slug") || "";
    const qv = row.querySelector(".qtyval");
    const qty = Math.max(1, parseInt(qv && qv.getAttribute("data-qty"), 10) || 1);
    if(slug) items.push({slug, qty});
  });
  if(!items.length) return;
  btn.disabled = true;
  const label = btn.textContent; btn.textContent = "Setting up your order…";
  try{
    const c = await fetch(`/api/portal/${encodeURIComponent(token)}/checkout`, {
      method: "POST",
      headers: {"Content-Type": "application/json"},
      body: JSON.stringify({items}),
    });
    const cj = await c.json();
    if(cj.stripe_url){ location.href = cj.stripe_url; return; }
    throw new Error(cj.error || "Checkout is temporarily unavailable. Please reach out to us and we’ll help.");
  }catch(e){
    if(err){ err.textContent = e.message || "Something went wrong. Please reach out and we’ll help."; err.hidden = false; }
    btn.disabled = false; btn.textContent = label;
  }
}
```

- [ ] **Step 5: Add minimal CSS for the new controls (in the `.wishitem` style block, ~202-215)**

Add rules so the checkbox and stepper lay out cleanly in the row (match the existing token vars; no new colors):

```css
  .wishitem-pick{display:inline-flex;align-items:center;margin-right:8px}
  .wishitem .qty-ctl{margin-left:auto}
```

(Adjust only if the existing `.wishitem` is a flex row; if it is not, wrap the row contents so the checkbox, link, stepper, and × sit on one line. Verify visually in Step 6.)

- [ ] **Step 6: Render-verify locally**

Boot the app against a scratch DB with the flag on, seed a wishlist for a portal email, open the portal, and confirm: checkboxes toggle the button + count, steppers change per-row qty, "Order selected (N)" posts one `{items:[...]}` (check the network request payload has all checked slugs with their quantities). Because Stripe is inactive locally the redirect will 503 — that is expected; assert the POST body is correct and the error surfaces in `#wishErr`.

Run (write a scratch boot script per the terminal-paste-wrap rule):
```bash
cd /tmp/wt-deploy-chat-74463de6
env DATA_DIR=/tmp/rv-w2 WISHLIST_ENABLED=true doppler run -p remedy-match -c prd -- python3 app.py
```
Drive the page in headless Chrome (or the browser tools): dismiss the Terms modal, locate `#wishlistCard`, check two rows, bump one stepper, click `#wishOrderBtn`, and read the outgoing `/checkout` request body.

- [ ] **Step 7: Commit**

```bash
cd /tmp/wt-deploy-chat-74463de6
git add static/client-portal.html
git commit -m "feat: orderable wishlist rows with combined move-to-order checkout"
```

---

## Self-Review

**Spec coverage:**
- "select which items" → Task 2 checkboxes ✓
- "quantity per item" → Task 2 per-row stepper ✓
- "checkout selected together in one Stripe checkout" → Task 2 `orderWishlist` posts one `{items:[...]}` ✓
- "at member price" → unchanged `_portal_priced_lines` (Task 1 leaves pricing untouched) ✓
- "wishlist item passes entitlement gate" → Task 1 union ✓
- "quantity order-time only, no schema change" → no migration in either task ✓
- "ordered items stay on wishlist" → no auto-remove added ✓
- flag-guarded + try/except → Task 1 Step 7 ✓

**Placeholder scan:** none — every step has concrete code/commands. Step 5's CSS caveat is a real verify-visually instruction, not a placeholder.

**Type consistency:** `slugs_for(cx, owner) -> set` used consistently (Task 1 Steps 3, 7). `orderWishlist(btn)` matches its call site (Task 2 Step 3). `data-qty` / `.qtyval` / `.qty-ctl` names match the existing reorder markup.
