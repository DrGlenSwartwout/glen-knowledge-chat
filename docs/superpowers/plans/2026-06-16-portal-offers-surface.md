# Portal "What's next" Offer Surface — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Fill the portal's `upgrade` seam with a role/tier-aware "What's next for you" surface showing the single next ladder rung (Live Group $99/mo → Biofield $300) with a working checkout CTA.

**Architecture:** A pure, `cx`-based offers module (`dashboard/portal_offers.py`) + eligibility resolver, wired into `get_portal_view`'s `upgrade` block and rendered in `client-portal.html`. A new standalone group-join checkout mirrors the existing studio card-vault membership flow. Whole surface gated by `PORTAL_OFFERS_ENABLED`; each rung respects its own flag.

**Tech Stack:** Flask, sqlite (DATA_DIR/chat_log.db), Stripe (setup-session card vault), pytest. Run tests via `doppler run -p remedy-match -c prd -- env DATA_DIR="$HOME/deploy-chat" ~/.venvs/deploy-chat311/bin/python -m pytest`.

**Spec:** `docs/superpowers/specs/2026-06-16-portal-offers-surface-design.md`

---

## File Structure

- **Create** `dashboard/portal_offers.py` — offer catalog + `next_offers` resolver (pure, cx-based, no `app` import).
- **Create** `tests/test_portal_offers.py` — unit tests for the resolver.
- **Modify** `dashboard/portal_view.py` — replace the `upgrade` stub with resolver output.
- **Modify** `tests/test_portal_view.py` — update the `upgrade` assertion to the new shape.
- **Modify** `app.py` — `_portal_offers_enabled()` flag, `enabled_offer_keys()` helper, group-join checkout + return routes, pass flags into `get_portal_view`.
- **Modify** `tests/test_client_portal_routes.py` — route tests for group-join + view integration.
- **Modify** `static/client-portal.html` — render the offer card + CTA.

---

## Task 1: Offer catalog + eligibility resolver

**Files:**
- Create: `dashboard/portal_offers.py`
- Test: `tests/test_portal_offers.py`

- [ ] **Step 1: Write the failing tests**

```python
# tests/test_portal_offers.py
"""The upgrade-ladder resolver: given a person, return the eligible rungs
(flag-on AND not owned) in ladder order. Pure + cx-based, mirrors portal_view."""
import sqlite3
import pytest


def _conn(tmp_path):
    from dashboard import subscriptions as subs
    from dashboard import biofield_store as bf
    cx = sqlite3.connect(str(tmp_path / "t.db"))
    subs.init_subscriptions_table(cx)
    subs.migrate_add_membership_columns(cx)
    bf.init_table(cx)
    return cx


ALL = {"live_group", "biofield"}


def test_new_client_gets_live_group_first(tmp_path):
    from dashboard import portal_offers as po
    cx = _conn(tmp_path)
    offers = po.next_offers(cx, "new@x.com", ["client"], enabled_keys=ALL)
    assert [o["key"] for o in offers] == ["live_group", "biofield"]
    assert offers[0]["price_cents"] == 9900
    assert offers[0]["checkout_path"] == "/portal/offer/live-group/checkout"


def test_group_member_skips_to_biofield(tmp_path):
    from dashboard import portal_offers as po
    from dashboard import subscriptions as subs
    cx = _conn(tmp_path)
    subs.create_membership(cx, email="m@x.com", stripe_customer_id="c",
                           stripe_payment_method_id="pm", amount_cents=9900,
                           next_charge_date="2026-07-16")
    offers = po.next_offers(cx, "m@x.com", ["client"], enabled_keys=ALL)
    assert [o["key"] for o in offers] == ["biofield"]


def test_owns_both_returns_empty(tmp_path):
    from dashboard import portal_offers as po
    from dashboard import subscriptions as subs
    from dashboard import biofield_store as bf
    cx = _conn(tmp_path)
    subs.create_membership(cx, email="b@x.com", stripe_customer_id="c",
                           stripe_payment_method_id="pm", amount_cents=9900,
                           next_charge_date="2026-07-16")
    bf.seed_paid(cx, "b@x.com", via="checkout", order_ref="o1")
    assert po.next_offers(cx, "b@x.com", ["client"], enabled_keys=ALL) == []


def test_flag_off_rung_is_excluded(tmp_path):
    from dashboard import portal_offers as po
    cx = _conn(tmp_path)
    # only biofield flag on -> live_group hidden even though unowned
    offers = po.next_offers(cx, "new@x.com", ["client"], enabled_keys={"biofield"})
    assert [o["key"] for o in offers] == ["biofield"]
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `... -m pytest tests/test_portal_offers.py -q`
Expected: FAIL — `ImportError: cannot import name 'portal_offers'`.

- [ ] **Step 3: Write the module**

```python
# dashboard/portal_offers.py
"""Upgrade-ladder offer catalog + eligibility resolver.

next_offers() returns the ladder rungs a person is eligible for — flag-on AND
not already owned — in ladder order. The portal surfaces the FIRST (single next
rung). Pure + cx-based; never imports app, so it unit-tests in isolation.
"""
from dashboard import subscriptions as _subs
from dashboard import biofield_store as _bf

MEMBERSHIP_PRICE_CENTS = 9900
BIOFIELD_PRICE_CENTS = 30000


def _owns_group(cx, email):
    try:
        _subs.init_subscriptions_table(cx)
        _subs.migrate_add_membership_columns(cx)
        return bool(_subs.active_memberships_by_email(cx, email))
    except Exception:
        return False


def _owns_biofield(cx, email):
    try:
        _bf.init_table(cx)
        row = cx.execute(
            "SELECT paid_at FROM biofield_readiness WHERE lower(email)=lower(?)",
            (str(email or "").strip(),)).fetchone()
        return bool(row and row[0])
    except Exception:
        return False


# Ladder order. Each rung: a static descriptor + an owned(cx,email) predicate.
_LADDER = [
    {"key": "live_group", "title": "Join the Live Group", "price_cents": MEMBERSHIP_PRICE_CENTS,
     "period": "/mo", "blurb": "Live group coaching with Dr. Glen — your next step on the path.",
     "cta_label": "Join", "checkout_path": "/portal/offer/live-group/checkout",
     "owned": _owns_group},
    {"key": "biofield", "title": "Causal Biofield Analysis", "price_cents": BIOFIELD_PRICE_CENTS,
     "period": "", "blurb": "A personalized Biofield-designed program reading your causal chain.",
     "cta_label": "Book", "checkout_path": "/biofield/checkout",
     "owned": _owns_biofield},
]

# Public rung view (drops the predicate).
def _public(rung):
    return {k: rung[k] for k in ("key", "title", "price_cents", "period", "blurb",
                                 "cta_label", "checkout_path")}


def next_offers(cx, email, roles, *, enabled_keys):
    """Eligible rungs (key in enabled_keys AND not owned), in ladder order."""
    email = (email or "").strip().lower()
    out = []
    for rung in _LADDER:
        if rung["key"] not in enabled_keys:
            continue
        if rung["owned"](cx, email):
            continue
        out.append(_public(rung))
    return out
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `... -m pytest tests/test_portal_offers.py -q`
Expected: PASS (4 passed). If `seed_paid`/`active_memberships_by_email` signatures differ, adjust the test seeds to match the real ones (verified at plan time: `seed_paid(cx, email, *, via, order_ref)`, `create_membership(cx, *, email, stripe_customer_id, stripe_payment_method_id, amount_cents, next_charge_date, cadence_months=1)`).

- [ ] **Step 5: Commit**

```bash
git add dashboard/portal_offers.py tests/test_portal_offers.py
git commit -m "Portal offers: ladder catalog + next_offers resolver"
```

---

## Task 2: Wire the resolver into `get_portal_view`

**Files:**
- Modify: `dashboard/portal_view.py` (the `upgrade` line and `get_portal_view` signature)
- Test: `tests/test_portal_view.py`

- [ ] **Step 1: Update the existing upgrade-stub test + add a resolver test**

In `tests/test_portal_view.py`, change the stub assertion in
`test_view_composes_account_orders_points_and_stub` from:

```python
    assert view["upgrade"] == {"enabled": False, "placeholder": True}
```

to:

```python
    # no offer flags passed -> upgrade disabled (block hidden)
    assert view["upgrade"] == {"enabled": False}
```

Then add:

```python
def test_view_surfaces_first_eligible_offer(tmp_path):
    from dashboard import portal_view as pv
    cx = _conn(tmp_path)
    pid = _add_person(cx, "off@example.com", "Offer Client")
    view = pv.get_portal_view(cx, pid, offers_enabled_keys={"live_group", "biofield"})
    assert view["upgrade"]["enabled"] is True
    assert view["upgrade"]["offer"]["key"] == "live_group"
    assert view["upgrade"]["offer"]["price_cents"] == 9900
```

- [ ] **Step 2: Run to verify the new test fails**

Run: `... -m pytest tests/test_portal_view.py::test_view_surfaces_first_eligible_offer -q`
Expected: FAIL — `get_portal_view() got an unexpected keyword argument 'offers_enabled_keys'`.

- [ ] **Step 3: Implement**

In `dashboard/portal_view.py`, add the import near the top:

```python
from dashboard import portal_offers as _po
```

Change the signature:

```python
def get_portal_view(cx, person_id, *, offers_enabled_keys=None):
```

Replace the `upgrade` line in the returned dict:

```python
        "upgrade": _upgrade_block(cx, email, roles, offers_enabled_keys),
```

And add the helper above `get_portal_view`:

```python
def _upgrade_block(cx, email, roles, enabled_keys):
    """The single next eligible ladder rung, or disabled when none/flags off."""
    if not enabled_keys:
        return {"enabled": False}
    try:
        offers = _po.next_offers(cx, email, roles, enabled_keys=enabled_keys)
    except Exception:
        offers = []
    if not offers:
        return {"enabled": False}
    return {"enabled": True, "offer": offers[0]}
```

- [ ] **Step 4: Run the portal_view tests**

Run: `... -m pytest tests/test_portal_view.py -q`
Expected: PASS (all, including the updated stub assertion).

- [ ] **Step 5: Commit**

```bash
git add dashboard/portal_view.py tests/test_portal_view.py
git commit -m "Portal view: surface next eligible offer in the upgrade block"
```

---

## Task 3: Group-join checkout + return routes

**Files:**
- Modify: `app.py` (add flag helper + two routes near the other portal routes, after `client_portal_me`)
- Test: `tests/test_client_portal_routes.py`

- [ ] **Step 1: Write the failing route tests**

Add to `tests/test_client_portal_routes.py`:

```python
# ── Group-join offer checkout (mirrors the studio card-vault flow) ───────────

def test_group_join_checkout_dark_by_default(client):
    c, _ = client
    assert c.post("/portal/offer/live-group/checkout").status_code == 404


def test_group_join_checkout_returns_stripe_url(client, monkeypatch):
    c, appmod = client
    monkeypatch.setattr(appmod, "_portal_offers_enabled", lambda: True)
    tok = _seed_portal(appmod, email="gj@example.com", name="GJ")
    from dashboard import stripe_pay
    monkeypatch.setattr(stripe_pay, "create_setup_session",
                        lambda **k: {"url": "https://checkout.stripe/grp"})
    r = c.post(f"/portal/offer/live-group/checkout?token={tok}")
    assert r.status_code == 200
    assert r.get_json()["stripe_url"] == "https://checkout.stripe/grp"


def test_group_join_return_creates_membership(client, monkeypatch):
    c, appmod = client
    monkeypatch.setattr(appmod, "_portal_offers_enabled", lambda: True)
    from dashboard import stripe_pay, subscriptions as subs
    monkeypatch.setattr(stripe_pay, "get_session",
                        lambda sid: {"metadata": {"email": "gj2@example.com"},
                                     "setup_intent": "si_1"})
    monkeypatch.setattr(stripe_pay, "get_setup_intent",
                        lambda si: {"customer": "cus_1", "payment_method": "pm_1"})
    r = c.get("/portal/offer/live-group/return?session_id=sess_1",
              follow_redirects=False)
    assert r.status_code in (302, 303)
    import sqlite3 as _sq
    cx = _sq.connect(appmod.LOG_DB)
    subs.init_subscriptions_table(cx); subs.migrate_add_membership_columns(cx)
    assert subs.active_memberships_by_email(cx, "gj2@example.com")
    cx.close()
```

- [ ] **Step 2: Run to verify they fail**

Run: `... -m pytest tests/test_client_portal_routes.py -k group_join -q`
Expected: FAIL — `_portal_offers_enabled` missing / routes 404 when enabled.

- [ ] **Step 3: Implement the flag + routes**

In `app.py`, add near `_client_login_enabled()`:

```python
def _portal_offers_enabled() -> bool:
    """Master flag for the portal 'What's next' offer surface. Dark by default."""
    return os.environ.get("PORTAL_OFFERS_ENABLED", "").strip().lower() in (
        "1", "true", "yes", "on")


def _enabled_offer_keys() -> set:
    """Which ladder rungs are purchasable right now (master + per-rung flags)."""
    if not _portal_offers_enabled():
        return set()
    keys = set()
    if os.environ.get("SUBSCRIPTIONS_ENABLED", "").strip().lower() in ("1", "true", "yes", "on"):
        keys.add("live_group")
    if os.environ.get("BIOFIELD_CHECKOUT_ENABLED", "").strip().lower() in ("1", "true", "yes", "on"):
        keys.add("biofield")
    return keys
```

Add the two routes after `client_portal_me`:

```python
@app.route("/portal/offer/live-group/checkout", methods=["POST"])
def portal_group_join_checkout():
    """Start a $99/mo Live Group membership: $0 Stripe setup session to vault the
    card; the membership row is created on return. Mirrors /api/studio/claim."""
    if not _portal_offers_enabled():
        return jsonify({"error": "not found"}), 404
    token = request.args.get("token", "") or (request.get_json(silent=True) or {}).get("token", "")
    sess_cookie = request.cookies.get("rm_portal_session", "")
    from dashboard import portal_identity as _pi
    with sqlite3.connect(LOG_DB) as cx:
        ident = _pi.resolve_identity(cx, token=token, session_token=sess_cookie,
                                     client_login_enabled=_client_login_enabled())
    if ident is None:
        return jsonify({"error": "not found"}), 404
    try:
        from dashboard import stripe_pay
        sess = stripe_pay.create_setup_session(
            customer_email=ident.email,
            metadata={"kind": "group_join", "email": ident.email},
            success_url=(f"{PUBLIC_BASE_URL}/portal/offer/live-group/return"
                         f"?session_id={{CHECKOUT_SESSION_ID}}"),
            cancel_url=f"{PUBLIC_BASE_URL}/portal/me")
    except Exception as e:
        app.logger.exception("group-join setup session failed")
        return jsonify({"error": "Could not start checkout. Please reach out and we'll help."}), 502
    return jsonify({"ok": True, "stripe_url": sess.get("url", "")})


@app.route("/portal/offer/live-group/return")
def portal_group_join_return():
    """Stripe setup return: vault the card and create the membership (first charge
    one cycle out, billed by the subscriptions scheduler). Idempotent; never 500s."""
    from flask import redirect as _redir
    sid = (request.args.get("session_id") or "").strip()
    if _portal_offers_enabled() and sid:
        try:
            from dashboard import stripe_pay, subscriptions as _subs, portal_offers as _po
            import datetime as _dt
            sess = stripe_pay.get_session(sid)
            email = ((sess.get("metadata") or {}).get("email") or "").strip().lower()
            si = stripe_pay.get_setup_intent(sess.get("setup_intent"))
            cus, pm = si.get("customer"), si.get("payment_method")
            with sqlite3.connect(LOG_DB) as cx:
                cx.row_factory = sqlite3.Row
                _subs.init_subscriptions_table(cx)
                _subs.migrate_add_membership_columns(cx)
                if email and cus and pm and not _subs.active_memberships_by_email(cx, email):
                    next_date = _subs.add_months(_dt.date.today().isoformat(), 1)
                    _subs.create_membership(
                        cx, email=email, stripe_customer_id=cus,
                        stripe_payment_method_id=pm,
                        amount_cents=_po.MEMBERSHIP_PRICE_CENTS, next_charge_date=next_date)
        except Exception as e:
            print(f"[group-join] return failed: {e!r}", flush=True)
    return _redir("/portal/me?joined=1")
```

- [ ] **Step 4: Run the route tests**

Run: `... -m pytest tests/test_client_portal_routes.py -k group_join -q`
Expected: PASS (3 passed).

- [ ] **Step 5: Commit**

```bash
git add app.py tests/test_client_portal_routes.py
git commit -m "Portal offers: PORTAL_OFFERS_ENABLED + group-join checkout/return"
```

---

## Task 4: Wire the view endpoint to pass offer flags + integration test

**Files:**
- Modify: `app.py` (the `/api/portal/<token>/view` handler — pass `offers_enabled_keys`)
- Test: `tests/test_client_portal_routes.py`

- [ ] **Step 1: Write the failing integration test**

Add to `tests/test_client_portal_routes.py`:

```python
def test_view_endpoint_includes_eligible_offer(client, monkeypatch):
    c, appmod = client
    monkeypatch.setattr(appmod, "_enabled_offer_keys", lambda: {"live_group", "biofield"})
    _seed_person(appmod, "vo@example.com", "VO")
    tok = _seed_portal(appmod, email="vo@example.com", name="VO")
    j = c.get(f"/api/portal/{tok}/view").get_json()
    assert j["upgrade"]["enabled"] is True
    assert j["upgrade"]["offer"]["key"] == "live_group"
```

- [ ] **Step 2: Run to verify it fails**

Run: `... -m pytest tests/test_client_portal_routes.py::test_view_endpoint_includes_eligible_offer -q`
Expected: FAIL — `upgrade.enabled` is False (flags not passed through).

- [ ] **Step 3: Implement**

In `app.py`, in `api_client_portal_view`, change the `get_portal_view` call:

```python
        view = _pv.get_portal_view(cx, ident.person_id,
                                   offers_enabled_keys=_enabled_offer_keys())
```

- [ ] **Step 4: Run the test**

Run: `... -m pytest tests/test_client_portal_routes.py::test_view_endpoint_includes_eligible_offer -q`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add app.py tests/test_client_portal_routes.py
git commit -m "Portal offers: view endpoint passes enabled offer keys"
```

---

## Task 5: Render the offer card in the portal page

**Files:**
- Modify: `static/client-portal.html` (replace the `.soon` placeholder block, lines ~190-194)

- [ ] **Step 1: Replace the placeholder rendering**

Find:

```javascript
  // Sales / upgrade — reserved seam (feature #2). Renders a quiet placeholder so
  // the section exists in the shell before it has content.
  if(v && v.upgrade && v.upgrade.placeholder){
    html += `<div class="card soon"><h2>What's next for you</h2>
      <p class="muted small">Personalized next steps and offers are on their way to your healing home.</p></div>`;
  }
```

Replace with:

```javascript
  // Sales / upgrade (feature #2): the single next eligible ladder rung + CTA.
  if(v && v.upgrade && v.upgrade.enabled && v.upgrade.offer){
    const o = v.upgrade.offer;
    const price = esc(money(o.price_cents)) + (o.period?esc(o.period):"");
    html += `<div class="card"><h2>What's next for you</h2>
      <div class="reitem"><span class="nm">${esc(o.title)}</span><span class="pr sp">${price}</span></div>
      <p class="muted small" style="margin:.5rem 0 1rem">${esc(o.blurb)}</p>
      <button class="btn full" id="offerBtn" data-path="${esc(o.checkout_path)}">${esc(o.cta_label)}</button>
      <p class="small err" id="offerErr" hidden></p></div>`;
  }
```

- [ ] **Step 2: Add the CTA handler**

After the existing `if(btn) btn.addEventListener("click", reorder);` line in `render()`, add:

```javascript
  const obtn = document.getElementById("offerBtn");
  if(obtn) obtn.addEventListener("click", ()=>startOffer(obtn));
```

And add this function next to `reorder()`:

```javascript
async function startOffer(btn){
  const err = document.getElementById("offerErr");
  const path = btn.getAttribute("data-path");
  err.hidden = true; btn.disabled = true; const label = btn.textContent; btn.textContent = "One moment…";
  try{
    // Carry the same path segment ("me" or a token) so the server resolves identity.
    const url = `${path}${path.includes("?")?"&":"?"}token=${encodeURIComponent(seg)}`;
    const r = await fetch(url, {method:"POST"});
    const j = await r.json();
    if(j.stripe_url){ location.href = j.stripe_url; return; }
    throw new Error(j.error || "This isn't available right now. Please reach out and we'll help.");
  }catch(e){
    err.textContent = e.message; err.hidden = false; btn.disabled = false; btn.textContent = label;
  }
}
```

Note: the biofield rung's `checkout_path` is `/biofield/checkout`; confirm during implementation that it accepts a POST with `?token=`/session for identity, or adjust `startOffer` to the biofield checkout's expected payload. If biofield checkout needs a different shape, branch on `o.key` in `startOffer`.

- [ ] **Step 3: Manual smoke (no JS unit harness)**

Run the app locally or rely on Task 6's full-suite + the route tests. The page-served test already covers a 200.

- [ ] **Step 4: Commit**

```bash
git add static/client-portal.html
git commit -m "Portal page: render the What's next offer card + CTA"
```

---

## Task 6: Full suite, push, PR

- [ ] **Step 1: Run the whole suite**

Run: `doppler run -p remedy-match -c prd -- env DATA_DIR="$HOME/deploy-chat" ~/.venvs/deploy-chat311/bin/python -m pytest -q`
Expected: all pass (≥1700), 0 failed.

- [ ] **Step 2: Push + open PR**

```bash
git push -u origin sess/5326cc61
gh pr create --base main --title "Feature #2 slice 1: portal What's-next offer surface" --body "..."
```

---

## Self-Review notes

- **Spec coverage:** offer catalog (Task 1), resolver (Task 1), group-join checkout (Task 3), view wiring (Tasks 2,4), page render (Task 5), master+per-rung flags (Task 3), graceful empty (Task 2 `_upgrade_block`). Cert/$149/menu/standalone pages explicitly out of scope.
- **Type consistency:** `next_offers(cx, email, roles, *, enabled_keys)`; offer dict keys `key/title/price_cents/period/blurb/cta_label/checkout_path`; `get_portal_view(cx, person_id, *, offers_enabled_keys=None)`; `upgrade` block shape `{"enabled": bool, "offer": {...}?}` — used identically in view, endpoint, tests, and HTML.
- **Open verification (do during impl):** confirm `biofield_store.seed_paid` and `subscriptions.add_months` signatures (used in tests/return route) match; confirm `/biofield/checkout` accepts the `startOffer` POST shape (branch by `o.key` if not).
