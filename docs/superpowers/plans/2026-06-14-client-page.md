# Branded Client Page (patient-paid) — Implementation Plan (Plan 3 of 4)

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development. Steps use checkbox (`- [ ]`) syntax.

**Goal:** The patient-facing client page — a patient reaches the practitioner's `/dispensary/<code>` link, buys Functional Formulations at the **practitioner's price** (≥ MAP $67, flat, no volume), pays us, and we ship to them; the practitioner's **margin** (S − base − fee) is credited to their **wallet**. This reworks today's dispensary flow (patient buys RM retail → flat $20/bottle) into the margin model. Branding is Plan 4; this plan makes it functional (default practitioner prices = RM retail until Plan 4's price-setting UI exists).

**Architecture:** New `build_client_order` in `dashboard/dropship_checkout.py` (patient-paid sibling of `build_dropship_order`: QBO customer = the **patient**, priced at the practitioner's S, returns the **margin** to credit). A patient-paid checkout route. The dispensary attribution reworked to credit the **margin** via `wallet.earn_dropship_margin` (Plan 1) instead of flat `earn_dropship`. `/dispensary/<code>` serves a dedicated client page instead of redirecting to `/begin`. Reuses Plan 1 `practitioner_pricing` (`quote_line`, MAP), the patient consent/ToS gate, Stripe, shipping.

**Tech Stack:** Python 3.11, Flask, QBO, Stripe, pytest.

**DESIGN (locked):** patient-paid fee = 33% of (S − base) where **S = the practitioner's set price** (we collect S, so we know the markup; the rest is the practitioner's margin → wallet). S ≥ MAP $67 (advertised floor). Flat per-bottle, **no volume pricing** (Plan-2 design decision). Margin → wallet (credit-only). Default S = RM retail R when the practitioner hasn't set a price (price-setting UI = Plan 4).

**Reuse:**
- `practitioner_pricing.quote_line(selling_cents=S, qty, modules, settings)` → base/fee/margin; `resolve_selling_cents`/`MapViolation`; `load_settings` (map_default_cents 6700).
- `dropship_checkout` (Plan 2 patterns), `wallet.earn_dropship_margin(pid, margin_cents, *, qbo_invoice_id)` (Plan 1).
- `app.py`: `_pp.practitioner_id_by_dispensary_code(code)`, `_pp.portal_data(pid)` (modules_completed), `_record_dispensary_sale` (rework), `/dispensary/<code>` (rework), `is_member`/the consent gate, `_normalize_ship_address`, `_get_product(slug)["price_cents"]` (retail), `_ingest_order`, `_stripe_checkout_url_for_*`, `/begin/checkout-return`, `_shipping_for_cart`/`_shipping_line`, US ship-to validation.

---

### Task 1: `build_client_order` (patient-paid, practitioner-priced, returns margin)

**Files:** Modify `dashboard/dropship_checkout.py`; Test `tests/test_client_order.py`

- [ ] **Step 1: Failing test**

```python
# tests/test_client_order.py
from dashboard import dropship_checkout as dc

def test_practitioner_price_for_defaults_to_retail(monkeypatch):
    monkeypatch.setattr(dc, "_retail_for", lambda slug: 7000)
    # no stored price -> defaults to retail
    monkeypatch.setattr(dc, "_practitioner_price_cents", lambda pid, slug, retail: retail)
    assert dc.practitioner_price_for("p1", "brain-boost") == 7000

def test_build_client_order_charges_patient_credits_margin(monkeypatch):
    cart = [{"slug": "brain-boost", "qty": 1}]
    prac = {"id": "p1", "modules_completed": 0, "dispensary_code": "abc"}
    patient = {"email": "pat@x.com", "ship": {"name": "Pat", "state": "CA", "country": "US"}}
    monkeypatch.setattr(dc, "_retail_for", lambda slug: 7000)
    monkeypatch.setattr(dc, "practitioner_price_for", lambda pid, slug: 7000)   # S = retail $70
    monkeypatch.setattr(dc.qb, "find_or_create_customer", lambda *a, **k: {"Id": "PATC"})
    cap = {}
    monkeypatch.setattr(dc.qb, "create_invoice",
        lambda cust, lines, **k: cap.update(cust=cust, lines=lines) or
        {"Id": "INV", "TotalAmt": 70.0})
    import dashboard.tax as _tax
    monkeypatch.setattr(_tax, "compute_get_cents", lambda s, *, channel, ship_to_state, resale_ok=False: 0)
    out = dc.build_client_order(cart, prac, patient=patient, method="card")
    assert out["ok"] is True
    assert out["source"] == "dispensary"
    assert out["customer_id"] == "PATC"            # the PATIENT pays
    assert out["ship_to"]["name"] == "Pat"
    # 1 bottle @ S=$70, base $50, fee 33%*(7000-5000)=660 -> margin 1340
    assert out["margin_cents"] == 1340
    assert cap["lines"][0]["amount"] == 70.0       # patient is charged S
```

- [ ] **Step 2: Run → fail.**

- [ ] **Step 3: Implement** (in `dashboard/dropship_checkout.py`)
- `_practitioner_price_cents(pid, slug, retail)` — read the practitioner's stored FF price for this slug; default `retail` if unset. (Minimal read; the SETTING UI is Plan 4. Back it with a `practitioner_pricing` store or, for now, a stub returning retail with a TODO + a single hook the Plan-4 settings writes to.)
- `practitioner_price_for(pid, slug)` = `_practitioner_price_cents(pid, slug, _retail_for(slug))`.
- `build_client_order(cart, practitioner, *, patient, method)`: for each line, S = `practitioner_price_for(pid, slug)` (already ≥ MAP by construction; if a stored price is below MAP, clamp up to MAP). base/fee/**margin** via `quote_line(selling_cents=S, qty=total_bottles, modules)`. QBO customer = the **patient** (`patient["email"]`); invoice lines at S (amount = S/100, qty = line qty); ship to `patient["ship"]`; `source="dispensary"`; GET recorded-not-charged on the patient state; sum `margin_cents` across lines. Return ok/invoice_id/total/customer_id(patient)/ship_to/source/**margin_cents**/get_cents. (NO wallet redeem here — that's the practitioner's, not the patient's. The margin is credited on PAID, in Task 3.)

- [ ] **Step 4: Run → pass.**
- [ ] **Step 5: Commit** — `feat(client-order): build_client_order (patient-paid, practitioner-priced, returns margin)`

---

### Task 2: client checkout route (consent-gated, patient-paid)

**Files:** Modify `app.py`; Test `tests/test_client_routes.py`

- [ ] **Step 1: Failing test** — `POST /api/client/<code>/checkout` (body: email, name, address, method, items): resolves the practitioner by dispensary code; **consent gate** (the patient ToS/`is_member` gate — 403 `need_optin` if not met, like `begin_checkout`); calls `build_client_order`; on ok → `_ingest_order(source="dispensary", ...)`, alt-pay or Stripe; the Stripe metadata carries `kind="client"`, `practitioner_id`, `margin_cents`, `invoice_id` so the return handler can credit the margin. Test: unknown code → 404; no consent → 403; happy path → 200 with stripe_url + the order recorded.

- [ ] **Step 2: Run → fail.**

- [ ] **Step 3: Implement** — mirror `begin_checkout`'s consent gate + Stripe path. Resolve practitioner via `_pp.practitioner_id_by_dispensary_code(code)` (404 if none) + `portal_data` for modules. Build `patient = {email, ship}`. Consent: reuse `is_member`/the ToS gate keyed to the patient email + session (403 `need_optin`). `build_client_order(...)`; record order; for card, `_stripe_checkout_url_*` with metadata `{kind:"client", practitioner_id, margin_cents, invoice_id, customer_id}`. US ship-to only.

- [ ] **Step 4: Run → pass.**
- [ ] **Step 5: Commit** — `feat(client): /api/client/<code>/checkout (patient-paid, consent-gated)`

---

### Task 3: credit the margin on paid (rework dispensary attribution)

**Files:** Modify `app.py` (`/begin/checkout-return` + `_record_dispensary_sale`); Test `tests/test_client_margin_credit.py`

- [ ] **Step 1: Failing test** — when a client order is paid (checkout-return, `kind="client"`): `wallet.earn_dropship_margin(practitioner_id, margin_cents, qbo_invoice_id=invoice_id)` is called once (idempotent), crediting the margin to the practitioner's wallet. Replaces the flat `earn_dropship` for the new client flow. Test with stubbed wallet + a paid session; assert the margin (not $20×bottles) is credited, idempotent on replay.

- [ ] **Step 2: Run → fail.**

- [ ] **Step 3: Implement** — in `/begin/checkout-return`, when `md.get("kind")=="client"` and paid: `wallet.earn_dropship_margin(md["practitioner_id"], int(md["margin_cents"]), qbo_invoice_id=md["invoice_id"])` + record the dispensary order; wrapped so it never breaks the redirect. Update `_record_dispensary_sale` (used by the old `rm_dispensary`-cookie path through the RM funnel) — decide: either keep the old flat-$20 path for legacy RM-funnel dispensary buys, or route everything through the new client page. For Plan 3, the NEW client page credits the margin; the old `/dispensary/<code>`→`/begin` path is replaced (Task 4). Leave `earn_dropship` defined but unused once the landing is reworked.

- [ ] **Step 4: Run → pass.**
- [ ] **Step 5: Commit** — `feat(client): credit practitioner margin to wallet on paid client order`

---

### Task 4: dispensary landing → branded client page

**Files:** Modify `app.py` (`/dispensary/<code>`); Create `static/practitioner-client.html`

- [ ] **Step 1:** Rework `/dispensary/<code>` to serve the client page (set the `rm_dispensary` cookie as today, but render `static/practitioner-client.html` instead of redirecting to `/begin`). The page: FF list at the **practitioner's prices** (fetch from a `GET /api/client/<code>/catalog` returning each FF + the practitioner's price ≥ MAP), cart + qty (flat per-bottle, no volume tiers), patient details + shipping address + the **consent checkbox** (name + ToS), and a Pay action POSTing to `/api/client/<code>/checkout`. Minimal branding placeholder (practice name) — full white-label is Plan 4. US-only note.
- [ ] **Step 2:** Verify the page renders + posts correctly (static read + the route tests stay green).
- [ ] **Step 3:** Commit — `feat(client): branded client page on /dispensary/<code>`

---

### Task 5: suite + doc

- [ ] **Step 1:** Run all client + dropship + practitioner-pricing + wallet tests — green.
- [ ] **Step 2:** Create `docs/client-page.md`: patient buys at the practitioner's price (≥ MAP $67, flat, no volume); patient pays us; margin → practitioner wallet (`earn_dropship_margin`, replacing flat $20/bottle); consent-gated; ship to patient, US-only; default price = retail until Plan 4's price-setting UI.
- [ ] **Step 3:** Commit.

---

## Self-review
- **Spec coverage:** §B.1 client page (all tasks); §A.3 patient-paid economics + margin→wallet (T1, T3); §A.4 MAP enforcement (T1); §A.5 no volume (T1, flat S); §E dispensary rework (T3, T4). Reuses Plan 1 quote_line + earn_dropship_margin.
- **Deferred to Plan 4:** the practitioner price-SETTING UI ($/% per SKU) + white-label branding (this plan defaults S to retail + minimal branding); to Plan 5: the support chat.
- **Risk:** patient money path + consent gate + the dispensary rework. Mirror `begin_checkout`'s consent + Stripe; credit the margin idempotently on PAID only; never break the checkout-return redirect. No impact on the live customer funnel (separate routes/code path).
- **Type consistency:** `practitioner_price_for(pid, slug)`, `build_client_order(cart, prac, *, patient, method) -> dict` (with `margin_cents`), `/api/client/<code>/{catalog,checkout}`, checkout-return `kind="client"` → `earn_dropship_margin`.

## Next
Plan 4 — white-label settings + branding + the practitioner price-setting UI ($/%); the support chat is Plan 5.
