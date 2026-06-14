# Practitioner Pricing Core + Wallet Rework — Implementation Plan (Plan 1 of ~4)

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development. Steps use checkbox (`- [ ]`) syntax.

**Goal:** A pure, fully-tested pricing layer for the practitioner drop-ship system — blended base, flat 33% drop-ship fee, $-or-% selling-price resolution with MAP enforcement, practitioner margin, and the drop-ship "wholesale" the practitioner pays — plus reworking the wallet to earn the **margin** instead of a flat $20/bottle. No UI, no routes; this is the foundation the three portal pages (Plans 2–4) all call.

**Architecture:** New `dashboard/practitioner_pricing.py` (pure functions, wraps `dashboard/wholesale_pricing.py` for the blended base). Wallet change in `dashboard/wallet.py` (add a margin-based earn; keep the old flat function until Plan 2/3 cut over). Settings: `fee_pct` (0.33) and per-SKU `map_cents` (default $67) live in the pricing-settings config (pairs with the pending console editor).

**Tech Stack:** Python 3.11, sqlite/Supabase (wallet only), pytest. Pure module → plain `~/.venvs/deploy-chat311/bin/python -m pytest <path>`.

**Spec:** `docs/superpowers/specs/2026-06-14-practitioner-dropship-portal-design.md` (§A pricing, §D data, §E rework).

**Reuse (confirmed):** `wholesale_pricing.blended_unit_price_cents(q, modules, B)` ($50@q1 → F at 2B), `certification_floor_cents(modules)` (4000−modules×125), `DEFAULT_B=20`, `KNOT_Q1_CENTS=5000`. `wallet.earn_dropship(pid, bottles, *, qbo_invoice_id, ref)` (currently flat $20/bottle), `wallet.get_balance_cents(pid)`, the `wallet_ledger` insert pattern (idempotent per invoice).

---

### Task 1: settings + drop-ship base

**Files:** Create `dashboard/practitioner_pricing.py`; Test `tests/test_practitioner_pricing.py`

- [ ] **Step 1: Failing test**

```python
# tests/test_practitioner_pricing.py
from dashboard import practitioner_pricing as pp

def test_defaults():
    s = pp.load_settings({})
    assert s["fee_pct"] == 0.33
    assert s["map_default_cents"] == 6700

def test_drop_ship_base_matches_blended_curve():
    # q1 = $50 for everyone; 12 uncertified ~ $47.11; 12 fully-certified ~ $42.76
    assert pp.drop_ship_base_cents(1, 0) == 5000
    assert pp.drop_ship_base_cents(1, 12) == 5000
    assert pp.drop_ship_base_cents(12, 0) == 4711
    assert pp.drop_ship_base_cents(12, 12) == 4276
    assert pp.drop_ship_base_cents(40, 12) == 2500   # floor at 2B, fully certified
```

- [ ] **Step 2: Run → fail** (`ModuleNotFoundError`).

- [ ] **Step 3: Implement**

```python
# dashboard/practitioner_pricing.py
"""Pure pricing for practitioner drop-ship: blended base, flat 33% fee, $/% selling-price
with MAP, practitioner margin. Wraps dashboard.wholesale_pricing for the blended base."""
from dashboard import wholesale_pricing as _wp

DEFAULTS = {
    "fee_pct": 0.33,            # service fee on the practitioner's markup (drop-ship only)
    "map_default_cents": 6700,  # $67 minimum advertised price (per-SKU override in console)
}

def load_settings(overrides):
    s = dict(DEFAULTS)
    for k, v in (overrides or {}).items():
        if v is not None:
            s[k] = v
    return s

def drop_ship_base_cents(qty, modules_completed):
    """Per-bottle blended wholesale base for a drop-ship of `qty` bottles at the
    practitioner's certification level (same curve as wholesale stocking)."""
    return _wp.blended_unit_price_cents(int(qty), int(modules_completed), _wp.DEFAULT_B)
```

- [ ] **Step 4: Run → pass.**
- [ ] **Step 5: Commit** — `feat(practitioner-pricing): settings + blended drop-ship base`

---

### Task 2: flat 33% service fee

**Files:** Modify `dashboard/practitioner_pricing.py`; Test same file

- [ ] **Step 1: Failing test**

```python
def test_service_fee_flat_33pct_of_markup():
    s = pp.load_settings({})
    assert pp.service_fee_cents(8000, 5000, s) == 990    # 33% of (80-50)=30 -> $9.90
    assert pp.service_fee_cents(5000, 5000, s) == 0      # no markup -> no fee
    assert pp.service_fee_cents(4000, 5000, s) == 0      # negative markup clamps to 0
```

- [ ] **Step 2: Run → fail.**
- [ ] **Step 3: Implement**

```python
def service_fee_cents(selling_cents, base_cents, settings):
    """Flat fee = fee_pct of the markup (selling - base), never negative. Drop-ship only."""
    markup = max(0, int(selling_cents) - int(base_cents))
    return int(round(settings["fee_pct"] * markup))
```

- [ ] **Step 4: Run → pass.**
- [ ] **Step 5: Commit** — `feat(practitioner-pricing): flat 33% service fee`

---

### Task 3: selling-price resolution ($ or %) with MAP

**Files:** Modify `dashboard/practitioner_pricing.py`; Test same file

- [ ] **Step 1: Failing test**

```python
def test_resolve_selling_from_dollars_and_percent():
    # retail $70, MAP $67
    assert pp.resolve_selling_cents({"price_cents": 8000}, retail_cents=7000, map_cents=6700) == 8000
    # +20% over retail -> $84
    assert pp.resolve_selling_cents({"markup_pct": 20}, retail_cents=7000, map_cents=6700) == 8400
    # default to retail when nothing set
    assert pp.resolve_selling_cents({}, retail_cents=7000, map_cents=6700) == 7000

def test_resolve_selling_rejects_below_map():
    import pytest
    with pytest.raises(pp.MapViolation):
        pp.resolve_selling_cents({"price_cents": 6000}, retail_cents=7000, map_cents=6700)   # $60 < MAP
    with pytest.raises(pp.MapViolation):
        pp.resolve_selling_cents({"markup_pct": -10}, retail_cents=7000, map_cents=6700)      # $63 < MAP

def test_companion_figure_helper():
    # given a dollar price, report the implied markup %, and vice versa, for the UI
    assert pp.markup_pct_for(8400, 7000) == 20.0
    assert pp.price_for_markup(20, 7000) == 8400
```

- [ ] **Step 2: Run → fail.**
- [ ] **Step 3: Implement**

```python
class MapViolation(ValueError):
    """Selling price resolves below the Minimum Advertised Price."""

def price_for_markup(markup_pct, retail_cents):
    return int(round(int(retail_cents) * (1 + float(markup_pct) / 100.0)))

def markup_pct_for(price_cents, retail_cents):
    if not retail_cents:
        return 0.0
    return round((int(price_cents) - int(retail_cents)) / int(retail_cents) * 100.0, 1)

def resolve_selling_cents(price_input, *, retail_cents, map_cents):
    """price_input: {"price_cents": int} OR {"markup_pct": number} OR {} (default retail).
    Returns the selling price in cents; raises MapViolation if it is below MAP (advertised)."""
    if price_input.get("price_cents") is not None:
        s = int(price_input["price_cents"])
    elif price_input.get("markup_pct") is not None:
        s = price_for_markup(price_input["markup_pct"], retail_cents)
    else:
        s = int(retail_cents)
    if s < int(map_cents):
        raise MapViolation(f"{s} below MAP {map_cents}")
    return s
```

- [ ] **Step 4: Run → pass.**
- [ ] **Step 5: Commit** — `feat(practitioner-pricing): $/% selling price with MAP enforcement`

---

### Task 4: margin, drop-ship wholesale, and a per-line quote

**Files:** Modify `dashboard/practitioner_pricing.py`; Test same file

- [ ] **Step 1: Failing test**

```python
def test_margin_and_dropship_wholesale():
    s = pp.load_settings({})
    # S=$80, base=$50 -> fee $9.90 -> margin $20.10; practitioner-paid pays base+fee=$59.90
    q = pp.quote_line(selling_cents=8000, qty=1, modules=0, settings=s)
    assert q["base_cents"] == 5000
    assert q["fee_cents"] == 990
    assert q["margin_cents"] == 2010
    assert q["dropship_wholesale_cents"] == 5990       # what practitioner-paid mode pays us
    assert q["line_selling_cents"] == 8000

def test_quote_line_qty_uses_blended_volume():
    s = pp.load_settings({})
    # 12 bottles uncertified -> base $47.11/bottle; selling $80 each
    q = pp.quote_line(selling_cents=8000, qty=12, modules=0, settings=s)
    assert q["base_cents"] == 4711
    assert q["fee_cents"] == round(0.33 * (8000 - 4711))
    assert q["margin_cents"] == 8000 - 4711 - q["fee_cents"]

def test_margin_never_negative():
    s = pp.load_settings({})
    q = pp.quote_line(selling_cents=5000, qty=1, modules=0, settings=s)   # S == base, fee 0
    assert q["margin_cents"] == 0
```

- [ ] **Step 2: Run → fail.**
- [ ] **Step 3: Implement**

```python
def quote_line(*, selling_cents, qty, modules, settings):
    """Per-bottle economics for a drop-ship line. base = blended at this qty+cert;
    fee = 33% of markup; margin = selling - base - fee (>=0); dropship_wholesale = base+fee
    (what the practitioner pays in practitioner-paid mode)."""
    base = drop_ship_base_cents(qty, modules)
    fee = service_fee_cents(selling_cents, base, settings)
    margin = max(0, int(selling_cents) - base - fee)
    return {
        "line_selling_cents": int(selling_cents),
        "base_cents": base,
        "fee_cents": fee,
        "margin_cents": margin,
        "dropship_wholesale_cents": base + fee,
    }
```

- [ ] **Step 4: Run → pass.**
- [ ] **Step 5: Commit** — `feat(practitioner-pricing): per-line margin + drop-ship wholesale quote`

---

### Task 5: wallet earns the margin (rework the flat $20/bottle)

**Files:** Modify `dashboard/wallet.py`; Test `tests/test_wallet_margin.py`

First READ `dashboard/wallet.py` — mirror the existing `earn_dropship` idempotency (per `qbo_invoice_id`) and ledger-insert pattern exactly.

- [ ] **Step 1: Failing test**

```python
# tests/test_wallet_margin.py
# Use the wallet module's test seam (see how existing wallet tests stub the DB);
# if wallet hits Supabase, follow the same monkeypatch/stub the repo's wallet tests use.
from dashboard import wallet

def test_earn_dropship_margin_credits_margin_not_flat(monkeypatch):
    # earn_dropship_margin(pid, margin_cents, qbo_invoice_id) credits exactly margin_cents,
    # idempotent per invoice (a second call with the same invoice is a no-op).
    ...  # implement against the repo's wallet test harness; assert balance == margin once,
        # and unchanged on replay.
```

- [ ] **Step 2: Run → fail.**
- [ ] **Step 3: Implement** — add `earn_dropship_margin(pid, margin_cents, *, qbo_invoice_id, ref=None) -> int` that credits `margin_cents` to the wallet with entry_type `"earn_dropship_margin"`, idempotent per `qbo_invoice_id` (reuse the existing replay guard). Keep the old `earn_dropship` (flat $20) in place; Plan 3 (client-page rework) switches the dispensary hook over to the margin version, then the flat one can be removed. If the repo's wallet tests have no local-DB seam, add a minimal one mirroring the existing test setup and note it.

- [ ] **Step 4: Run → pass.**
- [ ] **Step 5: Commit** — `feat(wallet): earn_dropship_margin (margin-based credit, idempotent)`

---

### Task 6: full suite + doc

**Files:** Create `docs/practitioner-pricing.md`

- [ ] **Step 1:** Run `tests/test_practitioner_pricing.py tests/test_wallet_margin.py` — all green.
- [ ] **Step 2:** Write `docs/practitioner-pricing.md`: the blended base (= wholesale curve), flat 33% drop-ship fee, $/% selling input + MAP ($67 default, per-SKU console), margin = selling−base−fee → wallet credit (replaces flat $20/bottle), drop-ship wholesale = base+fee (practitioner-paid). Note the three pages (Plans 2-4) all price through `quote_line`.
- [ ] **Step 3:** Commit.

---

## Self-review
- **Spec coverage:** §A.1 base (T1), §A.2 fee (T2), §A.4 MAP + §A.6 $/% (T3), §A.3 margin + drop-ship wholesale (T4), §E wallet rework (T5). No routes/UI — those are Plans 2-4.
- **Deferred:** the three pages (client/drop-ship/wholesale), white-label settings/branding, the dispensary-hook cutover to `earn_dropship_margin`, per-SKU MAP/pricing console editor, no-quantity-pricing enforcement on the client page (it just uses a flat selling price — nothing to compute).
- **Placeholders:** Task 5's test is intentionally harness-dependent (wallet may hit Supabase) — the implementer mirrors the repo's existing wallet test seam; everything else has complete code.
- **Type consistency:** `drop_ship_base_cents(qty, modules)`, `service_fee_cents(selling, base, settings)`, `resolve_selling_cents(input, *, retail_cents, map_cents)`, `quote_line(*, selling_cents, qty, modules, settings) -> dict`, `MapViolation`, `earn_dropship_margin(pid, margin_cents, *, qbo_invoice_id, ref)` used consistently.

## Next
Plan 2 — practitioner-paid drop-ship page (patient address + selling price → pay base+fee, ship to patient). Plan 3 — client-page rework on `/dispensary/<code>` (branded, practitioner-priced ≥ MAP, patient-paid → `earn_dropship_margin`). Plan 4 — white-label settings + branding. Wholesale stocking page is mostly existing.
