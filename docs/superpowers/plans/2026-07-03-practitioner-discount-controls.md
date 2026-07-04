# Practitioner Discount Controls — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Let each practitioner opt in/out and set the three product-discount types for their own portal's patients, each clamped to our live global curve, replacing the global config for their patients.

**Architecture:** A new `dashboard/practitioner_pricing.py` module holds a per-practitioner config (two schedules — Standard + Program — each with a per-type enable + pass-through dial) and a **pure** `effective_settings()` builder that turns that config into a `pricing.compute`-shaped `settings["discounts"]` block, clamped to the live global ceilings. The practitioner portal gets a Pricing panel + `POST /api/practitioner/pricing`. The dispensary patient-order path (`build_client_order`) is routed through the practitioner-effective engine so a practitioner's patient actually receives the discounts, off the practitioner's own S price.

**Tech Stack:** Python 3, Flask, SQLite, pytest, vanilla JS (static HTML portal).

## Global Constraints

- Dial is a fraction `0.0 ≤ dial ≤ 1.0` = share of the ceiling passed through. Effective pct = `dial × ceiling_pct`, which guarantees effective ≤ ceiling at every quantity. No other clamp is needed.
- The three types: `same_sku`, `program_total`, `open_total`. Ceiling for `same_sku` = live global `same_sku` ramp; for `program_total` = live global `program_total` ramp; **for `open_total` = live global `program_total` ramp** (private-channel decision — never global `open_total`, which is 0/off for the public store).
- Ceilings are **dynamic**: always read from the live global settings (`pricing._discount_cfg(pricing.load_settings(settings))`), never snapshotted.
- Resolution is **replace, not stack**: a practitioner's patient prices against the practitioner-effective settings only.
- `program_total` stays gated on `program_member` inside `pricing.compute` — always pass `program_member` through. V1 "in a paid program" = the existing `_is_paid_member(email)` signal.
- Money is integer cents everywhere. Reuse `pricing.apply_discount` / `pricing.unit_floor_cents` — do not hand-roll discount math.
- Follow existing module patterns: `cx`-first SQLite helpers like `dashboard/client_prices.py`; `_practitioner_session_pid()` guards practitioner routes.

---

## File Structure

- **Create** `dashboard/practitioner_pricing.py` — table CRUD (`init_table`/`get_config`/`set_config`), `validate_config`, pure `ceilings`, pure `effective_settings`.
- **Create** `tests/test_practitioner_pricing.py` — unit tests for the pure builder, validation, CRUD.
- **Modify** `dashboard/practitioner_portal.py:portal_data()` — include `pricing_config` + `pricing_ceilings`.
- **Modify** `app.py` — add `POST /api/practitioner/pricing`; add `_practitioner_effective_settings(pid, program_member)` helper.
- **Create** `tests/test_practitioner_pricing_routes.py` — route tests (GET portal-data carries config+ceilings; POST validates + saves).
- **Modify** `static/practitioner-portal.html` — Pricing panel (two schedules, per-type toggle + dial, live ceilings).
- **Modify** `dashboard/dropship_checkout.py:build_client_order` — route patient pricing through the practitioner-effective engine; recompute margin off the discounted price.
- **Create** `tests/test_dispensary_practitioner_discount.py` — end-to-end: a practitioner's patient receives the dialed discount, clamped, schedule-selected.

---

### Task 1: `practitioner_pricing` data layer

**Files:**
- Create: `dashboard/practitioner_pricing.py`
- Test: `tests/test_practitioner_pricing.py`

**Interfaces:**
- Produces: `init_table(cx)`, `get_config(cx, pid) -> dict`, `set_config(cx, pid, config: dict) -> None`.

- [ ] **Step 1: Write the failing test**

```python
# tests/test_practitioner_pricing.py
import sqlite3
from dashboard import practitioner_pricing as pp


def _cx():
    return sqlite3.connect(":memory:")


def test_get_config_defaults_empty():
    with _cx() as cx:
        assert pp.get_config(cx, "7") == {}


def test_set_then_get_roundtrips():
    cfg = {"standard": {"same_sku": {"enabled": True, "dial": 0.5}}}
    with _cx() as cx:
        pp.set_config(cx, "7", cfg)
        assert pp.get_config(cx, "7") == cfg


def test_set_config_upserts():
    with _cx() as cx:
        pp.set_config(cx, "7", {"standard": {"same_sku": {"enabled": True, "dial": 0.2}}})
        pp.set_config(cx, "7", {"standard": {"same_sku": {"enabled": False, "dial": 0.9}}})
        assert pp.get_config(cx, "7")["standard"]["same_sku"]["dial"] == 0.9
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/test_practitioner_pricing.py -q`
Expected: FAIL (`ModuleNotFoundError: dashboard.practitioner_pricing`)

- [ ] **Step 3: Write minimal implementation**

```python
# dashboard/practitioner_pricing.py
"""Per-practitioner product-discount controls. Pure builder + SQLite config.

Config shape (both schedules optional; program has an extra 'enabled' master flag):
{
  "standard": {"same_sku": {"enabled": bool, "dial": 0..1}, "program_total": {...}, "open_total": {...}},
  "program":  {"enabled": bool, "same_sku": {...}, "program_total": {...}, "open_total": {...}}
}
"""
import json
import sqlite3
from datetime import datetime, timezone

from dashboard import pricing as _pricing

_TYPES = ("same_sku", "program_total", "open_total")


def _now():
    return datetime.now(timezone.utc).isoformat()


def init_table(cx):
    cx.execute(
        "CREATE TABLE IF NOT EXISTS practitioner_pricing ("
        "practitioner_id TEXT PRIMARY KEY, config_json TEXT NOT NULL, updated_at TEXT NOT NULL)"
    )


def get_config(cx, pid):
    init_table(cx)
    row = cx.execute(
        "SELECT config_json FROM practitioner_pricing WHERE practitioner_id=?", (str(pid),)
    ).fetchone()
    return json.loads(row[0]) if row else {}


def set_config(cx, pid, config):
    init_table(cx)
    cx.execute(
        "INSERT INTO practitioner_pricing (practitioner_id, config_json, updated_at) "
        "VALUES (?,?,?) ON CONFLICT(practitioner_id) DO UPDATE SET "
        "config_json=excluded.config_json, updated_at=excluded.updated_at",
        (str(pid), json.dumps(config), _now()),
    )
```

- [ ] **Step 4: Run test to verify it passes**

Run: `python -m pytest tests/test_practitioner_pricing.py -q`
Expected: PASS (3 passed)

- [ ] **Step 5: Commit**

```bash
git add dashboard/practitioner_pricing.py tests/test_practitioner_pricing.py
git commit -m "feat(practitioner-pricing): config table CRUD"
```

---

### Task 2: `ceilings()` + `effective_settings()` pure builder

**Files:**
- Modify: `dashboard/practitioner_pricing.py`
- Test: `tests/test_practitioner_pricing.py`

**Interfaces:**
- Consumes: `_TYPES`, `pricing.load_settings`, `pricing._discount_cfg`, `pricing.same_sku_pct`, `pricing.program_total_pct`, `pricing.open_total_pct` (from `dashboard/pricing.py`).
- Produces:
  - `ceilings(settings: dict) -> {"same_sku": float, "program_total": float, "open_total": float}` (max pct per type).
  - `effective_settings(config: dict, *, program_member: bool, settings: dict) -> dict` (a full settings dict whose `["discounts"]` is the clamped, schedule-selected block).

- [ ] **Step 1: Write the failing test**

```python
# add to tests/test_practitioner_pricing.py
from dashboard import pricing as _pricing


def _global():
    # live global settings from code defaults: same_sku ON, program_total ON, open_total OFF
    return _pricing.load_settings({})


def test_ceilings_open_total_uses_program_total_curve():
    c = pp.ceilings(_global())
    # program_total default maxes at 29 (qty 18); open_total ceiling mirrors it
    assert c["program_total"] == 29.0
    assert c["open_total"] == 29.0
    assert c["same_sku"] == 29.0


def test_effective_disabled_type_yields_zero():
    eff = pp.effective_settings({}, program_member=False, settings=_global())
    assert _pricing.same_sku_pct(12, eff) == 0.0
    assert _pricing.open_total_pct(18, eff) == 0.0


def test_effective_dial_scales_and_clamps_to_ceiling():
    cfg = {"standard": {"same_sku": {"enabled": True, "dial": 0.5}}}
    eff = pp.effective_settings(cfg, program_member=False, settings=_global())
    # half of the 29% ceiling at max qty
    assert abs(_pricing.same_sku_pct(12, eff) - 14.5) < 1e-6
    # never exceeds ceiling even if dial were >1 (clamped)
    cfg["standard"]["same_sku"]["dial"] = 5.0
    eff2 = pp.effective_settings(cfg, program_member=False, settings=_global())
    assert _pricing.same_sku_pct(12, eff2) == 29.0


def test_effective_open_total_dialed_off_program_curve():
    cfg = {"standard": {"open_total": {"enabled": True, "dial": 1.0}}}
    eff = pp.effective_settings(cfg, program_member=False, settings=_global())
    # open_total fires for everyone (not member-gated), using the program_total ceiling
    assert _pricing.open_total_pct(18, eff) == 29.0


def test_program_schedule_only_for_members_and_when_enabled():
    cfg = {
        "standard": {"same_sku": {"enabled": True, "dial": 0.25}},
        "program": {"enabled": True, "same_sku": {"enabled": True, "dial": 1.0}},
    }
    non = pp.effective_settings(cfg, program_member=False, settings=_global())
    mem = pp.effective_settings(cfg, program_member=True, settings=_global())
    assert abs(_pricing.same_sku_pct(12, non) - 7.25) < 1e-6   # standard: 0.25*29
    assert _pricing.same_sku_pct(12, mem) == 29.0               # program: 1.0*29
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/test_practitioner_pricing.py -q`
Expected: FAIL (`AttributeError: module ... has no attribute 'ceilings'`)

- [ ] **Step 3: Write minimal implementation**

```python
# append to dashboard/practitioner_pricing.py

def _ceiling_anchors(gcfg, ptype):
    # open_total's ceiling is the program_total curve (private-channel decision).
    key = "program_total" if ptype == "open_total" else ptype
    return [list(a) for a in gcfg[key]["anchors"]]


def ceilings(settings):
    gcfg = _pricing._discount_cfg(_pricing.load_settings(settings))
    return {t: float(_ceiling_anchors(gcfg, t)[-1][1]) for t in _TYPES}


def _scaled(anchors, dial):
    d = max(0.0, min(1.0, float(dial)))
    return [[a[0], round(a[1] * d, 4)] for a in anchors]


def effective_settings(config, *, program_member, settings):
    base = _pricing.load_settings(settings)
    gcfg = _pricing._discount_cfg(base)
    cfg = config or {}
    use_program = bool(program_member) and bool((cfg.get("program") or {}).get("enabled"))
    sched = cfg.get("program" if use_program else "standard") or {}
    disc = {}
    for t in _TYPES:
        ent = sched.get(t) or {}
        ceil = _ceiling_anchors(gcfg, t)
        if bool(ent.get("enabled")):
            disc[t] = {"enabled": True, "anchors": _scaled(ceil, ent.get("dial", 0.0))}
        else:
            disc[t] = {"enabled": False, "anchors": ceil}
    out = dict(base)
    out["discounts"] = disc
    return out
```

- [ ] **Step 4: Run test to verify it passes**

Run: `python -m pytest tests/test_practitioner_pricing.py -q`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add dashboard/practitioner_pricing.py tests/test_practitioner_pricing.py
git commit -m "feat(practitioner-pricing): dynamic ceilings + effective_settings builder"
```

---

### Task 3: `validate_config`

**Files:**
- Modify: `dashboard/practitioner_pricing.py`
- Test: `tests/test_practitioner_pricing.py`

**Interfaces:**
- Produces: `validate_config(payload) -> list[str]` (empty list = valid).

- [ ] **Step 1: Write the failing test**

```python
# add to tests/test_practitioner_pricing.py
def test_validate_accepts_good_config():
    ok = {"standard": {"open_total": {"enabled": True, "dial": 0.5}},
          "program": {"enabled": True, "program_total": {"enabled": True, "dial": 1.0}}}
    assert pp.validate_config(ok) == []


def test_validate_rejects_unknown_type():
    errs = pp.validate_config({"standard": {"mystery": {"enabled": True, "dial": 0.5}}})
    assert any("unknown discount type" in e for e in errs)


def test_validate_rejects_dial_out_of_range():
    assert any("between 0 and 1" in e for e in
               pp.validate_config({"standard": {"same_sku": {"enabled": True, "dial": 1.5}}}))
    assert any("between 0 and 1" in e for e in
               pp.validate_config({"standard": {"same_sku": {"enabled": True, "dial": -0.1}}}))


def test_validate_rejects_non_bool_enabled():
    assert any("boolean" in e for e in
               pp.validate_config({"standard": {"same_sku": {"enabled": "yes", "dial": 0.5}}}))
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/test_practitioner_pricing.py -q`
Expected: FAIL (`AttributeError: ... 'validate_config'`)

- [ ] **Step 3: Write minimal implementation**

```python
# append to dashboard/practitioner_pricing.py

def validate_config(payload):
    errors = []
    if not isinstance(payload, dict):
        return ["config must be an object"]
    for sched_name in ("standard", "program"):
        sched = payload.get(sched_name)
        if sched is None:
            continue
        if not isinstance(sched, dict):
            errors.append(f"{sched_name} must be an object")
            continue
        for k, v in sched.items():
            if sched_name == "program" and k == "enabled":
                if not isinstance(v, bool):
                    errors.append("program.enabled must be boolean")
                continue
            if k not in _TYPES:
                errors.append(f"unknown discount type: {sched_name}.{k}")
                continue
            if not isinstance(v, dict):
                errors.append(f"{sched_name}.{k} must be an object")
                continue
            if "enabled" in v and not isinstance(v["enabled"], bool):
                errors.append(f"{sched_name}.{k}.enabled must be boolean")
            if "dial" in v:
                dv = v["dial"]
                if isinstance(dv, bool) or not isinstance(dv, (int, float)):
                    errors.append(f"{sched_name}.{k}.dial must be a number")
                elif dv < 0 or dv > 1:
                    errors.append(f"{sched_name}.{k}.dial must be between 0 and 1")
    return errors
```

- [ ] **Step 4: Run test to verify it passes**

Run: `python -m pytest tests/test_practitioner_pricing.py -q`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add dashboard/practitioner_pricing.py tests/test_practitioner_pricing.py
git commit -m "feat(practitioner-pricing): validate_config"
```

---

### Task 4: Portal read (`portal_data`) + write route (`POST /api/practitioner/pricing`)

**Files:**
- Modify: `dashboard/practitioner_portal.py` (`portal_data`, near line 804)
- Modify: `app.py` (new route after `api_practitioner_portal_data`, ~line 11257; new helper near `_pricing_settings`, ~line 121)
- Test: `tests/test_practitioner_pricing_routes.py`

**Interfaces:**
- Consumes: `practitioner_pricing.get_config/set_config/validate_config/ceilings`, `_practitioner_session_pid()`, `_pricing_settings()`, `_pp.portal_data`.
- Produces: `portal_data(...)` dict gains `"pricing_config"` (dict) + `"pricing_ceilings"` (dict). New route `POST /api/practitioner/pricing`. Helper `_practitioner_effective_settings(pid, program_member) -> dict`.

- [ ] **Step 1: Write the failing test**

```python
# tests/test_practitioner_pricing_routes.py
import importlib, sqlite3, sys
from pathlib import Path


def _app():
    repo = Path(__file__).resolve().parent.parent
    if str(repo) not in sys.path:
        sys.path.insert(0, str(repo))
    return importlib.import_module("app")


def test_portal_data_carries_config_and_ceilings(monkeypatch):
    app = _app()
    from dashboard import practitioner_portal as pp
    data = pp.portal_data("1")  # any existing seed pid in the test DB helper
    assert "pricing_config" in data
    assert "pricing_ceilings" in data
    assert data["pricing_ceilings"]["open_total"] == 29.0


def test_post_pricing_rejects_bad_dial(monkeypatch):
    app = _app()
    monkeypatch.setattr(app, "_practitioner_session_pid", lambda: "1")
    client = app.app.test_client()
    r = client.post("/api/practitioner/pricing",
                    json={"config": {"standard": {"same_sku": {"enabled": True, "dial": 2}}}})
    assert r.status_code == 400
    assert not r.get_json()["ok"]


def test_post_pricing_saves_valid(monkeypatch):
    app = _app()
    monkeypatch.setattr(app, "_practitioner_session_pid", lambda: "1")
    client = app.app.test_client()
    cfg = {"standard": {"open_total": {"enabled": True, "dial": 0.5}}}
    r = client.post("/api/practitioner/pricing", json={"config": cfg})
    assert r.status_code == 200 and r.get_json()["ok"]
    from dashboard import practitioner_portal as pp
    assert pp.portal_data("1")["pricing_config"] == cfg
```

Note: if `portal_data("1")` needs a seeded practitioner, follow the existing seeding pattern in `tests/test_practitioner_portal.py` (reuse its fixture/helper for a valid pid + DB path) rather than inventing one.

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/test_practitioner_pricing_routes.py -q`
Expected: FAIL (`KeyError: 'pricing_config'` / 404 on the route)

- [ ] **Step 3a: Add config to `portal_data`**

In `dashboard/practitioner_portal.py`, inside `portal_data(...)` before its `return`, add (using the module's existing `db_path`/connection pattern):

```python
    from dashboard import practitioner_pricing as _ppx
    try:
        with sqlite3.connect(db_path or _db_path()) as _pcx:
            out["pricing_config"] = _ppx.get_config(_pcx, practitioner_id)
    except Exception:
        out["pricing_config"] = {}
    out["pricing_ceilings"] = _ppx.ceilings(_pricing_overrides())
```

where `_pricing_overrides()` is the module-local accessor for saved global overrides. If `practitioner_portal.py` has no access to the app's `_pricing_settings()`, pass `{}` (code defaults) — ceilings from defaults are correct until a console override exists; the route helper in Step 3c uses the live app settings for actual pricing. Use `{}` for simplicity here:

```python
    out["pricing_ceilings"] = _ppx.ceilings({})
```

(Confirm `out` is the dict being returned and `practitioner_id`/`db_path` are the in-scope names; adapt to the actual local variable names in `portal_data`.)

- [ ] **Step 3b: Add the write route in `app.py`** (after `api_practitioner_portal_data`)

```python
@app.route("/api/practitioner/pricing", methods=["POST"])
def api_practitioner_pricing():
    pid = _practitioner_session_pid()
    if not pid:
        return jsonify({"ok": False, "error": "not signed in"}), 401
    from dashboard import practitioner_pricing as _ppx
    body = request.get_json(silent=True) or {}
    config = body.get("config") or {}
    errs = _ppx.validate_config(config)
    if errs:
        return jsonify({"ok": False, "error": "; ".join(errs)}), 400
    with sqlite3.connect(LOG_DB) as cx:
        _ppx.set_config(cx, pid, config)
    return jsonify({"ok": True, **(_pp.portal_data(pid) or {})})
```

- [ ] **Step 3c: Add the effective-settings helper in `app.py`** (near `_pricing_settings`)

```python
def _practitioner_effective_settings(pid, program_member):
    """Live global settings with the practitioner's clamped discount block swapped in.
    Falls back to plain global settings when the practitioner has no saved config."""
    from dashboard import practitioner_pricing as _ppx
    with sqlite3.connect(LOG_DB) as cx:
        cfg = _ppx.get_config(cx, pid)
    return _ppx.effective_settings(cfg, program_member=bool(program_member),
                                   settings=_pricing_settings())
```

- [ ] **Step 4: Run test to verify it passes**

Run: `python -m pytest tests/test_practitioner_pricing_routes.py -q`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add dashboard/practitioner_portal.py app.py tests/test_practitioner_pricing_routes.py
git commit -m "feat(practitioner-pricing): portal read + POST /api/practitioner/pricing + effective-settings helper"
```

---

### Task 5: Portal Pricing panel (UI)

**Files:**
- Modify: `static/practitioner-portal.html`

**Interfaces:**
- Consumes: `GET /api/practitioner/portal-data` fields `pricing_config`, `pricing_ceilings`; `POST /api/practitioner/pricing` with `{config}`.
- Produces: a "Pricing" panel under an existing tab (Partner Program or Clients). No new backend.

This task is UI-only; it has no pytest cycle. Deliverable = a working panel verified in the browser.

- [ ] **Step 1: Add the panel markup**

Add a Pricing section (match the file's existing card/tab styling). For each of the two schedules (Standard always shown; Program behind a master toggle), render three rows — same-SKU, program-total, open-total — each with: an enable checkbox, a 0–100% pass-through slider (`dial × 100`), and a read-only "= up to N% off" derived from `pricing_ceilings[type] × dial`. Label the open-total row: "Mix-and-match across remedies (your patients only)."

- [ ] **Step 2: Wire load**

On portal-data load, populate the controls from `data.pricing_config` (default everything off / dial 0 when absent) and show each ceiling from `data.pricing_ceilings`.

- [ ] **Step 3: Wire save**

A "Save pricing" button POSTs `{config}` (rebuild the config object from the controls; dial = slider/100) to `/api/practitioner/pricing`; on `ok` show a saved confirmation, on error show `error`.

- [ ] **Step 4: Verify in browser**

Run the app locally (per `reference_deploy_chat_local_tests` / the repo's run instructions), open `/practitioner/portal?token=...` for a seeded practitioner, set a dial, save, reload → values persist and the derived "up to N% off" tracks the slider.

- [ ] **Step 5: Commit**

```bash
git add static/practitioner-portal.html
git commit -m "feat(practitioner-pricing): portal Pricing panel (two schedules, dials, live ceilings)"
```

---

### Task 6: Route dispensary patient pricing through the practitioner-effective engine

**Files:**
- Modify: `dashboard/dropship_checkout.py` (`build_client_order`, lines ~170–239)
- Modify: `app.py` (dispensary client-checkout call site, ~line 12533, to pass `program_member` + config through)
- Test: `tests/test_dispensary_practitioner_discount.py`

**Interfaces:**
- Consumes: `practitioner_pricing.effective_settings`, `pricing.same_sku_pct/program_total_pct/open_total_pct/apply_discount/unit_floor_cents`, `app._get_product`, `app._qty_eligible`, `app._is_paid_member`.
- Produces: `build_client_order(..., *, effective_settings=None, program_member=False)` — when `effective_settings` is provided, each line's patient price = S reduced by the practitioner's best-of volume discount (clamped to ceilings), floored via `unit_floor_cents`; margin recomputed off the discounted price. When `effective_settings` is None, behavior is unchanged (flat S).

- [ ] **Step 1: Write the failing test**

```python
# tests/test_dispensary_practitioner_discount.py
from dashboard import practitioner_pricing as ppx, pricing as _pricing
from dashboard import dropship_checkout as dc


def test_no_config_leaves_price_at_S(monkeypatch):
    # With no effective_settings, patient pays flat S (baseline behavior preserved).
    monkeypatch.setattr(dc, "practitioner_price_for", lambda pid, slug: 6997)
    # ... build a 1-line cart, call build_client_order(effective_settings=None)
    # assert the line amount == 69.97 (stub qb create_invoice/find_or_create_customer per existing tests)


def test_dialed_open_total_discounts_patient_off_S(monkeypatch):
    # same_sku off, open_total dialed to full: 12 volume-eligible bottles off a $69.97 S
    # should apply the program_total-curve discount at qty 12 (~18.8%) off S.
    ...
```

Follow `tests/test_practitioner_personal_order.py` for how `qb.find_or_create_customer` / `qb.create_invoice` / `_retail_for` are stubbed so `build_client_order` runs without network. Assert the returned line `amount` and `margin_cents` reflect the discount.

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/test_dispensary_practitioner_discount.py -q`
Expected: FAIL (`build_client_order() got an unexpected keyword argument 'effective_settings'`)

- [ ] **Step 3: Implement the discounted pricing branch**

In `build_client_order`, change the signature to accept `effective_settings=None, program_member=False`, and replace the per-line pricing loop so each line applies the practitioner's best-of discount off S:

```python
def build_client_order(cart, practitioner, *, patient, method=None,
                       points_to_redeem_cents=0, points_balance_cents=0,
                       effective_settings=None, program_member=False):
    ...
    import app as _app
    eff = effective_settings
    # order-wide volume-eligible bottle count (mirrors the direct engine)
    total_ff = sum(int(i.get("qty", 0)) for i in cart
                   if _app._qty_eligible(_app._get_product(i["slug"])))
    open_pct = _pricing.open_total_pct(total_ff, eff) if eff else 0.0
    prog_pct = _pricing.program_total_pct(total_ff, eff, program_member) if eff else 0.0

    for item in cart:
        slug = item["slug"]
        line_qty = int(item.get("qty", 1))
        s_cents = practitioner_price_for(pid, slug)
        prod = _app._get_product(slug)
        elig = bool(_app._qty_eligible(prod))
        if eff and elig:
            t1 = _pricing.same_sku_pct(line_qty, eff)
            line_pct = max(t1, prog_pct, open_pct)
        else:
            line_pct = 0.0
        floor = _pricing.unit_floor_cents(prod, s_cents, eff or settings, "discount")
        paid_unit = _pricing.apply_discount(s_cents, line_pct, floor)
        q = _pp.quote_line(selling_cents=s_cents, qty=total_bottles,
                           modules=modules, settings=settings)
        line_margin = q["margin_cents"] - (s_cents - paid_unit)  # discount comes out of margin
        subtotal_cents += paid_unit * line_qty
        total_margin_cents += line_margin * line_qty
        total_fee_cents += q["fee_cents"] * line_qty
        lines.append({"name": slug, "amount": paid_unit / 100.0,
                      "qty": line_qty, "description": f"{slug} (dispensary)"})
```

(Leave the rest of `build_client_order` — QBO invoice, GET, points, return dict — unchanged; it already reads `subtotal_cents`, `total_margin_cents`, `total_fee_cents`, `lines`.)

- [ ] **Step 4: Pass practitioner context from the dispensary route**

In `app.py` at the dispensary client-checkout `build_client_order(...)` call (~line 12533), add:

```python
    _program_member = _is_paid_member(email)
    out = _dropship.build_client_order(
        items, prac, patient=patient, method=method,
        points_to_redeem_cents=redeem_req, points_balance_cents=bal_cents,
        effective_settings=_practitioner_effective_settings(pid, _program_member),
        program_member=_program_member,
    )
```

- [ ] **Step 5: Run tests to verify they pass**

Run: `python -m pytest tests/test_dispensary_practitioner_discount.py tests/test_practitioner_personal_order.py -q`
Expected: PASS (new discount tests + existing dispensary/personal-order tests still green — proves the `effective_settings=None` path is unchanged)

- [ ] **Step 6: Commit**

```bash
git add dashboard/dropship_checkout.py app.py tests/test_dispensary_practitioner_discount.py
git commit -m "feat(practitioner-pricing): apply practitioner-effective discounts to dispensary patient orders"
```

---

## Self-Review

**Spec coverage:**
- 3 types + ceilings (same_sku/program_total; open_total off program_total curve) → Tasks 2, and enforced in 6. ✓
- Dynamic ceilings → Task 2 (`ceilings`/`effective_settings` read live global). ✓
- Two schedules (Standard/Program), member-gated program schedule → Task 2 + tested. ✓
- Replace-not-stack → Task 6 uses effective settings only; `effective_settings=None` preserves baseline (tested). ✓
- Data layer + validation → Tasks 1, 3. ✓
- Portal read/write API + UI → Tasks 4, 5. ✓
- Deploy-safety: no practitioner path enables global open_total on the public store — practitioner open_total lives only in the per-call effective settings; global settings are never mutated (Task 4 helper builds a copy; Task 2 `out = dict(base)`). ✓

**Placeholder scan:** Task 5 (UI) and Task 6 (Step 1 test bodies) intentionally reference existing seeding/stub patterns rather than duplicating fixtures — the implementer must copy the real fixture from the named sibling test. All code steps for the pure/data/route layers are complete. ✓

**Type consistency:** `effective_settings(config, *, program_member, settings)` and `ceilings(settings)` signatures match across Tasks 2, 4, 6. Config shape (`standard`/`program` → `{type: {enabled, dial}}`, `program.enabled`) is consistent across Tasks 1–5. `build_client_order(..., effective_settings=None, program_member=False)` consistent across Task 6 Steps 3–4. ✓

## Notes / open confirmations for review

- **Integration channel (v1):** the live wiring targets the **dispensary patient-order** path (`build_client_order`), because that is the patient-pays-us channel with an unambiguous `practitioner_id` in hand. The main direct-store `_price_cart` path is NOT practitioner-attributed and is intentionally left on global settings. If patient-portal (`/portal/<token>`) sales should also carry practitioner discounts, that's a follow-up attribution task.
- **Discount comes off the practitioner's S price** (not off list), floored via `unit_floor_cents`. The discount reduces the practitioner's margin — the intended "trade margin for depth." Confirm this is the desired economic direction before executing Task 6.
