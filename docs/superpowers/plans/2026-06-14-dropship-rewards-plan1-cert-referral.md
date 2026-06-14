# Drop-ship Rewards Plan 1 — Cert-tiered Referral Reward

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development. Steps use checkbox (`- [ ]`) syntax.

**Goal:** Scale the practitioner referral reward by certification: a referrer who is a practitioner earns a % interpolated from `referral_cert_anchors` (modules_completed → %), defaulting to `[[0,5],[6,10],[12,15]]`; everyone else stays at the base 5%. Console-editable, additive, gated by the existing `REWARDS_TIERS_ENABLED`.

**Architecture:** `dashboard/rewards.py` gains the anchor default + a pure `referral_pct_for_modules(modules, settings)` interpolator (returns a fraction). `dashboard/practitioner_portal.py` gains a lightweight `modules_completed_for_email(email)` Postgres lookup. `app.py` `_settle_referral` resolves the referrer's cert via a new `_referral_pct_for_referrer` (practitioner → interp; non-practitioner → base) instead of the flat read. `dashboard/pricing_settings.py` validates `referral_cert_anchors` (anchor-table validation extracted + reused). The editor page gets a referral-cert anchor table.

**Tech Stack:** Python 3.11, Flask, sqlite (referral ledger) + Supabase Postgres (practitioners), pytest.

**Spec:** `docs/superpowers/specs/2026-06-14-dropship-rewards-design.md` (Piece 1). Curve approved by Glen: `[[0,5],[6,10],[12,15]]`.

**Test invocation:** pure modules → `~/.venvs/deploy-chat311/bin/python -m pytest <path> -q`. App/Postgres-touching tests → `doppler run -p remedy-match -c prd -- env DATA_DIR="$HOME/deploy-chat" ~/.venvs/deploy-chat311/bin/python -m pytest <path> -q` (from inside the worktree; ignore the 2 known pre-existing failures `test_pf_playwright_fetch`, `test_bos_routes::test_home_page_served`).

---

### Task 1: `rewards.referral_pct_for_modules` + `referral_cert_anchors` default

**Files:**
- Modify: `dashboard/rewards.py` (DEFAULTS + new function)
- Test: `tests/test_rewards_model.py` (append)

- [ ] **Step 1: Write the failing test** (append to `tests/test_rewards_model.py`):

```python
def test_referral_pct_for_modules_interpolates():
    from dashboard import rewards
    s = rewards.load_settings({})            # defaults include referral_cert_anchors
    assert rewards.referral_pct_for_modules(0, s) == 0.05     # 5%
    assert rewards.referral_pct_for_modules(6, s) == 0.10     # 10%
    assert rewards.referral_pct_for_modules(12, s) == 0.15    # 15%
    assert abs(rewards.referral_pct_for_modules(3, s) - 0.075) < 1e-9   # midpoint
    assert rewards.referral_pct_for_modules(99, s) == 0.15    # flat beyond last
    assert rewards.referral_pct_for_modules(-4, s) == 0.05    # clamp at 0


def test_referral_pct_for_modules_falls_back_to_flat_when_no_anchors():
    from dashboard import rewards
    s = rewards.load_settings({"referral_cert_anchors": None})
    # None override is skipped by load_settings -> default anchors remain; force-remove:
    s2 = dict(s); s2.pop("referral_cert_anchors", None)
    assert rewards.referral_pct_for_modules(12, s2) == s2["referral_reward_pct"]


def test_referral_pct_for_modules_bad_anchors_falls_back():
    from dashboard import rewards
    s = dict(rewards.load_settings({}))
    s["referral_cert_anchors"] = "garbage"
    assert rewards.referral_pct_for_modules(12, s) == s["referral_reward_pct"]
```

- [ ] **Step 2: Run → fail.**
Run: `~/.venvs/deploy-chat311/bin/python -m pytest tests/test_rewards_model.py -q`
Expected: FAIL (no `referral_pct_for_modules`, and the default lacks anchors).

- [ ] **Step 3: Implement** in `dashboard/rewards.py`.

Add the anchor default to DEFAULTS:
```python
DEFAULTS = {
    "referral_reward_pct": 0.05,
    "cash_out_threshold_cents": 10000,
    "cash_out_face_pct": 0.70,
    # referral reward by certification: [modules_completed, whole-pct] knots, ascending;
    # linear-interpolated, flat beyond the last. modules 0 == base 5% (same as a plain affiliate).
    "referral_cert_anchors": [[0, 5], [6, 10], [12, 15]],
}
```

Add the interpolator (place after `load_settings`):
```python
def referral_pct_for_modules(modules, settings):
    """Referral reward FRACTION (0-1) for a practitioner with `modules` completed,
    interpolated through settings['referral_cert_anchors'] ([modules, whole-pct] knots,
    ascending; flat beyond the last). Falls back to the flat referral_reward_pct when the
    anchors are absent or malformed."""
    flat = float(settings.get("referral_reward_pct", 0.05))
    anchors = settings.get("referral_cert_anchors")
    try:
        if not anchors:
            return flat
        m = max(0, int(modules or 0))
        if m <= anchors[0][0]:
            pct = float(anchors[0][1])
        else:
            pct = float(anchors[-1][1])
            for (m0, p0), (m1, p1) in zip(anchors, anchors[1:]):
                if m <= m1:
                    pct = p0 + (p1 - p0) * (m - m0) / (m1 - m0)
                    break
        return pct / 100.0
    except Exception:
        return flat
```

- [ ] **Step 4: Run → pass.**
Run: `~/.venvs/deploy-chat311/bin/python -m pytest tests/test_rewards_model.py -q`
Expected: PASS.

- [ ] **Step 5: Commit** — `feat(dropship-rewards): referral_pct_for_modules + cert anchor default`

---

### Task 2: `practitioner_portal.modules_completed_for_email`

**Files:**
- Modify: `dashboard/practitioner_portal.py`
- Test: `tests/test_practitioner_portal.py` (append; reuse its `_FakeCtx` Supabase stub)

- [ ] **Step 1: Write the failing test.** First read the top of `tests/test_practitioner_portal.py` to reuse its existing `_FakeCtx` / fake-cursor helper (it monkeypatches `db_supabase.supabase_cursor`). Append:

```python
def test_modules_completed_for_email(monkeypatch):
    import db_supabase
    from dashboard import practitioner_portal as pp

    class _Cur:
        def __init__(self, row): self._row = row
        def execute(self, *a, **k): self._a = a
        def fetchone(self): return self._row

    # a practitioner with 9 modules
    monkeypatch.setattr(db_supabase, "supabase_cursor", lambda: _FakeCtx(_Cur({"modules_completed": 9})))
    assert pp.modules_completed_for_email("doc@x.com") == 9

    # null modules -> 0
    monkeypatch.setattr(db_supabase, "supabase_cursor", lambda: _FakeCtx(_Cur({"modules_completed": None})))
    assert pp.modules_completed_for_email("doc@x.com") == 0

    # not a practitioner -> None
    monkeypatch.setattr(db_supabase, "supabase_cursor", lambda: _FakeCtx(_Cur(None)))
    assert pp.modules_completed_for_email("nobody@x.com") is None

    # empty email -> None (no query)
    assert pp.modules_completed_for_email("") is None
```

> If the existing `_FakeCtx` in this test file has a different shape (e.g. expects a cursor with specific methods), adapt the `_Cur` stub to match what `_FakeCtx` wraps — keep the four assertions identical.

- [ ] **Step 2: Run → fail.**
Run: `doppler run -p remedy-match -c prd -- env DATA_DIR="$HOME/deploy-chat" ~/.venvs/deploy-chat311/bin/python -m pytest tests/test_practitioner_portal.py -q`
Expected: FAIL (`modules_completed_for_email` not defined).

- [ ] **Step 3: Implement** in `dashboard/practitioner_portal.py` (near `find_practitioner_id_by_email`):

```python
def modules_completed_for_email(email) -> Optional[int]:
    """modules_completed for the practitioner with this email, or None if the email is
    blank or not a practitioner. Null modules count as 0."""
    if not email:
        return None
    from db_supabase import supabase_cursor
    with supabase_cursor() as cur:
        cur.execute("SELECT modules_completed FROM practitioners "
                    "WHERE lower(email)=lower(%s) AND portal_role IS NOT NULL LIMIT 1",
                    (str(email).strip(),))
        row = cur.fetchone()
    return int(row["modules_completed"] or 0) if row else None
```

- [ ] **Step 4: Run → pass.**
- [ ] **Step 5: Commit** — `feat(dropship-rewards): modules_completed_for_email lookup`

---

### Task 3: cert resolver in `_settle_referral`

**Files:**
- Modify: `app.py` (add `_referral_pct_for_referrer`; rewire the flat read in `_settle_referral`)
- Test: `tests/test_referral_settlement.py` (append)

- [ ] **Step 1: Write the failing test** (append to `tests/test_referral_settlement.py`; reuse its `_db` + `_refer` helpers):

```python
def test_referral_cert_scaled_for_practitioner(monkeypatch, tmp_path):
    cx = _db(monkeypatch, tmp_path)
    _refer(cx, "buyer@x.com", "doc", "doc@x.com", ["type:practitioner"])
    # referrer is a practitioner with 12 modules -> 15%
    monkeypatch.setattr(appmod._pp, "modules_completed_for_email", lambda e: 12)
    order = {"email": "buyer@x.com", "total_cents": 6000, "shipping_cents": 0, "get_cents": 0,
             "discount_cents": 0, "points_redeemed_cents": 0}
    appmod._settle_referral(order, order_ref="INVC1")
    assert points.balance(cx, "doc@x.com") == 900        # 15% of 6000


def test_referral_base_pct_for_non_practitioner(monkeypatch, tmp_path):
    cx = _db(monkeypatch, tmp_path)
    _refer(cx, "buyer@x.com", "doc", "doc@x.com", ["type:practitioner"])
    # not a practitioner record -> base 5%
    monkeypatch.setattr(appmod._pp, "modules_completed_for_email", lambda e: None)
    order = {"email": "buyer@x.com", "total_cents": 6000, "shipping_cents": 0, "get_cents": 0,
             "discount_cents": 0, "points_redeemed_cents": 0}
    appmod._settle_referral(order, order_ref="INVC2")
    assert points.balance(cx, "doc@x.com") == 300        # 5% of 6000


def test_referral_cert_lookup_failure_falls_back_to_base(monkeypatch, tmp_path):
    cx = _db(monkeypatch, tmp_path)
    _refer(cx, "buyer@x.com", "doc", "doc@x.com", ["type:practitioner"])
    def _boom(e): raise RuntimeError("supabase down")
    monkeypatch.setattr(appmod._pp, "modules_completed_for_email", _boom)
    order = {"email": "buyer@x.com", "total_cents": 6000, "shipping_cents": 0, "get_cents": 0,
             "discount_cents": 0, "points_redeemed_cents": 0}
    appmod._settle_referral(order, order_ref="INVC3")
    assert points.balance(cx, "doc@x.com") == 300        # base 5% on lookup failure
```

The existing `test_points_referrer_credited` (no monkeypatch of `modules_completed_for_email`) must still pass: with `REWARDS_TIERS_ENABLED` on and no monkeypatch, the resolver calls the real `_pp.modules_completed_for_email`, which would hit Supabase. To keep that test hermetic, the resolver MUST swallow any lookup exception and fall back to base 5% (the real lookup will raise/return without creds in CI) — so `doc@x.com` still gets 300. Verify this test stays green in Step 4.

- [ ] **Step 2: Run → fail.**
Run: `doppler run -p remedy-match -c prd -- env DATA_DIR="$HOME/deploy-chat" ~/.venvs/deploy-chat311/bin/python -m pytest tests/test_referral_settlement.py -q`
Expected: the 3 new tests FAIL (cert scaling not applied).

- [ ] **Step 3: Implement** in `app.py`.

Add the resolver (place just above `_settle_referral`):
```python
def _referral_pct_for_referrer(referrer_email, settings):
    """Cert-scaled referral FRACTION. If the referrer is a practitioner, interpolate their
    modules_completed through referral_cert_anchors; otherwise (or on any lookup error) the
    base referral_reward_pct. Best-effort: never raises."""
    from dashboard import rewards as _rewards
    base = float(settings.get("referral_reward_pct", 0.05))
    try:
        modules = _pp.modules_completed_for_email(referrer_email)
        if modules is None:
            return base
        return _rewards.referral_pct_for_modules(modules, settings)
    except Exception as _e:
        print(f"[rewards] cert lookup failed for {referrer_email}: {_e!r}", flush=True)
        return base
```

In `_settle_referral`, replace:
```python
            settings = _rewards.load_settings(_rewards_settings())
            referral_reward_pct = float(settings["referral_reward_pct"])
            reward = round(product_cents * referral_reward_pct)
```
with:
```python
            settings = _rewards.load_settings(_rewards_settings())
            referral_reward_pct = _referral_pct_for_referrer(referrer_email, settings)
            reward = round(product_cents * referral_reward_pct)
```

- [ ] **Step 4: Run → pass** (all of `test_referral_settlement.py`, including the pre-existing tests).
Run: `doppler run -p remedy-match -c prd -- env DATA_DIR="$HOME/deploy-chat" ~/.venvs/deploy-chat311/bin/python -m pytest tests/test_referral_settlement.py -q`
Expected: PASS.

- [ ] **Step 5: Commit** — `feat(dropship-rewards): cert-scaled referral pct in _settle_referral`

---

### Task 4: validate `referral_cert_anchors` in `pricing_settings`

**Files:**
- Modify: `dashboard/pricing_settings.py` (extract `_validate_anchors`; reuse for volume + referral; copy anchors in `defaults_view`)
- Test: `tests/test_pricing_settings.py` (append)

- [ ] **Step 1: Write the failing test** (append):

```python
def test_defaults_view_includes_referral_cert_anchors():
    d = ps.defaults_view()
    assert d["rewards"]["referral_cert_anchors"] == [[0, 5], [6, 10], [12, 15]]
    # must be a copy, not the shared module default
    d["rewards"]["referral_cert_anchors"][0][1] = 99
    assert _rewards.DEFAULTS["referral_cert_anchors"][0][1] == 5


def test_validate_accepts_referral_cert_anchors():
    clean, errors = ps.validate({"rewards": {"referral_cert_anchors": [[0, 5], [4, 9], [12, 15]]}})
    assert errors == []
    assert clean["rewards"]["referral_cert_anchors"] == [[0, 5], [4, 9], [12, 15]]


def test_validate_rejects_bad_referral_cert_anchors():
    _, e = ps.validate({"rewards": {"referral_cert_anchors": [[4, 9], [0, 5]]}})   # not ascending
    assert any("referral_cert_anchors" in x for x in e)
    _, e = ps.validate({"rewards": {"referral_cert_anchors": [[0, 150]]}})         # pct > 100
    assert any("referral_cert_anchors" in x for x in e)
    _, e = ps.validate({"rewards": {"referral_cert_anchors": [[-1, 5]]}})          # modules < 0
    assert any("referral_cert_anchors" in x for x in e)


def test_volume_anchors_still_validated_after_refactor():
    _, e = ps.validate({"volume_anchors": [[3, 14], [1, 0]]})   # not ascending
    assert any("volume_anchors" in x for x in e)
    clean, e2 = ps.validate({"volume_anchors": [[1, 0], [3, 14]]})
    assert e2 == [] and clean["volume_anchors"] == [[1, 0], [3, 14]]
```

- [ ] **Step 2: Run → fail.**
Run: `~/.venvs/deploy-chat311/bin/python -m pytest tests/test_pricing_settings.py -q`
Expected: the new tests FAIL.

- [ ] **Step 3: Implement** in `dashboard/pricing_settings.py`.

Add the shared anchor validator (near `_check_fraction`):
```python
def _validate_anchors(anchors, min_x):
    """Validate + normalize an ascending [x, pct] anchor table: x integer >= min_x, strictly
    ascending; pct a number in [0, 100]. Returns the normalized list, or None if invalid."""
    if not (isinstance(anchors, list) and len(anchors) >= 1):
        return None
    norm, last = [], None
    for pair in anchors:
        if not (isinstance(pair, (list, tuple)) and len(pair) == 2
                and _is_number(pair[0]) and _is_number(pair[1])):
            return None
        x, p = int(pair[0]), float(pair[1])
        if x < min_x or not (0.0 <= p <= 100.0):
            return None
        if last is not None and x <= last:
            return None
        last = x
        norm.append([x, int(p) if float(p).is_integer() else p])
    return norm
```

Replace the existing inline `volume_anchors` block in `validate` with:
```python
    if "volume_anchors" in payload:
        norm = _validate_anchors(payload["volume_anchors"], 1)
        if norm is not None:
            clean["volume_anchors"] = norm
        else:
            errors.append("volume_anchors must be ascending [months>=1, pct 0-100] pairs")
```

In the `rewards` block of `validate`, after the existing rewards-fraction + threshold checks (inside `if "rewards" in payload:`, building `rclean`), add:
```python
        if "referral_cert_anchors" in rwd:
            norm = _validate_anchors(rwd["referral_cert_anchors"], 0)
            if norm is not None:
                rclean["referral_cert_anchors"] = norm
            else:
                errors.append("referral_cert_anchors must be ascending [modules>=0, pct 0-100] pairs")
```

In `defaults_view`, after `d["rewards"] = dict(_rewards.DEFAULTS)`, copy the anchor list so callers can't mutate the module default:
```python
    d["rewards"] = dict(_rewards.DEFAULTS)
    if isinstance(d["rewards"].get("referral_cert_anchors"), list):
        d["rewards"]["referral_cert_anchors"] = [list(a) for a in d["rewards"]["referral_cert_anchors"]]
```

- [ ] **Step 4: Run → pass** (full `tests/test_pricing_settings.py`, confirming the volume-anchor refactor didn't regress).
- [ ] **Step 5: Commit** — `feat(dropship-rewards): validate referral_cert_anchors (shared anchor validator)`

---

### Task 5: editor page — referral cert anchor table

**Files:**
- Modify: `static/console-pricing-settings.html`

- [ ] **Step 1:** Read `static/console-pricing-settings.html`. In the "Rewards & referrals" section, add a referral-cert anchor TABLE titled "Referral reward by certification (modules → %)", reusing the existing volume-anchors table pattern (rows of two number inputs: modules int min 0, pct 0-100; "Add" / "Remove" buttons). Help text: "How a referring practitioner's reward % scales with completed certification modules; a non-practitioner referrer always earns the base referral reward above."
- [ ] **Step 2:** On load, populate it from `effective.rewards.referral_cert_anchors` (fall back to `defaults.rewards.referral_cert_anchors`). The pct is whole percentage points (NOT converted — same as the volume table). On Save, include `rewards.referral_cert_anchors: [[modules, pct], ...]` in the POST payload (the rewards object already carries referral %, cash-out threshold/face).
- [ ] **Step 3:** Verify the page still parses: `~/.venvs/deploy-chat311/bin/python -c "import pathlib,html.parser; html.parser.HTMLParser().feed(pathlib.Path('static/console-pricing-settings.html').read_text()); print('ok')"`. Re-run the route tests (they assert the page serves + no-store): `doppler run -p remedy-match -c prd -- env DATA_DIR="$HOME/deploy-chat" ~/.venvs/deploy-chat311/bin/python -m pytest tests/test_console_pricing_settings_routes.py -q` → PASS.
- [ ] **Step 4:** Commit — `feat(dropship-rewards): referral cert anchor table in settings editor`

---

### Task 6: doc + suite

**Files:**
- Modify: `docs/console-settings.md` (add `referral_cert_anchors`)

- [ ] **Step 1:** Add a row/paragraph to `docs/console-settings.md`: `rewards.referral_cert_anchors` — referral reward by certification (`[modules_completed, whole-pct]` knots, interpolated, flat beyond the last; default `[[0,5],[6,10],[12,15]]`); only applies to referrers who are practitioners (non-practitioners stay at base `referral_reward_pct`); gated by `REWARDS_TIERS_ENABLED`.
- [ ] **Step 2:** Run the combined suite — green:
`doppler run -p remedy-match -c prd -- env DATA_DIR="$HOME/deploy-chat" ~/.venvs/deploy-chat311/bin/python -m pytest tests/test_rewards_model.py tests/test_referral_settlement.py tests/test_pricing_settings.py tests/test_console_pricing_settings_routes.py tests/test_practitioner_portal.py -q`
Expected: PASS (ignore the 2 known pre-existing failures if the run is broadened).
- [ ] **Step 3:** Commit — `docs(dropship-rewards): document referral_cert_anchors`

---

## Self-review

- **Spec coverage:** cert-tiered referral via `referral_cert_anchors` curve (Task 1 default + interp, Task 3 wiring); practitioner-only scaling, non-practitioner base 5% (Task 3 resolver); console-editable + validated (Task 4, Task 5); safe fallbacks when anchors absent/malformed or the Postgres lookup fails (Task 1 + Task 3 try/except); gated by `REWARDS_TIERS_ENABLED` (unchanged `_rewards_enabled()` guard in `_settle_referral`).
- **Type consistency:** `referral_pct_for_modules(modules, settings) -> fraction`; `modules_completed_for_email(email) -> Optional[int]`; `_referral_pct_for_referrer(referrer_email, settings) -> fraction`; anchors are `[modules, whole-pct]` everywhere (file + UI), fraction only at the multiply site.
- **Deferred (Plan 2 / out of scope):** patient channel-locked points; pro-influencer cert scaling (no module record); refund reversal.
- **Risk:** mis-scaled payout — bounded by validation (ascending, pct 0-100) and the points/cash floors are unaffected; cert lookup is best-effort and falls back to base 5%, so a Postgres outage degrades to today's behavior, never an error.

## Done
A referring practitioner's reward scales with certification (5% → 15% across 0 → 12 modules), console-editable, with non-practitioner referrers unchanged at 5%.
