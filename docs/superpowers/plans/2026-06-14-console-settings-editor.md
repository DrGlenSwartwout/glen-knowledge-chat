# Console Settings Editor Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make the pricing-engine and rewards "go-live tunables" (volume curve anchors, 57%/43% floors, 5% points earn, referral 5% / $100 cash-out threshold / 70% face) editable in the Console, persisted to `pricing-settings.json` in `DATA_DIR`, with live-reload so an edit takes effect on the next order without a redeploy.

**Architecture:** A pure `dashboard/pricing_settings.py` module owns the file *shape*, validation, and the effective-merge view (it leans on the existing `pricing.DEFAULTS` and `rewards.DEFAULTS`). `app.py` replaces the import-time `_PRICING_SETTINGS` global with an mtime-cached `_pricing_settings()` accessor (live-reload) plus a `_rewards_settings()` accessor, and rewires every consumer. A CONSOLE_SECRET-gated `GET/POST /api/console/pricing-settings` reads/validates/atomically-writes the file. A `/console/pricing-settings` page (`static/console-pricing-settings.html`, mirroring `console-products.html`) edits every value and can hit the existing `/api/pricing/preview` to sanity-check.

**Tech Stack:** Python 3.11, Flask, vanilla JS/HTML, pytest. Integer cents throughout. Percentages stored as fractions (0.57) except `volume_anchors` whose second element is whole percentage points (0–100), matching the existing engine.

**Scope note:** Per-SKU floor/MAP overrides are intentionally OUT of scope — they already live on the product record (`product["wholesale_cents"]`, `sku_discount_floor_pct`, `sku_points_floor_pct`) read by `pricing.unit_floor_cents`, and the practitioner MAP lives in `practitioner_settings`. This editor covers the GLOBAL engine + rewards tunables only.

---

## File shape: `pricing-settings.json` (in DATA_DIR)

The file does not exist today (engine falls back to `pricing.DEFAULTS`). First Save materializes it on the Render persistent disk. Shape:

```json
{
  "discount_floor_pct": 0.57,
  "points_floor_pct": 0.43,
  "points_earn_pct": 0.05,
  "points_redeem_per_point_cents": 5,
  "subscribe_tiers": [5, 10, 15],
  "cadences": [1, 2, 3],
  "volume_anchors": [[1, 0], [3, 14], [6, 29], [12, 43]],
  "rewards": {
    "referral_reward_pct": 0.05,
    "cash_out_threshold_cents": 10000,
    "cash_out_face_pct": 0.70
  }
}
```

## File structure

- Create `dashboard/pricing_settings.py` — pure: `defaults_view()`, `effective(raw)`, `validate(payload) -> (clean, errors)`.
- Modify `app.py` — replace `_PRICING_SETTINGS` global (line ~96) with `_pricing_settings()` + `_rewards_settings()` accessors; rewire consumers (2409, 2559, 15770, 2491, 2514); add `/api/console/pricing-settings` GET/POST and `/console/pricing-settings` page route.
- Create `static/console-pricing-settings.html` — the editor page (mirror `console-products.html` gate/key/op-nav).
- Create `docs/console-settings.md` — doc.
- Tests: `tests/test_pricing_settings.py` (module), `tests/test_console_pricing_settings_routes.py` (API + page + live-reload).

---

### Task 1: `dashboard/pricing_settings.py` — defaults view, effective merge, validation

**Files:**
- Create: `dashboard/pricing_settings.py`
- Test: `tests/test_pricing_settings.py`

- [ ] **Step 1: Write the failing test**

```python
# tests/test_pricing_settings.py
from dashboard import pricing_settings as ps
from dashboard import pricing as _pricing
from dashboard import rewards as _rewards


def test_defaults_view_combines_pricing_and_rewards():
    d = ps.defaults_view()
    assert d["discount_floor_pct"] == _pricing.DEFAULTS["discount_floor_pct"]
    assert d["volume_anchors"] == _pricing.DEFAULTS["volume_anchors"]
    assert d["rewards"]["referral_reward_pct"] == _rewards.DEFAULTS["referral_reward_pct"]
    assert d["rewards"]["cash_out_threshold_cents"] == _rewards.DEFAULTS["cash_out_threshold_cents"]


def test_effective_empty_raw_equals_defaults():
    eff = ps.effective({})
    assert eff["discount_floor_pct"] == _pricing.DEFAULTS["discount_floor_pct"]
    assert eff["points_earn_pct"] == _pricing.DEFAULTS["points_earn_pct"]
    assert eff["rewards"]["cash_out_face_pct"] == _rewards.DEFAULTS["cash_out_face_pct"]
    # the nested 'rewards' key must NOT leak into the pricing top level
    assert "rewards" in eff and isinstance(eff["rewards"], dict)


def test_effective_overrides_merge():
    eff = ps.effective({"discount_floor_pct": 0.50, "rewards": {"referral_reward_pct": 0.08}})
    assert eff["discount_floor_pct"] == 0.50
    assert eff["points_floor_pct"] == _pricing.DEFAULTS["points_floor_pct"]   # untouched
    assert eff["rewards"]["referral_reward_pct"] == 0.08
    assert eff["rewards"]["cash_out_threshold_cents"] == _rewards.DEFAULTS["cash_out_threshold_cents"]


def test_validate_accepts_full_valid_payload():
    payload = {
        "discount_floor_pct": 0.57, "points_floor_pct": 0.43, "points_earn_pct": 0.05,
        "points_redeem_per_point_cents": 5, "subscribe_tiers": [5, 10, 15],
        "cadences": [1, 2, 3], "volume_anchors": [[1, 0], [3, 14], [6, 29], [12, 43]],
        "rewards": {"referral_reward_pct": 0.05, "cash_out_threshold_cents": 10000,
                    "cash_out_face_pct": 0.70},
    }
    clean, errors = ps.validate(payload)
    assert errors == []
    assert clean["discount_floor_pct"] == 0.57
    assert clean["volume_anchors"] == [[1, 0], [3, 14], [6, 29], [12, 43]]
    assert clean["rewards"]["cash_out_threshold_cents"] == 10000


def test_validate_rejects_out_of_range_fractions():
    _, errors = ps.validate({"discount_floor_pct": 1.5})
    assert any("discount_floor_pct" in e for e in errors)
    _, errors = ps.validate({"points_earn_pct": -0.1})
    assert any("points_earn_pct" in e for e in errors)


def test_validate_rejects_points_floor_above_discount_floor():
    # points floor must sit at or below the discount floor (points can discount deeper)
    _, errors = ps.validate({"discount_floor_pct": 0.40, "points_floor_pct": 0.50})
    assert any("points_floor_pct" in e for e in errors)


def test_validate_rejects_bad_volume_anchors():
    _, errors = ps.validate({"volume_anchors": [[3, 14], [1, 0]]})          # not ascending
    assert any("volume_anchors" in e for e in errors)
    _, errors = ps.validate({"volume_anchors": [[1, 0], [3, 150]]})         # pct > 100
    assert any("volume_anchors" in e for e in errors)
    _, errors = ps.validate({"volume_anchors": [[1, 0, 9]]})                # not a pair
    assert any("volume_anchors" in e for e in errors)


def test_validate_rejects_bad_rewards():
    _, errors = ps.validate({"rewards": {"cash_out_threshold_cents": -5}})
    assert any("cash_out_threshold_cents" in e for e in errors)
    _, errors = ps.validate({"rewards": {"cash_out_face_pct": 2.0}})
    assert any("cash_out_face_pct" in e for e in errors)


def test_validate_ignores_unknown_keys():
    clean, errors = ps.validate({"discount_floor_pct": 0.57, "bogus": 123})
    assert errors == []
    assert "bogus" not in clean


def test_validate_partial_payload_only_validates_present_keys():
    # a partial save (only one field) is allowed; absent keys simply aren't in clean
    clean, errors = ps.validate({"points_earn_pct": 0.06})
    assert errors == []
    assert clean == {"points_earn_pct": 0.06}
```

- [ ] **Step 2: Run → fail.**

Run: `~/.venvs/deploy-chat311/bin/python -m pytest tests/test_pricing_settings.py -q`
Expected: FAIL (module not found).

- [ ] **Step 3: Implement**

```python
# dashboard/pricing_settings.py
"""Shape, validation, and effective-merge for the console-editable pricing + rewards
settings (persisted to DATA_DIR/pricing-settings.json). Pure: no Flask, no file IO."""
from dashboard import pricing as _pricing
from dashboard import rewards as _rewards

# fraction keys live in [0, 1]
_PRICING_FRACTIONS = ("discount_floor_pct", "points_floor_pct", "points_earn_pct")
_REWARDS_FRACTIONS = ("referral_reward_pct", "cash_out_face_pct")


def defaults_view():
    """Built-in defaults in the file's shape (rewards nested)."""
    d = dict(_pricing.DEFAULTS)
    d["volume_anchors"] = [list(a) for a in _pricing.DEFAULTS["volume_anchors"]]
    d["subscribe_tiers"] = list(_pricing.DEFAULTS["subscribe_tiers"])
    d["cadences"] = list(_pricing.DEFAULTS["cadences"])
    d["rewards"] = dict(_rewards.DEFAULTS)
    return d


def effective(raw):
    """Merge raw overrides over the defaults, returning the file-shaped effective view.
    The nested 'rewards' overrides are kept out of the pricing merge."""
    raw = raw or {}
    pricing_over = {k: v for k, v in raw.items() if k != "rewards"}
    eff = _pricing.load_settings(pricing_over)
    eff["rewards"] = _rewards.load_settings(raw.get("rewards") or {})
    return eff


def _is_number(v):
    return isinstance(v, (int, float)) and not isinstance(v, bool)


def _check_fraction(name, v, errors):
    if not _is_number(v) or not (0.0 <= float(v) <= 1.0):
        errors.append(f"{name} must be a number between 0 and 1")
        return None
    return float(v)


def validate(payload):
    """Validate a (possibly partial) settings payload. Returns (clean, errors).
    clean contains only known, well-formed keys in the file shape; errors is a list
    of human-readable messages (non-empty => caller should reject)."""
    payload = payload or {}
    clean, errors = {}, []

    for name in _PRICING_FRACTIONS:
        if name in payload:
            v = _check_fraction(name, payload[name], errors)
            if v is not None:
                clean[name] = v

    if "points_redeem_per_point_cents" in payload:
        v = payload["points_redeem_per_point_cents"]
        if isinstance(v, int) and not isinstance(v, bool) and v >= 1:
            clean["points_redeem_per_point_cents"] = v
        else:
            errors.append("points_redeem_per_point_cents must be an integer >= 1")

    for name in ("subscribe_tiers", "cadences"):
        if name in payload:
            v = payload[name]
            if (isinstance(v, list) and v
                    and all(_is_number(x) and x >= 0 for x in v)):
                clean[name] = [int(x) if float(x).is_integer() else float(x) for x in v]
            else:
                errors.append(f"{name} must be a non-empty list of non-negative numbers")

    if "volume_anchors" in payload:
        anchors = payload["volume_anchors"]
        ok = isinstance(anchors, list) and len(anchors) >= 1
        norm = []
        if ok:
            last_m = None
            for pair in anchors:
                if not (isinstance(pair, (list, tuple)) and len(pair) == 2
                        and _is_number(pair[0]) and _is_number(pair[1])):
                    ok = False
                    break
                m, p = int(pair[0]), float(pair[1])
                if m < 1 or not (0.0 <= p <= 100.0):
                    ok = False
                    break
                if last_m is not None and m <= last_m:
                    ok = False
                    break
                last_m = m
                norm.append([m, int(p) if float(p).is_integer() else p])
        if ok:
            clean["volume_anchors"] = norm
        else:
            errors.append("volume_anchors must be ascending [months>=1, pct 0-100] pairs")

    if "rewards" in payload:
        rwd = payload["rewards"] or {}
        rclean = {}
        for name in _REWARDS_FRACTIONS:
            if name in rwd:
                v = _check_fraction(name, rwd[name], errors)
                if v is not None:
                    rclean[name] = v
        if "cash_out_threshold_cents" in rwd:
            v = rwd["cash_out_threshold_cents"]
            if isinstance(v, int) and not isinstance(v, bool) and v >= 0:
                rclean["cash_out_threshold_cents"] = v
            else:
                errors.append("cash_out_threshold_cents must be an integer >= 0")
        clean["rewards"] = rclean

    # cross-field: points floor must sit at or below the discount floor. Compare against
    # the effective values so a partial save is still checked.
    df = clean.get("discount_floor_pct", _pricing.DEFAULTS["discount_floor_pct"])
    pf = clean.get("points_floor_pct", _pricing.DEFAULTS["points_floor_pct"])
    if _is_number(df) and _is_number(pf) and pf > df:
        errors.append("points_floor_pct must be <= discount_floor_pct")

    return clean, errors
```

- [ ] **Step 4: Run → pass.**

Run: `~/.venvs/deploy-chat311/bin/python -m pytest tests/test_pricing_settings.py -q`
Expected: PASS.

- [ ] **Step 5: Commit** — `feat(console-settings): pricing_settings module (defaults/effective/validate)`

---

### Task 2: live-reload accessor in app.py + rewire consumers

**Files:**
- Modify: `app.py` (line ~96 load; consumers at 2409, 2559, 15770, 2491, 2514)
- Test: `tests/test_console_pricing_settings_routes.py` (live-reload portion)

- [ ] **Step 1: Write the failing test** (the live-reload behaviour)

```python
# tests/test_console_pricing_settings_routes.py
import json
import importlib


def _client(tmp_path, monkeypatch):
    monkeypatch.setenv("DATA_DIR", str(tmp_path))
    import app as _app
    importlib.reload(_app)
    _app.app.config["TESTING"] = True
    return _app


def test_pricing_settings_accessor_live_reloads(tmp_path, monkeypatch):
    _app = _client(tmp_path, monkeypatch)
    # no file yet -> empty overrides -> engine uses DEFAULTS
    assert _app._pricing_settings() == {}
    assert _app._rewards_settings() == {}
    # write a file -> next read sees it (mtime changed)
    path = _app._PRICING_SETTINGS_PATH
    path.write_text(json.dumps({"discount_floor_pct": 0.50,
                                "rewards": {"referral_reward_pct": 0.08}}))
    # bump mtime explicitly in case the write lands in the same coarse tick
    import os, time
    os.utime(path, (time.time() + 1, time.time() + 1))
    assert _app._pricing_settings()["discount_floor_pct"] == 0.50
    assert _app._rewards_settings()["referral_reward_pct"] == 0.08
```

> NOTE: `app.py` builds OpenAI/Pinecone/sqlite at import, so this test must run under the documented full-env invocation (Doppler + DATA_DIR), not the bare venv. See Step 4.

- [ ] **Step 2: Run → fail.**

Run (documented full-env invocation):
`doppler run -p remedy-match -c prd -- env DATA_DIR="$HOME/deploy-chat" ~/.venvs/deploy-chat311/bin/python -m pytest tests/test_console_pricing_settings_routes.py::test_pricing_settings_accessor_live_reloads -q`
Expected: FAIL (`_pricing_settings` / `_PRICING_SETTINGS_PATH` not defined).

- [ ] **Step 3: Implement**

Replace the single line at app.py:96
```python
_PRICING_SETTINGS = _load_json(DATA_DIR / "pricing-settings.json", default={})
```
with:
```python
_PRICING_SETTINGS_PATH = DATA_DIR / "pricing-settings.json"
_PRICING_SETTINGS_CACHE = {"mtime": None, "data": {}}


def _pricing_settings():
    """Live-reloaded pricing+rewards overrides from pricing-settings.json.
    Re-reads only when the file's mtime changes, so a console edit takes effect on the
    next order without a redeploy. Returns {} when the file is absent (engine then uses
    the built-in pricing.DEFAULTS)."""
    try:
        mt = _PRICING_SETTINGS_PATH.stat().st_mtime
    except OSError:
        _PRICING_SETTINGS_CACHE["mtime"] = None
        _PRICING_SETTINGS_CACHE["data"] = {}
        return {}
    if _PRICING_SETTINGS_CACHE["mtime"] != mt:
        _PRICING_SETTINGS_CACHE["data"] = _load_json(_PRICING_SETTINGS_PATH, default={})
        _PRICING_SETTINGS_CACHE["mtime"] = mt
    return _PRICING_SETTINGS_CACHE["data"]


def _rewards_settings():
    """Rewards-engine overrides (nested under 'rewards' in pricing-settings.json)."""
    return (_pricing_settings().get("rewards") or {})
```

Then rewire the consumers (exact replacements):
- app.py:2409 `settings = _pricing.load_settings(_PRICING_SETTINGS)` → `settings = _pricing.load_settings(_pricing_settings())`
- app.py:15770 `settings = _pricing.load_settings(_PRICING_SETTINGS)` → `settings = _pricing.load_settings(_pricing_settings())`
- app.py:2559 `earn_pct = float(_PRICING_SETTINGS.get("points_earn_pct", 0.05)) if isinstance(_PRICING_SETTINGS, dict) else 0.05` → `earn_pct = float(_pricing_settings().get("points_earn_pct", 0.05))`
- app.py:2491 `settings = _rewards.load_settings({})` → `settings = _rewards.load_settings(_rewards_settings())`
- app.py:2514 `settings = _rewards.load_settings({})` → `settings = _rewards.load_settings(_rewards_settings())`

Confirm no other reference to `_PRICING_SETTINGS` remains: `grep -n "_PRICING_SETTINGS\b" app.py` should show only the new `_PRICING_SETTINGS_PATH` / `_PRICING_SETTINGS_CACHE` names.

- [ ] **Step 4: Run → pass.**

Run: `doppler run -p remedy-match -c prd -- env DATA_DIR="$HOME/deploy-chat" ~/.venvs/deploy-chat311/bin/python -m pytest tests/test_console_pricing_settings_routes.py::test_pricing_settings_accessor_live_reloads -q`
Expected: PASS.

- [ ] **Step 5: Commit** — `feat(console-settings): live-reload _pricing_settings accessor + rewire consumers`

---

### Task 3: `GET/POST /api/console/pricing-settings` (CONSOLE_SECRET-gated)

**Files:**
- Modify: `app.py` (add routes; reuse `_pricing_settings`, `_PRICING_SETTINGS_PATH`, `_PRICING_SETTINGS_CACHE`)
- Test: `tests/test_console_pricing_settings_routes.py` (append)

- [ ] **Step 1: Write the failing test** (append to the file from Task 2)

```python
def _key(_app):
    return _app.CONSOLE_SECRET or ""


def test_get_requires_console_key(tmp_path, monkeypatch):
    _app = _client(tmp_path, monkeypatch)
    if not _app.CONSOLE_SECRET:
        return  # auth is a no-op when unset in this env; nothing to assert
    c = _app.app.test_client()
    assert c.get("/api/console/pricing-settings").status_code == 401


def test_get_returns_defaults_when_no_file(tmp_path, monkeypatch):
    _app = _client(tmp_path, monkeypatch)
    c = _app.app.test_client()
    r = c.get("/api/console/pricing-settings", headers={"X-Console-Key": _key(_app)})
    assert r.status_code == 200
    body = r.get_json()
    assert body["saved"] == {}
    assert body["effective"]["discount_floor_pct"] == 0.57
    assert body["defaults"]["rewards"]["cash_out_threshold_cents"] == 10000


def test_post_persists_and_live_applies(tmp_path, monkeypatch):
    _app = _client(tmp_path, monkeypatch)
    c = _app.app.test_client()
    payload = {"discount_floor_pct": 0.55,
               "volume_anchors": [[1, 0], [3, 15], [6, 30], [12, 45]],
               "rewards": {"referral_reward_pct": 0.07, "cash_out_threshold_cents": 12000,
                           "cash_out_face_pct": 0.70}}
    r = c.post("/api/console/pricing-settings",
               headers={"X-Console-Key": _key(_app), "Content-Type": "application/json"},
               data=json.dumps(payload))
    assert r.status_code == 200, r.get_data(as_text=True)
    assert r.get_json()["saved"]["discount_floor_pct"] == 0.55
    # file written
    assert _app._PRICING_SETTINGS_PATH.exists()
    # live-reload: the accessor now reflects it (cache busted on write)
    assert _app._pricing_settings()["discount_floor_pct"] == 0.55
    assert _app._rewards_settings()["referral_reward_pct"] == 0.07


def test_post_rejects_invalid(tmp_path, monkeypatch):
    _app = _client(tmp_path, monkeypatch)
    c = _app.app.test_client()
    r = c.post("/api/console/pricing-settings",
               headers={"X-Console-Key": _key(_app), "Content-Type": "application/json"},
               data=json.dumps({"discount_floor_pct": 9.9}))
    assert r.status_code == 400
    assert any("discount_floor_pct" in e for e in r.get_json()["errors"])
    # nothing persisted
    assert not _app._PRICING_SETTINGS_PATH.exists()
```

- [ ] **Step 2: Run → fail.**

Run: `doppler run -p remedy-match -c prd -- env DATA_DIR="$HOME/deploy-chat" ~/.venvs/deploy-chat311/bin/python -m pytest tests/test_console_pricing_settings_routes.py -q`
Expected: the new route tests FAIL (404 / not defined).

- [ ] **Step 3: Implement** — add near the other console/pricing routes in app.py (e.g. just after `api_pricing_preview`):

```python
@app.route("/api/console/pricing-settings", methods=["GET", "POST"])
def api_console_pricing_settings():
    """Console-gated read/write of the global pricing + rewards tunables, persisted to
    pricing-settings.json in DATA_DIR (live-reloaded by _pricing_settings)."""
    if CONSOLE_SECRET:
        key = request.headers.get("X-Console-Key", "") or request.args.get("key", "")
        if key != CONSOLE_SECRET:
            return jsonify({"error": "Unauthorized"}), 401
    from dashboard import pricing_settings as _ps
    if request.method == "GET":
        raw = _pricing_settings()
        return jsonify({"saved": raw, "effective": _ps.effective(raw),
                        "defaults": _ps.defaults_view()})
    # POST
    payload = request.get_json(silent=True) or {}
    clean, errors = _ps.validate(payload)
    if errors:
        return jsonify({"errors": errors}), 400
    # atomic write, then bust the live-reload cache so the next read re-reads
    import tempfile
    _PRICING_SETTINGS_PATH.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(dir=str(_PRICING_SETTINGS_PATH.parent), suffix=".tmp")
    try:
        with os.fdopen(fd, "w") as f:
            json.dump(clean, f, indent=2)
        os.replace(tmp, _PRICING_SETTINGS_PATH)
    except Exception:
        if os.path.exists(tmp):
            os.remove(tmp)
        raise
    _PRICING_SETTINGS_CACHE["mtime"] = None      # force re-read on next access
    raw = _pricing_settings()
    return jsonify({"saved": raw, "effective": _ps.effective(raw)})
```

- [ ] **Step 4: Run → pass.**

Run: `doppler run -p remedy-match -c prd -- env DATA_DIR="$HOME/deploy-chat" ~/.venvs/deploy-chat311/bin/python -m pytest tests/test_console_pricing_settings_routes.py -q`
Expected: PASS.

- [ ] **Step 5: Commit** — `feat(console-settings): GET/POST /api/console/pricing-settings`

---

### Task 4: `/console/pricing-settings` page route + editor HTML

**Files:**
- Modify: `app.py` (add the page route)
- Create: `static/console-pricing-settings.html`
- Test: `tests/test_console_pricing_settings_routes.py` (append page-serve test)

- [ ] **Step 1: Write the failing test**

```python
def test_console_page_served_no_store(tmp_path, monkeypatch):
    _app = _client(tmp_path, monkeypatch)
    c = _app.app.test_client()
    r = c.get("/console/pricing-settings")
    assert r.status_code == 200
    assert b"Pricing" in r.data
    assert "no-store" in r.headers.get("Cache-Control", "")
```

- [ ] **Step 2: Run → fail.**

Run: `doppler run -p remedy-match -c prd -- env DATA_DIR="$HOME/deploy-chat" ~/.venvs/deploy-chat311/bin/python -m pytest tests/test_console_pricing_settings_routes.py::test_console_page_served_no_store -q`
Expected: FAIL (404).

- [ ] **Step 3: Implement**

Add the route in app.py (near `practitioner_settings_page`):
```python
@app.route("/console/pricing-settings")
def console_pricing_settings_page():
    resp = send_from_directory(STATIC, "console-pricing-settings.html")
    resp.headers["Cache-Control"] = "no-cache, no-store, must-revalidate"
    resp.headers["Pragma"] = "no-cache"
    return resp
```

Create `static/console-pricing-settings.html`. Mirror `console-products.html` for the gate + console-key handling + op-nav. Requirements:
- `<script src="/static/op-nav.js" data-active="bos" data-sub="pricing-settings"></script>` (op-nav renders the shared bar; `data-sub` just highlights — fine if it has no matching sub-tab yet).
- Console-key gate identical to console-products: `key()` reads `localStorage.getItem('console_key')`; auto-seed from `?key=`; `hdr()` returns `{'X-Console-Key': key(), 'Content-Type':'application/json'}`; an unlock input shown until a key is present.
- On load: `GET /api/console/pricing-settings` → populate the form from `effective`, show `defaults` as placeholders/help text.
- **Field rendering + unit conversion (critical — two representations):**
  - Fractions shown as **percent** (multiply by 100 for display, divide by 100 on save), with 2-decimal precision: `discount_floor_pct`, `points_floor_pct`, `points_earn_pct`, `rewards.referral_reward_pct`, `rewards.cash_out_face_pct`. Labels: "Wholesale / discount floor (% of list)", "Points floor (% of list)", "Points earned (% of full-price spend)", "Referral reward (% of referred sale)", "Cash-out face value (% paid out)".
  - `points_redeem_per_point_cents`: integer cents per point (label "Point redemption value (cents per point)").
  - `rewards.cash_out_threshold_cents`: shown as **dollars** (cents/100; ×100 on save), label "Cash-out review threshold ($)".
  - `volume_anchors`: a small editable table of rows, each `[months, pct]` where pct is already whole percentage points (0–100, NOT converted). "Add anchor" / "Remove" buttons. Help: "Linear-interpolated volume discount by total cart months; first row should be [1, 0]."
  - `subscribe_tiers` (comma-separated whole %: "5, 10, 15") and `cadences` (comma-separated months: "1, 2, 3") as simple text inputs parsed to number lists.
- **Save** button → build the payload back in the file shape (fractions divided by 100, dollars ×100, anchors as `[[m,pct],...]`, tiers/cadences parsed) → `POST` → on 200 show "Saved — applies to the next order"; on 400 list `errors`.
- **Preview** panel: a slug + qty + optional months/subscriber inputs → `POST /api/pricing/preview` with `{items:[{slug,qty}], subscriber_tier_pct, points_to_redeem_cents}` → show the returned line prices / discount / total so Glen can sanity-check the just-saved curve. (Preview reads the live engine, so it reflects the saved file.)
- Keep styling consistent with `console-products.html` (reuse its CSS variables / classes; same dark-aware theme bootstrap line). No em dashes, no ALL CAPS in copy.

- [ ] **Step 4: Run → pass.**

Run: `doppler run -p remedy-match -c prd -- env DATA_DIR="$HOME/deploy-chat" ~/.venvs/deploy-chat311/bin/python -m pytest tests/test_console_pricing_settings_routes.py -q`
Expected: PASS.

- [ ] **Step 5: Commit** — `feat(console-settings): /console/pricing-settings editor page`

---

### Task 5: doc + full suite

**Files:**
- Create: `docs/console-settings.md`

- [ ] **Step 1:** Run the new suites together — green.

Run: `doppler run -p remedy-match -c prd -- env DATA_DIR="$HOME/deploy-chat" ~/.venvs/deploy-chat311/bin/python -m pytest tests/test_pricing_settings.py tests/test_console_pricing_settings_routes.py -q`
Expected: PASS.

- [ ] **Step 2:** Create `docs/console-settings.md` covering: the editable tunables and their meaning; the `pricing-settings.json` shape (fractions vs volume-anchor whole-percent); live-reload (mtime-cached re-read; file absent => DEFAULTS); the API (`GET/POST /api/console/pricing-settings`, CONSOLE_SECRET-gated, validation) and the page (`/console/pricing-settings`); that values ship at current defaults until first Save; that per-SKU floor/MAP overrides remain in the Products / practitioner settings (out of scope here).

- [ ] **Step 3:** Run the broader suite once to confirm no regressions in pricing/rewards/checkout:

Run: `doppler run -p remedy-match -c prd -- env DATA_DIR="$HOME/deploy-chat" ~/.venvs/deploy-chat311/bin/python -m pytest tests/test_pricing.py tests/test_rewards.py -q` (and any reorder/subscription/checkout tests that import the engine). Ignore the 2 known pre-existing failures (`test_pf_playwright_fetch`, `test_bos_routes::test_home_page_served`).

- [ ] **Step 4:** Commit — `docs(console-settings): editor + live-reload doc`

---

## Self-review

- **Spec/decision coverage:** "Leave the 5%/$100/70% as-is but set them up as adjustable variables in the console" — referral 5% / cash-out $100 threshold / 70% face are in `rewards` (Task 1 shape, Task 3 persist, Task 4 UI), defaults unchanged. Engine globals (57/43 floors, 5% earn, redeem cents, volume anchors, tiers, cadences) all editable. Live-reload (Task 2) so edits apply without a redeploy. Preview to sanity-check (Task 4).
- **Type consistency:** `pricing_settings.defaults_view()/effective(raw)/validate(payload)->(clean,errors)`; `_pricing_settings()` / `_rewards_settings()` accessors; `_PRICING_SETTINGS_PATH` / `_PRICING_SETTINGS_CACHE`; `/api/console/pricing-settings` returns `{saved, effective, defaults?}`; POST 400 `{errors:[...]}`. The fraction-vs-percent and cents-vs-dollar conversions live ONLY in the UI; the file + API speak fractions/cents/whole-percent-anchors exactly as the engine does.
- **Deferred (YAGNI):** per-SKU floor/MAP overrides (already on the product record / practitioner settings); an audit log of who changed what; subscribe-tier escalation editing beyond the list; a dedicated op-nav sub-tab entry.
- **Risk:** a bad save could mis-price live orders — mitigated by range + cross-field validation (points floor ≤ discount floor, fractions 0–1, ascending anchors) and the Preview panel; the file is atomically written; if the file is ever unreadable the accessor returns `{}` and the engine falls back to safe DEFAULTS.

## Done
The pricing-engine and rewards go-live tunables are editable in the Console, persisted to `pricing-settings.json`, and live-reloaded so changes apply on the next order without a redeploy.
