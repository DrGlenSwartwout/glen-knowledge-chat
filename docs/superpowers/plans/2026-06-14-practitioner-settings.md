# Practitioner Settings: Pricing ($/%) + White-Label Branding — Plan 4 of 4

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development. Steps use checkbox (`- [ ]`) syntax.

**Goal:** Let a practitioner set their FF selling prices (a default markup % over retail, with optional per-SKU $ overrides) and their white-label branding (logo, photo, practice name, contact, web link, 2 colors) — and apply it: the client page now prices at the practitioner's set price (instead of defaulting to retail) and the three portal pages render the branding.

**Architecture:** A new local sqlite `practitioner_settings` table (chat_log.db) keyed by practitioner id, holding `branding_json` + `pricing_json` — avoids a Supabase migration and is testable with the sqlite seam. Read/write helpers in a new `dashboard/practitioner_settings.py`. Wire Plan 3's `dropship_checkout._practitioner_price_cents` to read it. A settings API + portal settings page. Branding applied to the client/drop-ship/wholesale pages.

**Tech Stack:** Python 3.11, Flask, sqlite (`LOG_DB`), pytest.

**Spec:** `docs/superpowers/specs/2026-06-14-practitioner-dropship-portal-design.md` (§C branding, §D data, §A.6 $/% pricing, §I open items 1-3). Resolves §I item 2 (default markup % + optional per-SKU override) and item 3 (image = URL field in v1; upload via `/clips/upload` is a later add).

**Reuse:** `practitioner_pricing.resolve_selling_cents`/`price_for_markup`/`markup_pct_for`/`MapViolation` (Plan 1), `dropship_checkout._practitioner_price_cents` (Plan 3 stub to wire), `_practitioner_session_pid`, the existing portal page-serve + styling, `static/practitioner-client.html` / `practitioner-dropship.html` (apply branding).

---

### Task 1: settings store

**Files:** Create `dashboard/practitioner_settings.py`; Test `tests/test_practitioner_settings.py`

- [ ] **Step 1: Failing test**

```python
# tests/test_practitioner_settings.py
import sqlite3
from dashboard import practitioner_settings as ps

def _cx():
    cx = sqlite3.connect(":memory:"); cx.row_factory = sqlite3.Row
    ps.init_settings_table(cx); return cx

def test_defaults_when_unset():
    cx = _cx()
    s = ps.get_settings(cx, "p1")
    assert s["branding"] == {} and s["pricing"] == {"default_markup_pct": 0, "overrides": {}}

def test_set_and_get_roundtrip():
    cx = _cx()
    ps.set_branding(cx, "p1", {"practice_name": "Acme", "brand_color_1": "#0a0",
                               "logo_url": "https://x/l.png"})
    ps.set_pricing(cx, "p1", {"default_markup_pct": 15, "overrides": {"brain-boost": 8500}})
    s = ps.get_settings(cx, "p1")
    assert s["branding"]["practice_name"] == "Acme"
    assert s["pricing"]["default_markup_pct"] == 15
    assert s["pricing"]["overrides"]["brain-boost"] == 8500

def test_price_for_uses_override_then_markup_then_retail():
    cx = _cx()
    ps.set_pricing(cx, "p1", {"default_markup_pct": 20, "overrides": {"a": 9000}})
    # override wins
    assert ps.price_cents_for(cx, "p1", "a", retail_cents=7000, map_cents=6700) == 9000
    # else markup over retail: 7000*1.20 = 8400
    assert ps.price_cents_for(cx, "p1", "b", retail_cents=7000, map_cents=6700) == 8400
    # markup that lands below MAP clamps up to MAP
    ps.set_pricing(cx, "p1", {"default_markup_pct": -20, "overrides": {}})
    assert ps.price_cents_for(cx, "p1", "b", retail_cents=7000, map_cents=6700) == 6700
```

- [ ] **Step 2: Run → fail.**

- [ ] **Step 3: Implement** `dashboard/practitioner_settings.py`:
- `init_settings_table(cx)`: `practitioner_settings(practitioner_id TEXT PRIMARY KEY, branding_json TEXT DEFAULT '{}', pricing_json TEXT DEFAULT '{}', updated_at TEXT)`.
- `get_settings(cx, pid)` → `{"branding": {...}, "pricing": {"default_markup_pct": 0, "overrides": {}}}` (defaults when no row / empty json).
- `set_branding(cx, pid, dict)` / `set_pricing(cx, pid, dict)` — upsert the respective JSON (ON CONFLICT(practitioner_id) DO UPDATE), touch updated_at.
- `price_cents_for(cx, pid, slug, *, retail_cents, map_cents)`: override if present, else `practitioner_pricing.price_for_markup(default_markup_pct, retail)`, else retail; **clamp up to map_cents**. (Reuse Plan 1 helpers.)

- [ ] **Step 4: Run → pass.**
- [ ] **Step 5: Commit** — `feat(practitioner-settings): sqlite store + price_cents_for (markup/override/MAP)`

---

### Task 2: wire the stored price into the client page

**Files:** Modify `dashboard/dropship_checkout.py` (`_practitioner_price_cents`); Test `tests/test_client_order.py` (extend)

- [ ] **Step 1: Failing test** — `_practitioner_price_cents(pid, slug, retail)` now reads `practitioner_settings.price_cents_for` (a stored markup/override) instead of always returning retail. Test: with a stored 20% markup, a $70 retail → $84; with no settings → retail; below-MAP markup → MAP. (Open a sqlite via `LOG_DB`; or have `_practitioner_price_cents` accept an injected getter the test stubs.)

- [ ] **Step 2: Run → fail.**

- [ ] **Step 3: Implement** — replace the Plan-3 stub body: open `sqlite3.connect(LOG_DB)` (or the app's connection helper), `practitioner_settings.init_settings_table(cx)`, return `practitioner_settings.price_cents_for(cx, pid, slug, retail_cents=retail, map_cents=_settings()["map_default_cents"])`. Best-effort: on any error, fall back to `max(retail, map)` (never crash a checkout). Keep `practitioner_price_for` calling it.

- [ ] **Step 4: Run → pass** (client-order tests now reflect the practitioner's price).
- [ ] **Step 5: Commit** — `feat(client): client page prices at the practitioner's set price`

---

### Task 3: settings API

**Files:** Modify `app.py`; Test `tests/test_practitioner_settings_routes.py`

- [ ] **Step 1: Failing test** — `GET /api/practitioner/settings` (authed) returns `{branding, pricing}`; `POST /api/practitioner/settings` writes branding + pricing for the signed-in practitioner. 401 without a practitioner session. A POSTed per-SKU price below MAP is rejected or clamped (validate via `practitioner_pricing`); markup % accepted.

- [ ] **Step 2: Run → fail.**

- [ ] **Step 3: Implement** — `GET/POST /api/practitioner/settings`, authed via `_practitioner_session_pid()`. GET → `practitioner_settings.get_settings`. POST → validate (markup numeric; per-SKU overrides each ≥ MAP via `resolve_selling_cents`/clamp; colors are hex; URLs are strings) then `set_branding` + `set_pricing`. Open the connection on `LOG_DB`. Import `practitioner_settings` locally per convention.

- [ ] **Step 4: Run → pass.**
- [ ] **Step 5: Commit** — `feat(practitioner-settings): GET/POST /api/practitioner/settings`

---

### Task 4: settings UI page

**Files:** Modify `app.py` (`/practitioner/settings` route); Create `static/practitioner-settings.html`

- [ ] **Step 1:** Add `GET /practitioner/settings` serving the page (mirror the other portal page serves). Build `static/practitioner-settings.html` (mirror portal styling): a **Pricing** section (a default markup % input **or** per-SKU dollar prices — show the computed companion figure via `markup_pct_for`/`price_for_markup`; warn/block below MAP $67) and a **Branding** section (practice name, contact, web link, logo URL, photo URL, two color pickers). Loads `GET /api/practitioner/settings`, saves via POST. Live preview of the two colors. **Image = URL field in v1** (note: an upload widget reusing `/clips/upload` is a later add).
- [ ] **Step 2:** Verify (static read + routes green).
- [ ] **Step 3:** Commit — `feat(practitioner-settings): portal settings page (pricing + branding)`

---

### Task 5: apply branding to the three pages

**Files:** Modify `static/practitioner-client.html`, `static/practitioner-dropship.html`, and the wholesale page; add a branding read to their catalog/portal-data endpoints

- [ ] **Step 1:** Expose the practitioner's branding to the pages: extend `GET /api/client/<code>/catalog` (client page) and the practitioner portal-data (drop-ship/wholesale) to include `branding`. Apply it:
  - **Client page:** photo + logo + practice name + contact details + web link + the 2 brand colors (header/accent). Full white-label.
  - **Drop-ship + wholesale pages:** logo + practice name + the 2 colors.
  - Always **fall back to RM default branding** if an asset/field is missing — never break the page.
- [ ] **Step 2:** Verify (static read; the client/dropship route tests stay green).
- [ ] **Step 3:** Commit — `feat(practitioner-settings): apply white-label branding to the 3 pages`

---

### Task 6: suite + doc

- [ ] **Step 1:** Run all practitioner-settings + client + dropship + pricing tests — green.
- [ ] **Step 2:** Create `docs/practitioner-settings.md`: the settings store (local sqlite), pricing (default markup % + per-SKU $ override, MAP-clamped), branding fields, how each page applies branding, image = URL in v1, and that the client page now prices at the practitioner's set price.
- [ ] **Step 3:** Commit.

---

## Self-review
- **Spec coverage:** §A.6 $/% pricing (T1-3); §C branding store + apply (T1,3,4,5); §D data (T1, local sqlite — no Supabase migration); §B branding on the 3 pages (T5); resolves §I-2 (markup + override) and §I-3 (image URL v1).
- **Deferred:** image **upload** widget (URL field in v1; reuse `/clips/upload` later); Plan 5 = the scoped chat.
- **Risk:** the only money-affecting change is Task 2 (client page now uses the practitioner's price) — guarded by MAP-clamp + a best-effort fallback to `max(retail, MAP)` so a settings error never breaks checkout. Branding is cosmetic with RM-default fallbacks. No live customer-checkout impact.
- **Type consistency:** `get_settings(cx,pid)`, `set_branding`/`set_pricing(cx,pid,dict)`, `price_cents_for(cx,pid,slug,*,retail_cents,map_cents)`, `/api/practitioner/settings`, `/practitioner/settings`.

## Next
Plan 5 — practitioner-scoped support chat (self-contained, no RM links; toggle on the client page, default available on the practitioner pages) — Glen-confirmed design.
