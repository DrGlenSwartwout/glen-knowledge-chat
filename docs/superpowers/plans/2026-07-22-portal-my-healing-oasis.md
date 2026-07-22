# Portal — My Healing Oasis (Phase 2) Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add "My Healing Oasis" to the client portal hub (Act group): the unified "your stuff" page with two modes — **Replenish** (consumables the client owns, with reorder actions and running-low hints, absorbing today's "Order your remedies" / reorder / Life Stress Essences cards + wishlist) and **Build Out** (the client's tools inventory — items bought from us AND items they own elsewhere — plus a clinically-ordered roadmap of recommended additions that leads with our highest-impact tools: Harmony, Water Ionizer, Kloud). Ships dark behind `PORTAL_OASIS_ENABLED`.

**Architecture:** Replenish is a projection over existing data — the client's owned consumables come from their order line items (`orders`), reorder reuses the existing portal-reorder checkout, and the wishlist (`wishlist.py`) supplies wanted items. Build Out adds one new client-maintained store `dashboard/owned_tools.py` (tools they own from other sources) and one new pure roadmap module `dashboard/oasis_roadmap.py` that orders recommended tool additions from the client's analysis (terrain phase first), leads with the three hero tools, and excludes anything the client already owns (from us or externally). The portal payload gets one `oasis` block; the panel renders the two modes client-side. My Healing Oasis is downstream of My Analysis and receives the "add to my Oasis" handoff from My Remedies via the shared wishlist.

**Tech Stack:** Python 3.9 + Flask (`app.py`, `dashboard/portal_view.py`, new `dashboard/owned_tools.py`, new `dashboard/oasis_roadmap.py`, new `dashboard/oasis_block.py`), SQLite (`LOG_DB`), vanilla client JS/CSS in `static/client-portal.html`. Server changes get pytest; client changes get server-free Node parse/logic checks + a pre-flip headless render pass.

## Global Constraints

- **Flag `PORTAL_OASIS_ENABLED`, default OFF**, truthy set `("1","true","yes","on")` (mirror `_PORTAL_HUB_ENABLED` at `app.py:6110`).
- **Identity from the token, never the request body** — resolve via `_portal_record_for(cx, token)` on every `/api/portal/<token>/*` endpoint (see recommendations endpoints at `app.py:20007`).
- **The roadmap is driven by the client's analysis, not a generic catalog** — ordering is terrain-phase-first, then low-cost high-leverage, then larger investments; the three hero tools (Harmony, Water Ionizer, Kloud) lead as near-universal recommendations, ABOVE terrain-specific gap items.
- **Generous ownership check** — the roadmap must exclude tools the client already owns from ANY source (bought from us OR self-reported external), so it never recommends a device they have and sequences around the gap.
- **Reorder reuses the existing checkout** — no new payment path; Replenish links into the portal-reorder / checkout flow already in `app.py`.
- **Every block builder degrades to a safe empty on error** (mirror `_practitioner_finder_block` in `dashboard/portal_view.py`). An oasis failure must never break the rest of the portal payload.
- **Hero tool slugs are named constants**, defined once in `oasis_roadmap.py` and reused, never string-duplicated.
- Copy: no em dashes, no ALL CAPS. Theme-aware CSS vars only. Render-verify (not just parse) before flag flip.

---

### Task 1: `owned_tools.py` — client-maintained external tools inventory

**Files:**
- Create: `dashboard/owned_tools.py`
- Test: `tests/test_owned_tools.py`

**Interfaces:**
- Produces `dashboard/owned_tools.py` (pure + LOG_DB-only, unit-testable like `wishlist.py`):
  - `init_table(cx)` → `owned_tools(id INTEGER PK, email TEXT NOT NULL, name TEXT NOT NULL, brand TEXT, slug TEXT, source TEXT, added_at TEXT, UNIQUE(email, tool_key))`, index `(email)`. `slug` is set when the item maps to one of our products (so the roadmap's ownership check can match it); NULL for a purely external tool. `tool_key` is a normalized `name|brand` dedupe key.
  - `add(cx, email, name, brand="", slug=None, source="external") -> dict` — INSERT, idempotent on `(email, tool_key)`; returns `{"created": bool, "id": int|None}`.
  - `list_for(cx, email) -> list[dict]` — `[{id,name,brand,slug,source,added_at}]`.
  - `remove(cx, email, tool_id) -> dict` — DELETE by id scoped to email; `{"removed": bool}`.
  - `owned_slugs(cx, email) -> set[str]` — the non-null `slug`s the client has self-reported owning.
  - `_key(name, brand) -> str` — lowercased, whitespace-collapsed `name|brand` (reuse the pattern from `supplement_reviews._key`).

- [ ] **Step 1: Write the failing test**
```python
# tests/test_owned_tools.py
import sqlite3
from dashboard import owned_tools as ot

def _cx():
    cx = sqlite3.connect(":memory:"); cx.row_factory = sqlite3.Row
    ot.init_table(cx); return cx

def test_add_dedupes_and_lists():
    cx = _cx()
    assert ot.add(cx, "a@b.com", "Red Light Panel", "Joovv")["created"] is True
    assert ot.add(cx, "a@b.com", "red light panel", "Joovv")["created"] is False  # dedupe
    rows = ot.list_for(cx, "a@b.com")
    assert len(rows) == 1 and rows[0]["brand"] == "Joovv"

def test_owned_slugs_only_mapped():
    cx = _cx()
    ot.add(cx, "a@b.com", "Water Ionizer", "OtherCo", slug="water-ionizer")
    ot.add(cx, "a@b.com", "Random Gadget", "OtherCo")   # no slug
    assert ot.owned_slugs(cx, "a@b.com") == {"water-ionizer"}

def test_remove_scoped_to_email():
    cx = _cx()
    tid = ot.add(cx, "a@b.com", "Kloud", "X", slug="kloud")["id"]
    assert ot.remove(cx, "other@b.com", tid)["removed"] is False   # not your row
    assert ot.remove(cx, "a@b.com", tid)["removed"] is True
```

- [ ] **Step 2: Run test → FAIL** (module missing).

- [ ] **Step 3: Implement `dashboard/owned_tools.py`** per the interface (table + `_key` + add/list/remove/owned_slugs, `INSERT OR IGNORE` for dedupe, `datetime('now')` default for `added_at`).

- [ ] **Step 4: Run tests → PASS.**

- [ ] **Step 5: Commit** — `feat: owned_tools client-maintained external tools inventory`

---

### Task 2: `oasis_roadmap.py` — analysis-ordered recommended additions, hero-tool-led

**Files:**
- Create: `dashboard/oasis_roadmap.py`
- Test: `tests/test_oasis_roadmap.py`

**Interfaces:**
- Produces `dashboard/oasis_roadmap.py`:
  - `HERO_TOOLS: list[dict]` — the three near-universal tools in fixed lead order: `[{"slug":"harmony",...},{"slug":"water-ionizer",...},{"slug":"kloud",...}]` (name + one-line why + a stable `tier="hero"`).
  - `TERRAIN_TOOLS: dict[str, list[dict]]` — terrain-phase → ordered gap tools (`tier="terrain"`), keyed by the 5 R phases (`energize`/`rejuvenate`/`regenerate`/`cleanse`/`balance`) used elsewhere in the app.
  - `build_roadmap(owned_slugs, terrain_phase=None) -> list[dict]` — returns hero tools NOT in `owned_slugs` first (in fixed order), then the `terrain_phase`'s gap tools not owned, then a general low-cost-high-leverage tail; every returned item carries `{slug,name,why,tier}`. Excludes any slug in `owned_slugs`. Pure + injectable (no DB).

- [ ] **Step 1: Write the failing test**
```python
# tests/test_oasis_roadmap.py
from dashboard import oasis_roadmap as orm

def test_hero_tools_lead_and_ownership_excludes():
    rm = orm.build_roadmap(owned_slugs=set(), terrain_phase="cleanse")
    slugs = [x["slug"] for x in rm]
    assert slugs[:3] == ["harmony", "water-ionizer", "kloud"]     # hero order, above terrain items
    assert all(x["tier"] in ("hero", "terrain", "general") for x in rm)

def test_owned_hero_is_dropped_but_order_preserved():
    rm = orm.build_roadmap(owned_slugs={"water-ionizer"}, terrain_phase=None)
    slugs = [x["slug"] for x in rm]
    assert "water-ionizer" not in slugs
    assert slugs[:2] == ["harmony", "kloud"]                       # remaining heroes keep order

def test_terrain_phase_gap_items_follow_heroes():
    rm = orm.build_roadmap(owned_slugs={"harmony","water-ionizer","kloud"}, terrain_phase="cleanse")
    assert rm and all(x["tier"] != "hero" for x in rm)             # heroes owned -> only gap/general
```

- [ ] **Step 2: Run test → FAIL** (module missing).

- [ ] **Step 3: Implement `dashboard/oasis_roadmap.py`.** Define `HERO_TOOLS` and `TERRAIN_TOOLS` (seed each phase with the tools Glen has confirmed; a comment marks it as his to extend). `build_roadmap` filters heroes by ownership (preserving fixed order), appends the terrain phase's not-owned gap tools when `terrain_phase` is a known key, then a small general tail (not-owned), deduping by slug across the three tiers.

- [ ] **Step 4: Run tests → PASS.**

- [ ] **Step 5: Commit** — `feat: oasis_roadmap hero-led, analysis-ordered tool recommendations`

---

### Task 3: Replenish projection — owned consumables + running-low hint

**Files:**
- Create: `dashboard/oasis_replenish.py`
- Test: `tests/test_oasis_replenish.py`

**Interfaces:**
- Produces `dashboard/oasis_replenish.py`:
  - `replenish_items(cx, email, *, catalog=None, today=None) -> list[dict]` — projects the client's owned consumables from their order line items: for each consumable product ever ordered, `{slug, name, url, last_ordered, times_ordered, running_low}`. `running_low` is a lightweight heuristic: `True` when `days_since(last_ordered) >= _TYPICAL_BOTTLE_DAYS` (default 30). Consumables only — exclude `info_only`, services, and device/tool product types via the catalog's product type (mirror the "services/info-only/shipping/adjustments count 0" exclusion already used in `app.py`). `catalog`/`today` injectable for tests.
  - `_is_consumable(product) -> bool` — True unless product type is device/tool/service/info-only.

- [ ] **Step 1: Write the failing test**
```python
# tests/test_oasis_replenish.py
import sqlite3
from dashboard import oasis_replenish as rep, orders as _o

_CAT = {
  "neuro-mag": {"name":"Neuro-Mag","url":"/p/neuro-mag","type":"formulation"},
  "water-ionizer": {"name":"Water Ionizer","url":"/p/wi","type":"device"},
}

def _cx():
    cx = sqlite3.connect(":memory:"); cx.row_factory = sqlite3.Row
    _o.init_orders_tables(cx); return cx      # adjust to the real init name in dashboard/orders.py

def test_only_consumables_with_running_low():
    cx = _cx()
    # seed one order with a consumable + a device line (use the real orders insert helper)
    _o.record_order(cx, email="a@b.com", items=[{"slug":"neuro-mag","qty":1},
                                                {"slug":"water-ionizer","qty":1}],
                    created_at="2026-05-01")
    items = rep.replenish_items(cx, "a@b.com", catalog=_CAT, today="2026-07-01")
    slugs = {i["slug"] for i in items}
    assert slugs == {"neuro-mag"}                 # device excluded
    assert items[0]["running_low"] is True        # >30 days since last order
```
(Adjust `orders` init/insert/list calls to the real signatures in `dashboard/orders.py`; the assertion logic is what matters.)

- [ ] **Step 2: Run test → FAIL** (module missing).

- [ ] **Step 3: Implement `dashboard/oasis_replenish.py`.** Read the client's orders via `dashboard.orders.list_orders_by_email(cx, email, limit=200)` (as `_orders_block` does at `dashboard/portal_view.py:62`), expand each order's line items (from `items_json`; see the invoice line-item handling), keep the latest order date per slug + a count, resolve name/url/type from the catalog (`products.load_products()` when `catalog is None`), drop non-consumables via `_is_consumable`, and compute `running_low`. Sort running-low first, then most-recently-ordered.

- [ ] **Step 4: Run tests → PASS.**

- [ ] **Step 5: Commit** — `feat: oasis_replenish owned-consumables projection with running-low hint`

---

### Task 4: `oasis` block into the portal payload

**Files:**
- Create: `dashboard/oasis_block.py` (`build_block(cx, email, enabled, terrain_phase=None) -> dict`)
- Modify: `app.py` (define `_PORTAL_OASIS_ENABLED`; resolve the client's terrain phase; pass both to `get_portal_view`)
- Modify: `dashboard/portal_view.py` (`get_portal_view` param + `"oasis"` payload block)
- Test: `tests/test_oasis_block.py`

**Interfaces:**
- Produces `dashboard/oasis_block.py`:
  - `build_block(cx, email, enabled, terrain_phase=None) -> dict`: `{"enabled": bool, "replenish": [...], "build_out": {"owned_from_us": [...], "owned_external": [...], "roadmap": [...]}}`. `replenish` = `oasis_replenish.replenish_items`. `owned_from_us` = device/tool products from the client's order history. `owned_external` = `owned_tools.list_for`. `roadmap` = `oasis_roadmap.build_roadmap(owned_slugs, terrain_phase)` where `owned_slugs` = (our device slugs from order history) ∪ `owned_tools.owned_slugs` ∪ current wishlist slugs is NOT included (wanted != owned). Returns `{"enabled": False}` when the flag is off; degrades to empty structures on any error.

- [ ] **Step 1: Write the failing test**
```python
# tests/test_oasis_block.py
import sqlite3
from dashboard import oasis_block, owned_tools as ot

def _cx():
    cx = sqlite3.connect(":memory:"); cx.row_factory = sqlite3.Row
    ot.init_table(cx); return cx

def test_block_off_when_disabled():
    assert oasis_block.build_block(None, "a@b.com", False) == {"enabled": False}

def test_owned_external_tool_excluded_from_roadmap():
    cx = _cx()
    ot.add(cx, "a@b.com", "Water Ionizer", "OtherCo", slug="water-ionizer")
    blk = oasis_block.build_block(cx, "a@b.com", True, terrain_phase="cleanse")
    road_slugs = {r["slug"] for r in blk["build_out"]["roadmap"]}
    assert "water-ionizer" not in road_slugs                       # owned externally -> not recommended
    assert {"name","brand"} <= set(blk["build_out"]["owned_external"][0].keys())
```

- [ ] **Step 2: Run test → FAIL** (module missing).

- [ ] **Step 3: Implement `dashboard/oasis_block.py`.** Compose the three sub-modules; build `owned_slugs` from order-history device slugs + `owned_tools.owned_slugs`; guard every sub-call with try/except so one failing source degrades that field to `[]`, not the whole block.

- [ ] **Step 4: Thread the flag + block.** In `app.py` add `_PORTAL_OASIS_ENABLED` (same parse as `_PORTAL_HUB_ENABLED`). Resolve the client's `terrain_phase` from their current biofield/analysis (reuse whatever the portal already knows — the biofield block's phase; if not readily available, pass `None` and let the roadmap fall back to hero + general). Pass `oasis_enabled=...` and `terrain_phase=...` into `get_portal_view`. In `dashboard/portal_view.py` add the params and `"oasis": _ob.build_block(cx, email, oasis_enabled, terrain_phase)` to the returned dict.

- [ ] **Step 5: Run tests → PASS.**

- [ ] **Step 6: Commit** — `feat: PORTAL_OASIS_ENABLED flag + oasis portal block (replenish + build-out + roadmap)`

---

### Task 5: Render the My Healing Oasis tile + two-mode panel

**Files:** Modify `static/client-portal.html` (add tile to `buildHubHtml` **Act** group; add `data-panel="oasis"`; add `buildOasisHtml(v)` with Replenish / Build Out sub-tabs).

**Interfaces:** Consumes `v.oasis` (Task 4) and the existing `backToHub()` / panel-wrap conventions (same tile+panel pattern as the Health Profile plan Task 2; `buildHubHtml` at `static/client-portal.html:678`).

- [ ] **Step 1** — In `buildHubHtml`, insert into the **Act** group, before `["finder", ...]`, the tile `["oasis", "My Healing Oasis", "Your remedies to reorder and tools to build out"]`.
- [ ] **Step 2** — Add a global `buildOasisHtml(v)` with two sub-sections (a simple in-panel toggle, not new panels):
  - **Replenish** — iterate `v.oasis.replenish`; each row = name (linked to `url`), a "Running low" pill when `running_low`, last-ordered date, and a **Reorder** button that routes into the existing portal-reorder/checkout flow for that slug.
  - **Build Out** — three blocks: "Tools you own with us" (`owned_from_us`), "Tools you own elsewhere" (`owned_external`, each with a Remove control) + an "Add a tool you own" form (name, brand), and "Recommended to complete your Oasis" (`roadmap`, hero tier visually emphasized above terrain/general, each with an "Add to wishlist" action).
  Escape every value with `esc()`.
- [ ] **Step 3** — In the wrap block, emit `${_hub && v.oasis && v.oasis.enabled ? \`<section data-panel="oasis" hidden>${back}${buildOasisHtml(v)}</section>\` : ""}`.
- [ ] **Step 4** — Server-free Node check: `buildOasisHtml` for a sample `v` renders both modes, the hero tools appear above terrain items in the roadmap, a running-low pill shows, and both script blocks parse. Commit — `feat: My Healing Oasis tile + Replenish/Build-Out panel`

---

### Task 6: Build-Out endpoints (add / remove external tool; add roadmap item to wishlist)

**Files:** Modify `app.py` (`POST /api/portal/<token>/oasis/tool/add`, `.../tool/remove`, `.../roadmap/want`); Test `tests/test_oasis_api.py`.

**Interfaces:** All token-authed via `_portal_record_for`, flag-gated (404 when `_PORTAL_OASIS_ENABLED` off), `_db_lock` on writes, each returns the refreshed `oasis_block.build_block`.
- `POST /api/portal/<token>/oasis/tool/add` body `{name, brand?, slug?}` → `owned_tools.add(...)`.
- `POST /api/portal/<token>/oasis/tool/remove` body `{tool_id}` → `owned_tools.remove(...)`.
- `POST /api/portal/<token>/oasis/roadmap/want` body `{slug}` → adds the slug to the wishlist (`wishlist`, ensure-present, never toggle-off) — the same shared handoff surface My Remedies uses.

- [ ] **Step 1: Failing test** — with the flag patched on: POST tool/add → the tool appears in `build_block(...)["build_out"]["owned_external"]` AND, if it carried a slug matching a roadmap hero, that hero disappears from `roadmap`; POST tool/remove drops it; POST roadmap/want lands a wishlist row; bad token → 404.
- [ ] **Step 2: Run → FAIL.**
- [ ] **Step 3: Implement the three endpoints** mirroring the recommendations POST routes at `app.py:20036-20105` (flag gate → `_db_lock, sqlite3.connect(LOG_DB)` → inits → `_portal_record_for` → email → mutate → refreshed block). For roadmap/want reuse the ensure-present wishlist logic from the My Remedies plan Task 6.
- [ ] **Step 4: Run tests → PASS.**
- [ ] **Step 5: Commit** — `feat: My Healing Oasis build-out endpoints (add/remove tool, want roadmap item)`

---

### Task 7: Reorder hand-off + receive the "add to my Oasis" wishlist items

**Files:** Modify `static/client-portal.html` (Replenish "Reorder" button → existing portal-reorder/checkout; surface wishlist-added items from My Remedies inside Build Out).

**Interfaces:** Consumes the shared wishlist (populated by My Remedies Task 6 `/remedies/to-oasis` and this plan's `/oasis/roadmap/want`). No new endpoint — Build Out's "Recommended / wanted" area reads `v.oasis` which already reflects wishlist state, and Reorder routes to the existing checkout with the slug.

- [ ] **Step 1** — Wire the Replenish "Reorder" button to the existing portal reorder/checkout entry (confirm the exact route from `app.py` — the `portal-reorder`/`reorder` checkout kinds at `app.py:10868`); pass the slug + token. No server change if the existing checkout accepts a slug + token; if it needs a cart shape, build the minimal cart client-side as the current reorder cards do.
- [ ] **Step 2** — In Build Out, show a "Wanted / added from My Remedies" list sourced from the block's wishlist reflection so the My Remedies → Oasis handoff is visibly closed. Server-free Node parse + logic check. Commit — `feat: Oasis reorder handoff + surface wanted items from My Remedies`

---

### Task 8: Rollout note + pre-flip verification checklist

- [ ] Write `docs/superpowers/plans/2026-07-22-portal-my-healing-oasis-rollout.md`: the flag (`PORTAL_OASIS_ENABLED`, off), a headless render-verify checklist — tile appears in Act; Replenish lists owned consumables with a running-low pill and Reorder reaches checkout; Build Out lists owned-from-us + owned-external, add/remove an external tool, and the roadmap leads with Harmony/Water Ionizer/Kloud and drops any owned tool; add-to-wishlist from a roadmap item persists; an "add to my Oasis" item from My Remedies shows here. Note the merge = self-deploy + brief illtowell 502 (dark, low-risk). Commit.

## Self-Review

- **Spec coverage:** Replenish = owned consumables + reorder + running-low, absorbing order/reorder/essences/wishlist (Tasks 3,5,7); Build Out = owned-from-us + client-maintained external tools (Tasks 1,5,6) + analysis-ordered roadmap leading with the three hero tools and excluding owned (Tasks 2,4); downstream-of-analysis via `terrain_phase` (Task 4); handoff from My Remedies via shared wishlist (Task 7). Flag-gated dark (Task 4).
- **Placeholder scan:** `HERO_TOOLS`/`TERRAIN_TOOLS` ship with real seed entries + an explicit "Glen extends this" comment; `running_low` is a defined heuristic with a named constant, not a TODO; the one genuinely uncertain reuse (exact reorder-checkout route) is called out in Task 7 Step 1 with the file:line to confirm, not hand-waved.
- **Type consistency:** `add/list_for/remove/owned_slugs` (owned_tools), `HERO_TOOLS`/`build_roadmap` (oasis_roadmap), `replenish_items` (oasis_replenish), `build_block` (oasis_block), the `oasis` payload key, and `data-panel="oasis"` are consistent server↔client. `owned_slugs` is the shared exclusion identity between ownership and the roadmap.

## Design decisions (settled)
- **Replenish is a projection, not a new store** — owned consumables come from order history; only Build Out's external tools need a new client-maintained table.
- **Ownership is generous and cross-source** — the roadmap excludes tools owned from us OR elsewhere, so it never double-recommends.
- **Hero tools lead, analysis orders the rest** — Harmony / Water Ionizer / Kloud first (near-universal), then terrain-phase gaps, then general, all driven by the client's analysis.
- **One shared wishlist is the My Remedies ↔ Oasis handoff surface** — no bespoke transfer table.
