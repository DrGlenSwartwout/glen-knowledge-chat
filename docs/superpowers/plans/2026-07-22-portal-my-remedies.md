# Portal — My Remedies (Phase 2) Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add "My Remedies" to the client portal hub (Understand group): a page with two sections — (1) **Top recommended for you (ranked)**, the decision view that reuses the existing recommendations stack and gives each item an "add to my Oasis / order" hand-off; and (2) **What you take from other companies**, a client-maintained external stack (brand / product / reason / importance) that reuses `dashboard/supplement_reviews.py` for request-review + hosted-review, plus a conservative suggested-upgrade pointer where one genuinely helps. Ships dark behind `PORTAL_REMEDIES_ENABLED`.

**Architecture:** No new recommendation engine. Section 1 reads the SAME data the live `/api/portal/<token>/recommendations` endpoint already serves (`recommendation_events.product_sources` → `portal_recommendations.build_sections`, ranked, top-N). Section 2 EXTENDS the existing `supplement_reviews` store (adds `reason`, `importance`, and a `listed` pre-review state) rather than creating a second table, so a listed item can be promoted to a review request through the existing `requested → ai_draft → confirmed` pipeline and console approval flow. A new pure module `dashboard/remedy_upgrades.py` maps an external product to our equivalent formulation, returning `None` for well-chosen products (not a blanket upsell). The portal payload gets one `remedies` block; the panel renders both sections client-side.

**Tech Stack:** Python 3.9 + Flask (`app.py`, `dashboard/portal_view.py`, `dashboard/supplement_reviews.py`, new `dashboard/remedy_upgrades.py`, new `dashboard/remedies_block.py`), SQLite (`LOG_DB`), vanilla client JS/CSS in `static/client-portal.html`. Server changes get pytest; client changes get server-free Node parse/logic checks + a pre-flip headless render pass.

## Global Constraints

- **Flag `PORTAL_REMEDIES_ENABLED`, default OFF**, truthy set `("1","true","yes","on")` (mirror `_PORTAL_HUB_ENABLED` at `app.py:6110`).
- **Identity from the token, never the request body** — resolve via `_portal_record_for(cx, token)` on every `/api/portal/<token>/*` endpoint (see the recommendations endpoints at `app.py:20007`).
- **Reuse, do not fork, `supplement_reviews`** — the external list stores rows in the existing `supplement_reviews` table; `request review` promotes a `listed` row to `requested` via the existing pipeline. Never a second reviews store. Never-downgrade rules in `set_status` stay authoritative.
- **Suggested upgrade is conservative** — `remedy_upgrades.suggest_upgrade(...)` returns `None` unless there is a confident equivalent. A well-chosen product is confirmed as such. Not every item gets a swap.
- **Ranked section is read-through** — it must show the same ranking/dedup the recommendations endpoint shows; do not re-rank differently in the block vs. the endpoint.
- **Every block builder degrades to a safe empty on error** (mirror `_practitioner_finder_block` / `_supplement_reviews_block` in `dashboard/portal_view.py`). A remedies failure must never break the rest of the portal payload.
- Copy: no em dashes, no ALL CAPS. Theme-aware CSS vars only. Render-verify (not just parse) before flag flip.

---

### Task 1: Extend the `supplement_reviews` store — reason, importance, `listed` state

**Files:**
- Modify: `dashboard/supplement_reviews.py` (add columns via idempotent migration; add `add_listed`, `set_meta`, `remove`; extend `_row`)
- Test: `tests/test_supplement_reviews_external.py`

**Interfaces:**
- Produces (all in `dashboard/supplement_reviews.py`):
  - `add_listed(cx, email, product_name, product_brand="", reason="", importance=None, source="portal") -> dict` — creates a row in the new `listed` state (client added it to their stack, no review requested yet). Idempotent on `(email, product_key)`; if a row already exists (any status) returns it untouched. Returns `{"created": bool, "id": int|None, "status": str}`.
  - `set_meta(cx, email, product_key, reason=None, importance=None) -> dict` — updates the client-set `reason` / `importance` on an existing row (leaves status/review untouched). `importance` clamped to 1..10 or None.
  - `remove(cx, email, product_key) -> dict` — deletes a row ONLY if its status is `listed` or `requested` (never deletes an `ai_draft`/`confirmed` review, which represents Glen's work). Returns `{"removed": bool}`.
  - `_row` gains `reason` and `importance` keys.
- `_RANK` gains `"listed": -1` so `listed` sits below `requested` and the never-downgrade guard in `set_status` treats a `listed → requested` promotion as an upgrade.

- [ ] **Step 1: Write the failing test**
```python
# tests/test_supplement_reviews_external.py
import sqlite3
from dashboard import supplement_reviews as sr

def _cx():
    cx = sqlite3.connect(":memory:"); cx.row_factory = sqlite3.Row
    sr.init_table(cx); return cx

def test_add_listed_then_set_meta_and_promote():
    cx = _cx()
    r = sr.add_listed(cx, "a@b.com", "Magnesium Glycinate", "Acme", reason="sleep", importance=8)
    assert r["created"] is True and r["status"] == "listed"
    key = sr._key("Magnesium Glycinate", "Acme")
    sr.set_meta(cx, "a@b.com", key, reason="sleep + cramps", importance=9)
    row = [x for x in sr.list_for_email(cx, "a@b.com")][0]
    assert row["reason"] == "sleep + cramps" and row["importance"] == 9
    # request-review promotes listed -> requested via the existing pipeline
    assert sr.create_request(cx, "a@b.com", "Magnesium Glycinate", "Acme")["status"] == "requested"

def test_remove_only_before_review_exists():
    cx = _cx()
    sr.add_listed(cx, "a@b.com", "Fish Oil", "Acme")
    key = sr._key("Fish Oil", "Acme")
    assert sr.remove(cx, "a@b.com", key)["removed"] is True
    # a confirmed review is protected from client removal
    sr.create_request(cx, "a@b.com", "Vit D", "Acme")
    rid = sr.list_for_email(cx, "a@b.com")[0]["id"]
    sr.set_draft(cx, rid, "review text"); sr.set_status(cx, rid, "confirmed")
    assert sr.remove(cx, "a@b.com", sr._key("Vit D", "Acme"))["removed"] is False
```

- [ ] **Step 2: Run test to verify it fails** — `python3 -m pytest tests/test_supplement_reviews_external.py -v` → FAIL (`add_listed` missing).

- [ ] **Step 3: Implement the migration + functions.** In `init_table`, after the `CREATE TABLE`, add an idempotent column migration:
```python
    _cols = {r[1] for r in cx.execute("PRAGMA table_info(supplement_reviews)")}
    if "reason" not in _cols:
        cx.execute("ALTER TABLE supplement_reviews ADD COLUMN reason TEXT")
    if "importance" not in _cols:
        cx.execute("ALTER TABLE supplement_reviews ADD COLUMN importance INTEGER")
```
Add `"listed": -1` to `_RANK`. Implement `add_listed` (INSERT with `status='listed'`, `requested_at=NULL`, storing `reason`/`importance`; same `(email, product_key)` dedupe + IntegrityError race handling as `create_request`). Implement `set_meta` (UPDATE reason/importance, clamp importance to 1..10 or NULL, stamp `updated_at`). Implement `remove` (SELECT status; DELETE only when status in `("listed","requested")`). Extend `_row` and every `SELECT` column list to include `reason, importance`. Confirm `create_request` still promotes a `listed` row: it currently returns the existing row when one is present — change it so that when the existing row's status is `listed`, it UPDATEs status to `requested` + stamps `requested_at` and returns `{"created": False, "status": "requested"}`.

- [ ] **Step 4: Run tests** → PASS.

- [ ] **Step 5: Commit** — `feat: supplement_reviews external-list fields (reason/importance) + listed state`

---

### Task 2: `remedy_upgrades.py` — conservative external-product → our-equivalent mapping

**Files:**
- Create: `dashboard/remedy_upgrades.py`
- Test: `tests/test_remedy_upgrades.py`

**Interfaces:**
- Produces `dashboard/remedy_upgrades.py`:
  - `suggest_upgrade(product_name, product_brand="", *, catalog=None) -> dict | None` — returns `{"slug","name","url","reason"}` for our equivalent formulation when there is a confident, clinically-justified match, else `None`. `catalog` is injectable for tests (defaults to `products.load_products()`). Matching is by a curated `_UPGRADE_MAP` keyed on a normalized `(ingredient/category)` derived from the product name (e.g. "magnesium glycinate" -> our Neuro-Mag), NOT a fuzzy name match; an unmapped product returns `None`.
  - `_normalize(name) -> str` — lowercased, whitespace-collapsed category key (reuse the `_key`-style normalization from `supplement_reviews`).

- [ ] **Step 1: Write the failing test**
```python
# tests/test_remedy_upgrades.py
from dashboard import remedy_upgrades as ru

_CAT = {"neuro-mag": {"name": "Neuro-Mag", "url": "/p/neuro-mag"}}

def test_mapped_ingredient_returns_our_equivalent():
    up = ru.suggest_upgrade("Magnesium Glycinate", "Acme", catalog=_CAT)
    assert up and up["slug"] == "neuro-mag" and up["url"] == "/p/neuro-mag"
    assert up["reason"]                    # a non-empty clinical reason

def test_unmapped_product_returns_none():
    assert ru.suggest_upgrade("Organic Kale Powder", "Acme", catalog=_CAT) is None

def test_well_chosen_product_not_swapped():
    # a product we already consider optimal maps to None on purpose (clinical integrity)
    assert ru.suggest_upgrade("Neuro-Mag", "Remedy Match", catalog=_CAT) is None
```

- [ ] **Step 2: Run test → FAIL** (module missing).

- [ ] **Step 3: Implement `dashboard/remedy_upgrades.py`.** Define `_UPGRADE_MAP` as a small curated dict of `normalized-category -> {"slug","reason"}` (start with the handful Glen has confirmed, e.g. `"magnesium glycinate" -> {"slug":"neuro-mag","reason":"..."}`; leave a comment that Glen extends this map). `suggest_upgrade` normalizes the name, looks up the map, returns `None` on miss; on hit, resolves `name`/`url` from `catalog` (skip + return `None` if the slug is absent from the catalog so we never point at a dead product), and returns the dict with the curated `reason`. Products already equal to our own slug map to `None`.

- [ ] **Step 4: Run tests → PASS.**

- [ ] **Step 5: Commit** — `feat: remedy_upgrades conservative external-product to equivalent mapping`

---

### Task 3: `remedies` block into the portal payload (ranked + external)

**Files:**
- Create: `dashboard/remedies_block.py` (`build_block(cx, email, enabled) -> dict`)
- Modify: `app.py` (define `_PORTAL_REMEDIES_ENABLED`; pass to `get_portal_view`)
- Modify: `dashboard/portal_view.py` (`get_portal_view` param + `"remedies"` payload block)
- Test: `tests/test_remedies_block.py`

**Interfaces:**
- Produces `dashboard/remedies_block.py`:
  - `build_block(cx, email, enabled) -> dict`: `{"enabled": bool, "ranked": [ {product_key,name,url,source,reason} ], "external": [ {product_key,product_name,product_brand,reason,importance,status,review,upgrade} ]}`. `ranked` is the top-N from `recommendation_events.product_sources` → `portal_recommendations.build_sections(..., top_n=5)`, flattened + deduped by product_key. `external` is `supplement_reviews.list_for_email` rows enriched with `remedy_upgrades.suggest_upgrade`; `review` text is included ONLY when `status == "confirmed"` (mirror `_supplement_reviews_block`). Returns `{"enabled": False}` when the flag is off; degrades to `{"enabled": True, "ranked": [], "external": []}` on any internal error.

- [ ] **Step 1: Write the failing test**
```python
# tests/test_remedies_block.py
import sqlite3
from dashboard import remedies_block, supplement_reviews as sr

def _cx():
    cx = sqlite3.connect(":memory:"); cx.row_factory = sqlite3.Row
    sr.init_table(cx); return cx

def test_block_off_when_disabled():
    assert remedies_block.build_block(None, "a@b.com", False) == {"enabled": False}

def test_external_hides_unconfirmed_review_text():
    cx = _cx()
    sr.add_listed(cx, "a@b.com", "Fish Oil", "Acme", reason="heart", importance=6)
    rid = sr.list_for_email(cx, "a@b.com")[0]["id"]
    sr.create_request(cx, "a@b.com", "Fish Oil", "Acme")
    sr.set_draft(cx, rid, "SECRET draft not for client")
    blk = remedies_block.build_block(cx, "a@b.com", True)
    ext = blk["external"][0]
    assert ext["reason"] == "heart" and ext["importance"] == 6
    assert "review" not in ext or ext.get("review") in (None, "")   # ai_draft text stays hidden
```

- [ ] **Step 2: Run test → FAIL** (module missing).

- [ ] **Step 3: Implement `dashboard/remedies_block.py`.** `ranked`: import `recommendation_events`, `recommendation_prefs`, `portal_recommendations`, `products`; replicate the resolve+build used by `api_portal_recommendations` (`app.py:20007`) but flatten `sections` to a deduped top list keyed by product_key. Wrap the whole ranked build in try/except → `[]`. `external`: `sr.init_table`; if `sr.access_enabled` is False return external `[]`; map each `list_for_email` row to the external item, attach `upgrade = remedy_upgrades.suggest_upgrade(name, brand)`, include `review` text only for `confirmed`. Guard with try/except.

- [ ] **Step 4: Thread the flag + block.** In `app.py`, add `_PORTAL_REMEDIES_ENABLED` (same parse as `_PORTAL_HUB_ENABLED`) and pass `remedies_enabled=_PORTAL_REMEDIES_ENABLED` into the `get_portal_view(...)` call at `app.py:25347`. In `dashboard/portal_view.py`, add `remedies_enabled=False` to the `get_portal_view` keyword params and add `"remedies": _rb.build_block(cx, email, remedies_enabled)` to the returned dict (import `remedies_block as _rb` at module top like the other block imports).

- [ ] **Step 5: Run tests → PASS.**

- [ ] **Step 6: Commit** — `feat: PORTAL_REMEDIES_ENABLED flag + remedies portal block (ranked + external)`

---

### Task 4: Render the My Remedies tile + panel

**Files:** Modify `static/client-portal.html` (add tile to `buildHubHtml` Understand group; add `data-panel="remedies"` in the wrap block; add `buildRemediesHtml(v)`).

**Interfaces:** Consumes `v.remedies` (Task 3) and the existing `backToHub()` / panel-wrap conventions (see the Health Profile plan Task 2 for the identical tile+panel pattern, `buildHubHtml` at `static/client-portal.html:678`).

- [ ] **Step 1** — In `buildHubHtml`, insert into the **Understand** group, after `["current", ...]`, the tile `["remedies", "My Remedies", "Your ranked matches and the stack you take"]`. It is safe when `v.remedies` is absent (showTab falls back to hub).
- [ ] **Step 2** — Add a global `buildRemediesHtml(v)` that renders two `<section>`s:
  - "Top recommended for you" — iterate `v.remedies.ranked`, each row = name (linked to `url`), its `source`/`reason`, and an **"Add to my Oasis / order"** button `data-oasis-add="${esc(product_key)}"` (wired in Task 6).
  - "What you take from other companies" — iterate `v.remedies.external`, each row = brand + product, the client-set reason/importance, a status pill (`listed`/`in progress`/`reviewed`), a **Request a review** button (hidden once status is past `listed`), the review text when present, and an `upgrade` pointer when non-null ("A stronger option: {upgrade.name}"). Plus an "Add a product you take" form (brand, product, reason, importance 1-10).
  Escape every value with `esc()`.
- [ ] **Step 3** — In the wrap block (beside the other `_hub && ...` panel emissions), emit `${_hub && v.remedies && v.remedies.enabled ? \`<section data-panel="remedies" hidden>${back}${buildRemediesHtml(v)}</section>\` : ""}`.
- [ ] **Step 4** — Server-free Node check: `buildRemediesHtml` output for a sample `v` contains both section titles, a ranked row, an external row with its importance, and does NOT contain unconfirmed review text; both script blocks parse. Commit — `feat: My Remedies tile + two-section panel (ranked + external stack)`

---

### Task 5: External-list endpoints (add / update / remove / request-review)

**Files:** Modify `app.py` (four `POST /api/portal/<token>/remedies/*` routes); Test `tests/test_remedies_api.py`.

**Interfaces:** All token-authed via `_portal_record_for`, flag-gated (404 when `_PORTAL_REMEDIES_ENABLED` off), `_db_lock` on writes, and each returns the refreshed `remedies_block.build_block` so the client re-renders from one source.
- `POST /api/portal/<token>/remedies/add` body `{product_name, product_brand?, reason?, importance?}` → `sr.add_listed(...)`.
- `POST /api/portal/<token>/remedies/meta` body `{product_key, reason?, importance?}` → `sr.set_meta(...)`.
- `POST /api/portal/<token>/remedies/remove` body `{product_key}` → `sr.remove(...)`.
- `POST /api/portal/<token>/remedies/request-review` body `{product_name, product_brand?}` → `sr.create_request(...)` (promotes `listed` → `requested`, feeding the existing console approval + analyzer pipeline).

- [ ] **Step 1: Failing test** — with the flag patched on: POST add → the item appears in `build_block(...)["external"]` as `listed`; POST meta updates importance; POST request-review flips status to `requested`; POST remove on a still-`listed` item drops it; a request with a bad token → 404.
- [ ] **Step 2: Run → FAIL.**
- [ ] **Step 3: Implement the four endpoints** mirroring the recommendations POST routes at `app.py:20036-20105` (flag gate → `_db_lock, sqlite3.connect(LOG_DB)` → `_cp.init_client_portal_table` + `sr.init_table` → `_portal_record_for` → email → mutate → return refreshed block). Gate all four behind `if not _PORTAL_REMEDIES_ENABLED: return jsonify({"error":"disabled"}), 404`.
- [ ] **Step 4: Run tests → PASS.**
- [ ] **Step 5: Commit** — `feat: My Remedies external-list endpoints (add/meta/remove/request-review)`

---

### Task 6: "Add to my Oasis / order" hand-off action

**Files:** Modify `static/client-portal.html` (wire the ranked-row button); Modify `app.py` (`POST /api/portal/<token>/remedies/to-oasis`); Test `tests/test_remedies_to_oasis.py`.

**Interfaces:** `POST /api/portal/<token>/remedies/to-oasis` body `{slug}` → adds the slug to the client's wishlist via `wishlist.toggle(cx, wishlist.resolve_owner(email, None), slug)` (ensuring present, not toggling off) and records a `record_click(email, slug, source="my-remedies")` engagement. Returns `{"ok": True}`. This is the explicit link into My Healing Oasis › Replenish/Build Out (the receiving side is built in that plan; the wishlist row is the shared handoff surface).

- [ ] **Step 1: Failing test** — POST to-oasis with a slug → `wishlist.slugs_for(cx, "email:a@b.com")` contains the slug; a second POST leaves it present (idempotent, never toggles off).
- [ ] **Step 2: Run → FAIL.**
- [ ] **Step 3: Implement** the endpoint (flag gate, token identity, `_db_lock`; call `wishlist.init_wishlist_table` then INSERT-OR-IGNORE semantics — use `slugs_for` to check presence and only `toggle` when absent so it never removes). Wire the Task 4 `data-oasis-add` button to POST it and confirm with a small inline "Added to your Oasis" state.
- [ ] **Step 4: Run tests → PASS.**
- [ ] **Step 5: Commit** — `feat: My Remedies add-to-Oasis handoff (wishlist + engagement)`

---

### Task 7: Rollout note + pre-flip verification checklist

- [ ] Write `docs/superpowers/plans/2026-07-22-portal-my-remedies-rollout.md`: the flag (`PORTAL_REMEDIES_ENABLED`, off), its independence from `SUPPLEMENT_REVIEW_ENABLED` (the request-review sub-flow still honors the supplement-review pipeline + per-client access), and a headless render-verify checklist — tile appears in Understand; ranked section shows the same items as `/api/portal/<token>/recommendations`; add/edit/remove an external item; request-review flips to "in progress" and shows in the console queue; a confirmed review becomes visible; an upgrade pointer renders only where mapped; add-to-Oasis lands a wishlist row. Note the merge = self-deploy + brief illtowell 502 (dark, low-risk). Commit.

## Self-Review

- **Spec coverage:** ranked decision view reusing formulation-matches/recommends/PRL/program via the recommendations stack (Tasks 3-4); add-to-Oasis handoff (Task 6); external client-maintained list with brand/product/reason/importance (Tasks 1,4,5); request-review reusing `supplement_reviews` pipeline (Tasks 1,5); review-access link on confirm (Task 3 `review` field); conservative suggested-upgrade, not-every-item (Task 2). Flag-gated dark (Task 3).
- **Placeholder scan:** `_UPGRADE_MAP` ships with real seed entries + an explicit "Glen extends this" comment, not a TODO; every endpoint shows its reuse target by file:line.
- **Type consistency:** `add_listed / set_meta / remove / create_request` (supplement_reviews), `suggest_upgrade` (remedy_upgrades), `build_block` (remedies_block), the `remedies` payload key, and `data-panel="remedies"` are consistent server↔client. `product_key` is the shared identity across store, block, and endpoints.

## Design decisions (settled)
- **Extend `supplement_reviews`, do not fork it** — one store, a new `listed` pre-review state, so a listed item promotes cleanly into the existing console/analyzer pipeline.
- **Ranked section is a read-through of the live recommendations engine** — no parallel ranking, so My Remedies and the recommendations endpoint never disagree.
- **Upgrade suggestions are curated and conservative** — `None` by default; clinical integrity over upsell.
