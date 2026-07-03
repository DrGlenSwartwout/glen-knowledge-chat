# Purchase-History Ingestion (FMP + GrooveKart) → Repertoire Seed — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax.

**Goal:** Backfill each client's real product purchase history from FileMaker (FMP) and GrooveKart into one slug-keyed `purchase_history` table, so a member's repertoire seeds from what they've *actually* bought — not just what the app happened to capture in `orders`.

**Architecture:** A new email-keyed `purchase_history` table (separate from the live `orders` board). Two derivation sources write into it: (1) **FMP** — map the stable `id_fk_product` integer to a catalog slug via a one-time human-reviewed map, then flatten `fmp_invoice_items ⨝ invoices ⨝ clients`; buildable now (data is already local). (2) **GrooveKart** — pull the past year of orders via its WebApi (full line items + `product_reference` SKU + email), map to slugs; **BLOCKED until Glen generates a WebApi key**. The repertoire seeder (already shipped) is extended to union `purchase_history` with `orders` within the 1-year window at conversion.

**Tech Stack:** Python 3, Flask (`app.py`), SQLite (`chat_log.db`), pure `dashboard/*.py` modules, pytest. GrooveKart WebApi = PrestaShop-derived REST + Basic Auth.

## Global Constraints

- **DB:** SQLite `chat_log.db`. Writes `with _db_lock, sqlite3.connect(LOG_DB) as cx:`; new pure modules take `cx` (house style: `dashboard/biofield_store.py`).
- **Emails** lowercased (`(email or "").strip().lower()`). Only rows with a real email are usable for seeding (repertoire is email-keyed).
- **Window:** repertoire seeding uses a **1-year (365-day) lookback** (Glen's decision) — legacy one-off purchases older than a year do NOT seed.
- **Separate table:** `purchase_history` is distinct from `orders` — never write legacy history into the live Orders board.
- **Idempotent rebuilds:** each source's derivation is a delete-that-source-then-insert (or `INSERT OR IGNORE` on a stable unique key), so a re-push/re-pull can't duplicate rows.
- **Mapping is human-reviewed** and drives PRICING ($50 reorders) → a wrong slug = a wrong price. Auto-map what's unambiguous; a human confirms the rest; hard-exclude non-products.
- **FMP coverage is 2024-03 onward** (the "newapp" export slice); pre-2024 history is not available. Not a bug — a documented limit.
- **Inert until `REPERTOIRE_ENABLED`:** populating `purchase_history` is harmless on its own; it only affects pricing once the seeder (Task 4) reads it under the existing `REPERTOIRE_ENABLED` flag.
- **Test runs:** `doppler run -p remedy-match -c dev -- python3 -m pytest tests/<file> -v`. Pure-module tests (Task 1) need no doppler. Judge regressions by isolation/before-after diff, never raw suite counts.

## Existing assets to REUSE (do not rebuild)

- `dashboard/fmp_orders.py` — `fmp_*` schema, CSV build, and `client_order_history()` (fmp_orders.py:125) which already joins `invoices ⨝ items ⨝ clients` by email — mirror its join.
- `dashboard/product_sales.py` — `slug_map_from_products_json()` (product_sales.py:27, the `fmp_id → slug` map) and `aggregate_rows()` (product_sales.py:71, modal description per id).
- `dashboard/practitioner_portal.py:276` `name_to_slug()` and `data/product-aliases.json` (108 curated aliases) — for the name-drift residue.
- `app.py:5285` `_resolve_buy_slug(name)` — fuzzy product-name → slug (reuse for GrooveKart line names).
- `data/products.json` (329 slugs; 120 carry `fmp_id`).
- FMP ingest route `/api/console/fmp-orders-ingest` (app.py:28599) and sales import `/api/console/sales/import` (app.py:28950).
- The repertoire seeder from the membership branch: `_order_slugs_since(cx, email, window_days)` and `_window_days_for_term` in `app.py`, seeding via `repertoire.seed_from_history`.

---

## File Structure

**Slice A — table + reviewed map (buildable now):**
- Create `dashboard/purchase_history.py` — table + add + `slugs_since`. Pure.
- Create `tests/test_purchase_history.py`.
- Create `scripts/build_fmp_slug_map.py` — emits `data/fmp_slug_map.json` (resolved + review + exclude). Human fills the review gaps.

**Slice B — FMP derivation (buildable now):**
- Create `dashboard/fmp_history.py` — `rebuild_from_fmp(cx, slug_map)`.
- Modify `app.py` — call `fmp_history.rebuild_from_fmp` at the end of the FMP ingest route so a re-push refreshes history.
- Create `tests/test_fmp_history.py`.

**Slice C — seeder union (buildable now):**
- Modify `app.py` — extend the conversion seed to union `orders` + `purchase_history` within the window.
- Modify `tests/test_repertoire_wiring.py` (or new `tests/test_history_seed.py`).

**Slice D — GrooveKart (BLOCKED on API key):**
- Create `dashboard/gk_client.py` — WebApi client (Basic Auth, orders list + pagination + date filter).
- Create `dashboard/gk_history.py` — `rebuild_from_gk(cx, since_iso, slug_resolver)`.
- Modify `app.py` — a console route `/api/console/gk-history-rebuild`.
- Create `tests/test_gk_client.py`, `tests/test_gk_history.py`.

---

## Task 1: `purchase_history` store module

**Files:** Create `dashboard/purchase_history.py`; Test `tests/test_purchase_history.py`.

**Interfaces (Produces):**
- `init_purchase_history_table(cx) -> None`
- `replace_source(cx, source, rows) -> int` — idempotently replaces ALL rows for `source` (delete where source=? then insert); `rows` = iterable of `(email, slug, purchased_at_iso, source_ref)`. Returns inserted count. Lowercases email/slug.
- `slugs_since(cx, email, window_days) -> set[str]` — distinct slugs for the email with `purchased_at >= now-window`.

- [ ] **Step 1: Write the failing test**

```python
# tests/test_purchase_history.py
import sqlite3
from datetime import datetime, timedelta, timezone
from dashboard import purchase_history as ph

def _cx():
    cx = sqlite3.connect(":memory:"); ph.init_purchase_history_table(cx); return cx

def _iso(days_ago):
    return (datetime.now(timezone.utc) - timedelta(days=days_ago)).isoformat()

def test_replace_source_is_idempotent_and_scoped():
    cx = _cx()
    n1 = ph.replace_source(cx, "fmp", [("A@x.com","neuro-mag",_iso(10),"inv1"),
                                        ("a@x.com","neuro-mag",_iso(10),"inv1b")])
    assert n1 == 2
    # re-run replaces, does not accumulate
    n2 = ph.replace_source(cx, "fmp", [("a@x.com","terrain-restore",_iso(5),"inv2")])
    assert n2 == 1
    assert ph.slugs_since(cx, "a@x.com", 365) == {"terrain-restore"}
    # a different source is untouched by replacing 'fmp'
    ph.replace_source(cx, "groovekart", [("a@x.com","wholomega",_iso(3),"gk9")])
    ph.replace_source(cx, "fmp", [("a@x.com","neuro-mag",_iso(2),"inv3")])
    assert ph.slugs_since(cx, "a@x.com", 365) == {"neuro-mag","wholomega"}

def test_window_excludes_old():
    cx = _cx()
    ph.replace_source(cx, "fmp", [("a@x.com","old-sku",_iso(400),"i1"),
                                   ("a@x.com","new-sku",_iso(100),"i2")])
    assert ph.slugs_since(cx, "a@x.com", 365) == {"new-sku"}
```

- [ ] **Step 2: Run to verify it fails.** `python3 -m pytest tests/test_purchase_history.py -v` → FAIL (module/attrs missing).

- [ ] **Step 3: Implement**

```python
# dashboard/purchase_history.py
"""Slug-keyed client purchase history backfilled from external sources
(FMP, GrooveKart). Separate from the live `orders` board. Feeds repertoire
seeding only. Pure: caller passes cx."""
from datetime import datetime, timedelta, timezone

def _norm(v): return (v or "").strip().lower()

def init_purchase_history_table(cx):
    cx.execute("""CREATE TABLE IF NOT EXISTS purchase_history (
        email TEXT NOT NULL, slug TEXT NOT NULL,
        purchased_at TEXT NOT NULL, source TEXT NOT NULL,
        source_ref TEXT NOT NULL,
        PRIMARY KEY (source, source_ref, slug))""")
    cx.execute("CREATE INDEX IF NOT EXISTS ix_ph_email ON purchase_history(email)")
    cx.commit()

def replace_source(cx, source, rows):
    cx.execute("DELETE FROM purchase_history WHERE source=?", (source,))
    n = 0
    for email, slug, purchased_at, source_ref in rows:
        e, s = _norm(email), _norm(slug)
        if not (e and s):
            continue
        if cx.execute("INSERT OR IGNORE INTO purchase_history"
                      "(email, slug, purchased_at, source, source_ref) VALUES (?,?,?,?,?)",
                      (e, s, purchased_at, source, str(source_ref))).rowcount == 1:
            n += 1
    cx.commit()
    return n

def slugs_since(cx, email, window_days):
    cutoff = (datetime.now(timezone.utc) - timedelta(days=int(window_days))).isoformat()
    return {r[0] for r in cx.execute(
        "SELECT DISTINCT slug FROM purchase_history WHERE email=? AND purchased_at>=?",
        (_norm(email), cutoff))}
```

- [ ] **Step 4: Run to verify pass.**
- [ ] **Step 5: Commit** — `feat(purchase-history): slug-keyed backfill store (separate from orders)`

---

## Task 2: FMP slug-map builder + review artifact

**Files:** Create `scripts/build_fmp_slug_map.py`; Output `data/fmp_slug_map.json` (committed after human review).

**Goal:** auto-resolve as many of the 396 `id_fk_product` values as possible and hand the human a short review list. NOT a fully-automated map — pricing correctness demands the review.

- [ ] **Step 1:** Write `scripts/build_fmp_slug_map.py` that, against `chat_log.db`:
  1. Loads `product_sales.slug_map_from_products_json()` → the `fmp_id → slug` pairs already in `products.json` (auto-resolved, high confidence).
  2. For every distinct `id_fk_product` in `fmp_invoice_items` not already resolved, computes the modal `description` (reuse `product_sales.aggregate_rows` logic) and runs it through `name_to_slug()` then `product-aliases.json` for a *suggestion*.
  3. Emits `data/fmp_slug_map.json`:
     ```json
     {
       "resolved":  { "425": "microbiome-30v", "448": "wholomega" },
       "review":    { "1187": {"suggestion": "neuro-magnesium", "description": "Neuro-Magnesium Powder", "line_count": 31} },
       "exclude":   [952],
       "_generated_note": "resolved = auto (products.json fmp_id or exact alias); review = human must confirm/replace 'suggestion' or move to exclude; exclude = non-products (Courtesy, equipment, services)"
     }
     ```
     Pre-seed `exclude` with id 952 (Courtesy) and obvious equipment/service ids (Zyto, tuning fork, toothbrush, certification — match by description keyword).
- [ ] **Step 2: HUMAN REVIEW (Glen/Rae).** Confirm/correct each `review` entry's slug, or move it to `exclude`. Move confirmed entries into `resolved`. This is the ~40–60 judgment-call step. Commit the finalized `data/fmp_slug_map.json`.
- [ ] **Step 3: Commit** — `chore(fmp): generate + review fmp_id→slug map` (script + reviewed JSON).

*(No unit test — this is a data/tooling task; Task 3 tests consume the finalized map.)*

---

## Task 3: FMP → purchase_history derivation

**Files:** Create `dashboard/fmp_history.py`; Modify `app.py` (FMP ingest route); Test `tests/test_fmp_history.py`.

**Interfaces (Produces):** `rebuild_from_fmp(cx, slug_map) -> dict` where `slug_map` = the loaded `data/fmp_slug_map.json`; returns `{rows, skipped_excluded, skipped_unmapped, skipped_noemail}`.

- [ ] **Step 1: Write the failing test** — seed tiny `fmp_clients`/`fmp_invoices`/`fmp_invoice_items` rows in an in-memory (or temp) DB: two products for one client (one in `resolved`, one in `exclude`), confirm `rebuild_from_fmp` writes only the resolved one into `purchase_history` with the invoice date + `source='fmp'`, and that a client with no email is skipped. (Mirror the `client_order_history` join in `fmp_orders.py:125` for the SQL.)
- [ ] **Step 2: Run to verify fail.**
- [ ] **Step 3: Implement** `rebuild_from_fmp`:
  - Join `fmp_invoice_items ⨝ fmp_invoices (date) ⨝ fmp_clients (email)` (read the exact column names from `fmp_orders.py` `_ITEM_COLS`/table defs — do NOT guess).
  - For each line: resolve `id_fk_product` → slug via `slug_map["resolved"]`; skip if in `exclude` or unresolved (count them); skip if client email blank.
  - Collect `(email, slug, invoice_date_iso, source_ref=item id_pk)` and call `purchase_history.replace_source(cx, "fmp", rows)`.
  - Wire a call to `rebuild_from_fmp` at the end of the FMP ingest route (app.py:28599) inside try/except (best-effort; never break the ingest).
- [ ] **Step 4: Run to verify pass.**
- [ ] **Step 5: Commit** — `feat(fmp-history): derive slug-keyed purchase_history from FMP invoices`

---

## Task 4: Union purchase_history into the repertoire seeder

**Files:** Modify `app.py` (the conversion-seed path from the membership branch); Test `tests/test_history_seed.py`.

**Interfaces (Consumes):** `purchase_history.slugs_since`, existing `_order_slugs_since`, `repertoire.add_skus`.

- [ ] **Step 1: Write failing test** — a member converting (prepay/continuous-care) whose ONLY record of buying SKU X is in `purchase_history` (source fmp), dated within 365d, gets X in their repertoire after conversion; a purchase_history row older than 365d does not seed.
- [ ] **Step 2: Run to verify fail.**
- [ ] **Step 3: Implement** — in the seed call site(s), after `seed_from_history(...)` from `orders`, also add the union from history:
  ```python
  try:
      if REPERTOIRE_ENABLED:
          import dashboard.purchase_history as purchase_history
          with sqlite3.connect(LOG_DB) as _hcx:
              purchase_history.init_purchase_history_table(_hcx)
              hist = purchase_history.slugs_since(_hcx, email, _window_days_for_term(term_months))
          if hist:
              repertoire.add_skus(cx, email, list(hist))
  except Exception as _e:
      print(f"[repertoire] history seed failed: {_e!r}", flush=True)
  ```
  (Keep it best-effort + fresh-DB-safe like the existing seed.) Note: use the SAME `_window_days_for_term(term_months)` the orders seed uses — the "1-year" global applies at the 12-month tier; shorter tiers still use their window for history too, consistent with orders.
- [ ] **Step 4: Run to verify pass.**
- [ ] **Step 5: Commit** — `feat(repertoire): seed from purchase_history (FMP/GK) alongside orders`

---

## Task 5: GrooveKart WebApi → purchase_history  ⛔ BLOCKED (needs API key)

**Precondition (Glen):** generate a WebApi key in remedymatch.com's GrooveKart admin, confirm the WebApi feature is on the store plan, and hand it over → stored as `GROOVEKART_API_KEY` (+ `GROOVEKART_BASE_URL`) in Doppler `remedy-match/prd`. Then run a spike hitting `GET /webapi/orders?display=full&limit=0,5` to confirm the real field shapes (docs vs reality can drift) BEFORE finalizing the parser.

**Files:** Create `dashboard/gk_client.py`, `dashboard/gk_history.py`; Modify `app.py` (console route); Tests `tests/test_gk_client.py`, `tests/test_gk_history.py`.

- [ ] **Step 1:** `dashboard/gk_client.py` — `iter_orders(since_iso, *, base_url, api_key, http=requests)`: Basic Auth (`base64(api_key + ":")`), `GET {base}/webapi/orders?display=full&sort=[id_ASC]&limit={offset},{page}` with `filter[date_add]=[{since},{now}]`, paginate until empty. Inject `http` for testing (no live calls in tests). Yield normalized `{order_id, email, date_add, lines:[{product_reference, product_name, qty}]}`.
- [ ] **Step 2:** `dashboard/gk_history.py` — `rebuild_from_gk(cx, orders_iter, *, slug_resolver)`: for each line, resolve slug via `slug_resolver` (a callable wrapping `_resolve_buy_slug(product_name)` + a reviewed `product_reference → slug` override map — same review discipline as FMP), skip unresolved/excluded, collect `(email, slug, date_add, source_ref=f"{order_id}:{line_ref}")`, call `purchase_history.replace_source(cx, "groovekart", rows)`.
- [ ] **Step 3:** Console route `POST /api/console/gk-history-rebuild` (console-key gated) that pulls the past 365 days and rebuilds. Best-effort, reports counts.
- [ ] **Steps TDD:** test `gk_client` against a fake `http` returning canned pages (pagination + date filter formatting); test `gk_history` maps + writes + skips unresolved. Commit `feat(gk-history): GrooveKart WebApi → purchase_history (past-year backfill)`.

---

## Deferred / Blocked / Notes

- **GrooveKart (Task 5)** is blocked ONLY on the API key + a field-shape spike. Everything else (Tasks 1–4) ships without it; GK adds retail buyers later with no rework.
- **Pre-2024 FMP history** is out of reach (not exported). If wanted, Glen exports the older FileMaker file → same pipeline.
- **GrooveKart CSV export** as an API fallback is UNVERIFIED — do not rely on it; the WebApi is the confirmed path.
- **Portal display is unaffected** — the portal reorder list stays portal-channel-only (Glen's rule); `purchase_history` feeds PRICING/seeding, not the display list.
- **Refresh cadence:** FMP rebuild piggybacks the existing manual CSV re-push; GK rebuild is the new console route (could later be a cron).

## Self-Review

**Coverage:** separate table (Task 1) ✓; 1-year window (Task 1 `slugs_since` + Task 4) ✓; reviewed id→slug map, exclude Courtesy/equipment (Task 2) ✓; FMP derivation idempotent (Task 3 via `replace_source`) ✓; seeder union (Task 4) ✓; GK via WebApi with SKU refs, blocked-on-key (Task 5) ✓; payment-only sources excluded (not in plan) ✓; portal display untouched (Notes) ✓.
**Placeholder scan:** Task 2 is intentionally a tooling+human-review task (no unit test) — flagged as such. Task 3/4 test bodies are described (mirror `fmp_orders.py` join / membership harness) rather than fully transcribed because they need the repo's FMP fixtures + app harness — flesh out at execution.
**Type consistency:** `replace_source(cx, source, rows)` rows are `(email, slug, purchased_at_iso, source_ref)` in Tasks 1/3/5; `slugs_since -> set[str]` consumed in Task 4; `slug_map` is the loaded `fmp_slug_map.json` dict in Tasks 2/3.
