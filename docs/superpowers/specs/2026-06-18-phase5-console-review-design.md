# Phase 5 — In-Funnel Sales Pages: Console Review (Approve / Edit / Regenerate)

**Date:** 2026-06-18
**Status:** Approved (design); ready for implementation plan
**Repo:** deploy-chat (Flask, Render `glen-knowledge-chat`)
**Builds on:** Phase 2 (AI copy drafts, live). Parent: `2026-06-18-funnel-sales-pages-design.md`.

---

## Problem

Phase 2 generates per-product narrative copy as a **draft** (the live page shows an AI caveat banner), but there's no way for Glen/Rae to review it. Phase 5 adds a console where they read each product's generated copy, **edit it inline**, **Regenerate** it (and immediately review the new copy), and **Approve** it — which drops the caveat banner on the live page (the copy is blessed).

## Scope

A console page (Business OS sub-tab) to review draft sales-page copy. Per product: edit the intro/overview/research text; **Regenerate** (immediate, in-process) to get fresh copy to review; **Approve** → `state=approved` → the live page's caveat banner disappears. Console-only (existing console auth + RBAC OWNER/OPS); no new public flag.

**Out of scope (Phase 5b):** capturing the triggering viewer + the "now Dr. Glen–reviewed" email. **Confirmed design choices:** editing keeps `state=draft` (Approve is the single deliberate publish step — editing is not auto-approval); **Regenerate generates immediately in-process** and returns the new copy for review (not a cache-clear deferred to the next visitor).

---

## Architecture

### Data (extend Phase-2 `sales_pages`)
- Reuse `sales_pages(product_slug PK, state, content_json, model, generated_at, …)`. Add `approved_at TEXT`, `approved_by TEXT` (lazy `ALTER TABLE … ADD COLUMN`, additive). `state` ∈ {draft, approved}.
- New data-layer functions in `dashboard/sales_pages.py`: `set_state(cx, slug, state, by="")` (sets state + approved_at/by when approved); `list_draft_pages(cx) -> list[{slug, sections:[...], state}]` (pages with non-empty `content_json`).

### Regeneration helper (reuse Phase-2 gen building blocks)
- `_regenerate_sales_copy(slug) -> dict|None` in `app.py`: for each section in `sales_copy.NARRATIVE_SECTIONS`, build the prompt via `sales_copy.build_section_prompt(section, product)` (enrich ingredients via `_product_card` like the gen endpoint), call `_cl.messages.create(model="claude-haiku-4-5-20251001", …)` (non-streaming — this is a console action, not a live stream), **strip em dashes** (`_strip_dash` over the text), `upsert_section(...)`, and `set_state(cx, slug, "draft")`. Returns the new `{section: text}` or None on failure. Compliance constraints come from `build_section_prompt` (unchanged).

### Console actions (dispatch_action spine)
New `dashboard/sales_pages_actions.py` (or registered inline), all RBAC `(OWNER, OPS)`, `risk_tier=LOW_WRITE`:
- `sales_pages.approve` `{slug}` → `set_state(cx, slug, "approved", by=actor.name)`; logs an event.
- `sales_pages.regenerate` `{slug}` → calls `_regenerate_sales_copy(slug)`; returns the new content; logs an event.
- `sales_pages.edit` `{slug, section, text}` → `upsert_section(cx, slug, section, text)` (state stays draft); logs an event.

### Console API + UI
- `GET /api/console/sales-pages` (console-key gated) → `list_draft_pages` rows (slug, name, which sections present, state).
- `GET /api/console/sales-page/<slug>` → the product's current `content_json` sections + state + the live-page URL (`/begin/product/<slug>`).
- `static/console-sales-pages.html` (modeled on `console-biofield-portal.html`): left list of products with draft copy; selecting one loads the editor — the intro/overview/research sections as labeled textareas, a "View live page ↗" link, and **Save** (per-section → `sales_pages.edit`), **Regenerate** (→ `sales_pages.regenerate`, then re-render the textareas with the returned copy), **Approve** (→ `sales_pages.approve`, then mark the row approved). Actions POST to the existing `/api/action/<key>` dispatch route with `X-Console-Key`. NO emoji.
- Add a BOS sub-tab entry in `static/op-nav.js` (`{ id: "sales", label: "Sales Pages", href: "/console/sales-pages" }`) + a `/console/sales-pages` route serving the page.

### Banner drop (page-data + frontend)
- `begin_product_page_data` adds `ai_state` = `sales_pages.get_page(slug).state` if a page row exists, else `"none"` (only when `_SALES_AI_COPY_ENABLED`).
- `static/begin-product.html`: the caveat banner currently shows when any section has `ai`. Change the condition to also require `data.ai_state !== 'approved'` — so an **approved** page shows the generated copy with **no banner**.

---

## Data flow

1. Viewer opens a page → Phase-2 generates draft copy → live page shows the caveat banner.
2. Glen opens console **Sales Pages** → sees the product in the draft list → reads the copy.
3. He edits a phrase (Save, stays draft) / clicks **Regenerate** (new copy appears in the editor) → reviews.
4. He clicks **Approve** → `state=approved` (+ approved_at/by) → the live page drops the banner; the blessed copy stands.

## Error handling

- `_regenerate_sales_copy` failure (Claude error) → leave existing copy unchanged, return None; the action reports the failure (copy not lost).
- `approve` is idempotent (approving an approved page is a no-op set).
- Console endpoints gated by the console key; actions gated by RBAC (OWNER/OPS) via `dispatch_action`. Bad slug → 404/empty.
- Banner block wrapped so a `sales_pages` read error never breaks page-data (degrade to showing the banner).

## Testing

- **Data layer:** `set_state` sets state + approved_at/by; `list_draft_pages` returns pages with content, excludes empty.
- **Actions:** `sales_pages.approve` → state=approved + approved_by; `sales_pages.edit` → upserts text, state stays draft; `sales_pages.regenerate` (mock `_cl`) → content updated, state=draft, em dashes stripped. RBAC denies a non-OWNER/OPS actor.
- **Console API:** list returns draft products; per-product load returns sections + state; console-key gating.
- **page-data `ai_state`:** draft page → `ai_state:"draft"`; approved → `"approved"`; no page → `"none"` / absent when copy flag off.
- **Banner:** (frontend, manual) approved → no banner; draft → banner.
- Follow deploy-chat test isolation (tmp `$DATA_DIR/chat_log.db`; mock Supabase; importorskip playwright). The console UI is a manual visual pass.

## Notes

Console-only feature — gated by the existing console auth, not a public flag, so it's safe to merge without a dark-launch flag (it changes nothing on the public funnel except that an **Approve** action drops a banner, which is the intended effect). Phase 5b adds the triggering-viewer capture + approval email.
