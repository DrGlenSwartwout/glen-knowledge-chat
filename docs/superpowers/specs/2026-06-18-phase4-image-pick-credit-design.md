# Phase 4 — In-Funnel Sales Pages: Pairwise Image Pick + Order-Redeemable Credit

**Date:** 2026-06-18
**Status:** Approved (design); ready for implementation plan
**Repo:** deploy-chat (Flask, Render `glen-knowledge-chat`)
**Builds on:** Phase 1/2 live, Phase 3 live (images: 2 variants/type stored, 1 displayed). Parent: `2026-06-18-funnel-sales-pages-design.md`.

---

## Problem

Phase 3 stores 2 image variants per type but only shows one. Phase 4 turns the images section into a **community pick**: the viewer chooses their favorite of each pair, which (a) gives us A/B preference data for the Phase-4b champion-challenger tournament and (b) rewards the viewer with an order-redeemable credit — turning image feedback into engagement + a conversion nudge.

## Scope

Pairwise forced-choice over the 2 variants/type + a "neither" escape; pick in **both** pairs → **1 credit, realized only when the viewer orders that product**. Votes tallied for 4b. **Out of scope (Phase 4b):** the champion-challenger tournament — significance thresholds, declaring winners, enqueuing challenger re-renders. Phase 4 just stores the votes and flags "neither" pairs as regeneration candidates.

**Gate:** new flag **`SALES_PAGES_IMAGE_PICK`** (default OFF); requires `SALES_PAGES_AI_IMAGES` on (needs images). Ships dark.

---

## Architecture

### `dashboard/sales_votes.py` — data layer (SQLite `chat_log.db`)
- `sales_page_votes(id, product_slug, kind, chosen_variant, session_id, email, created_at, updated_at)` — one logical pick per `(session_id, product_slug, kind)`; **upsert** (re-pick updates, never double-counts). `chosen_variant` is an int, or `0` for "neither". `email` filled when the member is known, else `''`.
- Functions: `init_table(cx)`, `record_pick(cx, slug, kind, variant, session_id, email="")` (upsert on `(session_id, slug, kind)`), `get_picks(cx, slug, *, session_id, email="")` → `{"botanical": variant|None, "mechanism": variant|None}` (match by session_id OR non-empty email), `picked_both(cx, slug, *, session_id, email="")` → bool (both kinds have a real pick ≥1, not "neither"), `tally(cx, slug)` → `{kind: {variant: count}}` (reads for 4b; "neither" excluded).

### Pick route
`POST /begin/product-image-pick/<slug>` body `{kind, variant}` (`variant` int ≥1 or the string `"neither"`):
- 404 if flag off / unknown slug; 400 if `kind ∉ {botanical, mechanism}` or invalid variant.
- `record_pick(...)` keyed on the `amg_session` cookie (+ member email via `get_authenticated_user`/portal identity if present). `"neither"` records variant `0` (a 4b regeneration-candidate flag; no re-render here).
- Returns `{"ok": true, "picks": {botanical, mechanism}, "both_picked": bool}`.

### Credit at order settlement
In `_settle_order_points(order, *, order_ref)` (app.py:2644), after existing points logic: if the order's product matches a slug for which the ordering viewer **picked both pairs**, grant **1 point** via `points.credit(cx, email, value_cents=<1 point>, reason="image_pick", order_ref=f"imgpick_{slug}", scope="rm")`. Idempotent via `points.has_entry(order_ref=f"imgpick_{slug}", reason="image_pick")` — granted at most once per (person, product). Match votes by the order email OR the order's `amg_session` (so an anonymous picker who later identifies at checkout still earns it). The order-completion caller passes the current `session_id` to the settlement helper (small signature add) or `_settle_order_points` reads it from `request` when in a request context.

### Page-data + frontend
- `begin_product_page_data`: when `SALES_PAGES_IMAGE_PICK` is on AND both variants exist for a kind AND the viewer hasn't picked it → the images body gains `pick: {botanical: [{variant,url}×2]|null, mechanism: [...]|null, picked: {botanical, mechanism}, both_picked}`. If the viewer has picked a kind, that kind's **chosen** image is the hero shown (their vote reflected back); unpicked kinds show the pair to choose. Flag off → exactly the Phase-3 images body (no `pick` field).
- `static/begin-product.html` `renderImagesBody`: render each unpicked pair as two side-by-side images with a "Pick this" affordance + a "Neither / show me something else" link; on pick → `POST` the pick, replace that pair with the chosen image; when both picked → show a muted "Thanks — you've helped shape this, and earned a credit toward this product." NO emoji.

---

## Data flow

1. Viewer opens images (flag on, 2 variants exist) → sees botanical pair + mechanism pair.
2. Picks one per pair (or "neither") → each pick upserts a vote keyed on session (+ email if known); the chosen image replaces the pair.
3. Both picked → "credit earned" note; the credit is **held** (a vote record), not yet a point.
4. Viewer later orders the product → `_settle_order_points` finds both-pair votes for them (by email or session) → grants 1 point, once.

## Error handling

- Unknown slug/kind/variant → 400/404; flag off → 404 (pick route) / no `pick` field (page-data).
- Re-pick updates the same row (last choice wins) — never double votes or double credit.
- Credit only via `has_entry` guard → at most one point per (person, product), only at order time.
- Settlement is wrapped so a vote-lookup error never blocks the order's normal points/settlement.
- "neither" stored as variant 0; excluded from tally and from `picked_both` (so "neither" alone earns no credit).

## Testing

- **Votes data layer:** `record_pick` upsert (re-pick updates, one row); `get_picks` matches by session and by email; `picked_both` true only when both kinds have a real pick (≥1), false if either is "neither"/missing; `tally` counts per variant, excludes "neither".
- **Pick route:** records/updates; 404 flag-off / unknown slug; 400 bad kind/variant; "neither" path; returns correct `both_picked`.
- **Page-data pick state:** flag on + 2 variants + unpicked → `pick` pairs present; after picking a kind → that kind shows chosen hero; flag off → no `pick` field (Phase-3 identical).
- **Credit at settlement:** with both-pair votes for the email/session, ordering the product grants exactly 1 `image_pick` point (idempotent on re-settle); no votes or only one pair → no credit; votes for a different product → no credit; "neither" in a pair → no credit.
- Follow deploy-chat test isolation (tmp `$DATA_DIR/chat_log.db`; mock Supabase; importorskip playwright). Pick UI is a manual visual pass.

## Flags

- `SALES_PAGES_ENABLED` + `SALES_PAGES_AI_COPY` + `SALES_PAGES_AI_IMAGES` (live) + **`SALES_PAGES_IMAGE_PICK`** (new, default OFF). Image-pick needs images on; with pick off, images render exactly as Phase 3.
