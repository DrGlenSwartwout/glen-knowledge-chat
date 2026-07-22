# Portal — My Health Profile (Phase 2) — Rollout & Verification Note

**Date:** 2026-07-22
**Branch:** `sess/c9f1462d-phase2` (off `main` at `4f311ed2`, i.e. after the Phase 0/1 hub merge)
**Plan:** `docs/superpowers/plans/2026-07-22-portal-health-profile.md`
**Flag:** `PORTAL_HEALTH_PROFILE_ENABLED` (OFF — ships dark)

## What shipped on this branch

A "My Health Profile" tile + drill-in panel in the client portal (only when the
hub is on and the flag is on): the client's own self-reported intake record,
editable — **including the 5 clinical-dimension scales**, which evolve as they
heal. Plus a suggestions queue the client approves.

- **Editable record** over the ONE source of truth: `intake_responses` (email-keyed,
  in `chat_log.db`). Every edit and every confirmed suggestion writes there via
  `intake.save_self_edit`, preserving `submitted` status (never clobbers). The
  admin console reads the same table (`/api/console/intake/<email>`), so client
  and console never diverge.
- **Editable scope:** field ids from INTAKE_FORM sections `goals`, `dimensions`,
  `history` (17 ids incl. the 5 dimensions). Excludes identity + consent (`terms`).
- **Suggestions queue** (`health_suggestions` table, pending until approved), two
  sources: **chat** (extracted from Ask-Dr-Glen turns) and **clinician** (a console
  endpoint the practitioner uses). Confirm/Edit writes the value into the record;
  Dismiss writes nothing. Ownership + identity resolved from the portal token
  (client side) or console auth (clinician side), never the request body.

Commits (oldest first): `ab9e06af` flag+read · `13fa20cf` tile+panel · `3c3f78ae`
write-back · `2d6a7b86` inline edit · `5cc90ca6` 400-guard+HST+endpoint test ·
`0feeac5f` CI-safe test · `2922a17c` suggestions table+chat hook · `c6e503df`
suggestions endpoints · `5aedc64c` clinician endpoint · `c8661a88` suggestions UI.

## Flag

| Flag | State | Notes |
|---|---|---|
| `PORTAL_HEALTH_PROFILE_ENABLED` | **OFF** | Ships dark. Same truthy set as the hub flag. Reversible by unsetting. The tile/panel only appear when this AND the hub flag are on. Set with `doppler secrets set PORTAL_HEALTH_PROFILE_ENABLED=on -p remedy-match -c prd`. |

## PRE-FLIP GATES (do these before turning the flag on)

1. **Postgres dialect (important).** Prod is Postgres now. The dedupe relies on
   `INSERT OR IGNORE` + a **partial UNIQUE index** (`... WHERE status='pending'`).
   The `health_suggestions` table is created lazily (`init_table`) only when the
   feature runs, so nothing hits Postgres until the flag is on. Before flipping,
   confirm the SQLite→Postgres adapter translates `INSERT OR IGNORE` (→ `ON
   CONFLICT DO NOTHING`) and creates the partial index (Postgres supports partial
   indexes natively). If the adapter passes these through raw, dedupe/init could
   error in prod. Verify against a live Postgres, or add an adapter translation.
2. **Chat extractor is a no-op seam.** `extract_from_turn` has NO live LLM
   extractor wired, so chat-sourced suggestions produce ZERO rows in prod. The
   **clinician** path DOES populate the queue. Decide: wire the LLM extractor
   (mirror `_CONCIERGE_EXTRACT_SYSTEM`, app.py:11162) before the flip if chat
   suggestions should work at launch, or flip with clinician-only suggestions and
   wire chat later.

## Render-verify checklist (run once against a live portal, flag on)

- [ ] Flag OFF → no My Health Profile tile, no panel (off-state unchanged).
- [ ] Flag ON → tile appears in the Understand group; badge shows pending count.
- [ ] Open the panel → the record renders (goals, the 5 dimensions, history).
- [ ] Edit a scalar/scale field → Save → value persists AND
      `/api/console/intake/<email>` shows the new value (one source of truth).
- [ ] A clinician POST to `/api/console/client/<email>/health-suggestion` → a
      pending item appears in the client's panel framed "Your practitioner
      suggests…" → client Confirm writes it into the record; Dismiss leaves the
      record unchanged.
- [ ] (If the chat extractor is wired) mentioning a health fact in the chat
      surfaces a pending "We heard this in your conversations…" item.
- [ ] No duplicate pending rows when the same thing is suggested twice.
- [ ] Dark mode readable; no browser alert/confirm dialogs anywhere in the flow.

## Deferred / follow-ups (non-blocking, documented)

- **Wire the LLM chat extractor** (gate #2 above).
- **Enrich `health_profile.build_block` field metadata:** forward `options` (so
  `single_choice` renders a `<select>` not a text box), scale `range` (so the
  frontend drops its hand-maintained `HEALTH_SCALE_MAX` map), and table `columns`
  labels (so table rows show labels not raw keys). One backend change fixes the
  Task 2 + Task 4 UI gaps.
- **Table-field editing** (health_concerns, supplements, etc.) — currently a
  "coming soon" note; add/remove-row editing is a follow-up.
- **`suggested_value` encoding:** `json.dumps` strings too, so a text value like
  `"3"` doesn't round-trip to an int (inherited Task 5/6 quirk, low impact).

## Not merged

This branch is NOT merged and has no PR yet — it builds on the Phase 0/1 hub
(PR #1139). Open a PR once the pre-flip gates are understood; like the hub it
merges dark (flag off), so the merge itself changes nothing users see.
