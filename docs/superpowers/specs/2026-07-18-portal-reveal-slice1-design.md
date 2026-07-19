# Portal-Reveal Unification — Slice 1: Portal reads the reveal

**Date:** 2026-07-18
**Status:** spec for review
**Part of:** reveal-in-portal unification (Option 2 — unify data+display, keep the funnel; backfill existing clients; single source of truth = `biofield_reveals` System A; System B folded/retired in a LATER slice).

## Scope of this slice

Make the client portal render a client's biofield **reveal** (`biofield_reveals`, System A) as their scan — most-recent on the portal home + in Scan History — for clients who already have portal access. Nothing else changes: no portal provisioning (slice 2), no backfill/mass-email (slice 3), no retiring System B (slice 4), no funnel changes.

**Why first:** every reveal client already has a System-A row, but the portal today reads only System B (`portal_biofield_reports`) + a legacy `client_portals` fallback — so a reveal-only client sees *nothing* in the portal. This slice closes that gap with the least risk and no data migration.

## Goal / non-goals

**Goal:** a portal user whose only biofield artifact is a `biofield_reveals` row sees that reveal in the portal (home most-recent + Scan History tabs), rendered with the **same blur/unlock state as the funnel** (never a free un-blur).

**Non-goals (explicitly out):** provisioning portals, emailing portal links, backfilling, migrating/retiring System B curated reports, changing `/begin/biofield`, changing the console queue.

## Design

### Resolution order (additive — System B still wins in this slice)

The portal biofield block resolves, in order:
1. **System B report** (`portal_biofield_reports` has rows for the email) → use it, exactly as today. (Curated reports are untouched; folding them is slice 4.)
2. **System A reveal** (NEW) — no System B rows, but `biofield_reveals` has ≥1 row for the email → assemble the block from the reveal(s).
3. **Legacy** `client_portals` content → as today.
4. Else `{"visible": False}`.

Net effect: clients with a curated B report are unchanged; the majority (reveal-only) now see their reveal; nobody loses anything.

### Consolidate the two builders FIRST

Biofield content is currently assembled in **two** independent places that both read `_pbr` directly:
- `dashboard/portal_view.py::_biofield_block` (the `/api/portal/<token>/view` path via `get_portal_view`).
- an inline reader in the live `/api/portal/<token>` route (app.py ~18398).

Before adding the System-A path, **consolidate** both onto the single `portal_view._biofield_block` (pure, cx-in) so the reveal read is added ONCE and the two paths cannot drift. If full consolidation is too invasive for this slice, the fallback is to add the System-A branch to BOTH and cover both with tests — but consolidation is preferred and is the recommended approach.

### Assembling a block from a reveal

New pure helper (e.g. `dashboard/portal_view.py::_biofield_block_from_reveal(cx, reveal_rows, scan_date, unlock)`), mapping `biofield_reveals` → the existing `_assemble_biofield` output shape `{visible,status,blurred,actionable,scan_date,scan_dates,greeting,video,layers,pricing_note}`:

- `greeting` ← `interpretation.greeting`.
- `layers[]` ← reveal `layers` (each `{n,title,meaning}`; `remedy`/`dosing` included only when unlocked — see blur).
- `scan_dates` ← all `biofield_reveals.scan_date` for the email (newest-first) → drives history tabs.
- `scan_date` ← requested date if present in the set, else newest.
- `status` ← derived: `confirmed` if `first_approved` or `paid`, else `requested` if `requested_at`, else `pending`. (Truthful; only display strings.)
- `actionable` ← `is_actionable(scan_date, today)` (30-day window), same as System B.

### Blur / unlock — mirror the funnel (revenue protection)

The portal reveal MUST NOT un-blur beyond what the funnel would show. The reveal assembler keys strictly off the funnel's own unlock flags:
- `paid` → all remedies visible.
- `top_unlocked` (the funnel's computed flag) → top remedy visible, rest blurred.
- else → all remedies blurred (interpretation + layer titles/meanings still show, as on the funnel).

`_biofield_unlock_flags(row, email)` (in `app.py`) computes `{paid, first_approved, top_unlocked, ...}`. `_biofield_block`/the reveal assembler stay pure in `portal_view`, so the caller computes the flags and passes them IN as an `unlock` struct (mirrors the existing `unlocked=` param, which is boolean-paid only — the reveal branch needs `top_unlocked` too, so widen it to a small dict for that branch). When blurred, remedy/dosing strings are NEVER assembled into the payload (same guard as `_assemble_biofield`'s `show`).

## Data

No schema change. `biofield_reveals` already carries interpretation, layers, remedies, `first_approved`, `requested_at` (#1014), and the spend/free-unlock ledgers.

## Edge cases

- Client has BOTH a B report and an A reveal → **B wins** (slice-1 rule).
- Household member re-point → read the *member's own* reveal by their email; a member with no reveal shows the empty state (never the account-holder's — preserve current fail-closed behavior).
- Multiple reveals same `scan_date` → `biofield_reveals` is unique on `(email, scan_date)`, so at most one per date.
- Suppressed/empty reveal (no layers/remedies) → `{"visible": False}`.
- `requested_at` set but not approved → status `requested`, still blurred.

## Testing

Unit (pure, in-memory sqlite — no app import, no email):
- reveal→block: not-approved → all `blurred`, no remedy leaks server-side; `first_approved` → top only; `paid` → all.
- resolution order: B present → B block; only A → A block; neither → `{visible:False}`.
- `scan_dates` lists all reveal dates newest-first; picking an explicit date works.

Integration / render-verify:
- Headless-render the portal for a reveal-only test client → the "most recent scan" card + Scan History show the reveal, remedies blurred; confirm no un-blur without paid/approved.
- Confirm a curated-B client's portal is byte-for-byte unchanged.

## Risks

- **Two-builder consolidation** could subtly shift `/view` vs live-portal output → verify both render identically before/after (snapshot the block for a B client).
- **Revenue leak** if blur is computed wrong → the mirror-the-funnel rule + explicit "remedy never leaves server when blurred" test is the guard.
- Consolidation touches a hot path (portal load) → keep the reveal read best-effort/None-raising so a reveal read failure never breaks portal load.

## Out of scope (later slices)
2. Auto-provision portals + portal link in the reveal email.
3. Backfill provisioning + notify (batched, dry-run, mass-email-cap-aware).
4. Fold + retire System B (migrate curated reports into reveal records; consolidate the portal request button onto `requested_at`; redirect the standalone reveal page).
