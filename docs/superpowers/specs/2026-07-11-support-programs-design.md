# Condition Support Programs — design

**Date:** 2026-07-11
**Status:** design → build (Glen approved the 9 remedy groups + broad-benefit list)
**Builds on:** FF matches (#777/#782/#784, live behind `FF_MATCHES_ENABLED`)

## Problem

Eye/vision is Glen's area of greatest concentration. When a client carries an eye-condition
health tag, we want to surface a curated **ongoing support program** — an ordered remedy
group Glen authored for that condition — on their portal, orderable / add-to-invoice, framed
as ongoing monthly support. Separately, a `broad_benefit` designation marks formulations that
are frequently a good match and broadly beneficial, so the FF matcher can favor them.

These are Glen's CLINICAL protocols. The 9 groups + broad-benefit list have been drafted from
his existing data, edited and approved by Glen (source of truth:
`data/condition_programs_seed.json`).

## The 9 conditions
`glaucoma-elevated-iop`, `glaucoma-normal-iop`, `dry-amd`, `wet-amd` (consult-recommended),
`senile-cataract`, `psc-cataract`, `dry-eye`, `retinitis-pigmentosa`, `diabetic-retinopathy`.

## Key facts (from investigation)
- **Health tags already exist**, email-keyed: `people.conditions` (JSON list synced from GHL)
  and `people.tags` in the `pb:<condition>` namespace; `dashboard/biofield_profile.py` already
  prettifies "Wet AMD" / "PSC Cataract". The FF endpoint already resolves token→email.
- **Glen-authored dosing SHOWS on these cards** — unlike the AI FF-matches card (which hides
  dosing until review), support programs carry Glen's own doses, so they are displayed.
- **Program items are structured:** each item is `{slug, name, dose?, note?, alts?}` where
  `alts` are either/or substitutes (e.g. OcuHeal *or* Neuro Eye Drops) and `note` is a
  conditional ("Add for brunescent cataracts").
- `_qty_eligible`/never-recommend gates do NOT apply here — these are Glen's explicit lists;
  render exactly what he authored (some items, e.g. eye drops, may not be qty-eligible).

## Architecture — 4 slices, all behind `SUPPORT_PROGRAMS_ENABLED` (dark)

### Slice 1 — data stores + editor
- **`dashboard/condition_programs.py`** — sqlite store. Table `condition_programs(condition_key
  PK, label, consult_recommended INT, items_json, updated_at)`. `init_table`, `seed_if_empty(cx,
  seed)`, `get(cx, key)`, `all(cx)`, `upsert(cx, key, label, consult_recommended, items)`.
- **`dashboard/broad_benefit.py`** — sqlite store. Table `broad_benefit(slug PK, added_at)`.
  `init_table`, `seed_if_empty`, `is_broad(cx, slug)`, `all_slugs(cx)`, `add(cx, slug)`,
  `remove(cx, slug)`.
- **Seed loader:** on first use, seed both from `data/condition_programs_seed.json`
  (`condition_programs` + `broad_benefit_slugs`). Idempotent — never overwrites operator edits.
- **Console API** (mirror `/api/console/ff-match-drafts`, `_portal_console_ok()` gated):
  `GET /api/console/condition-programs` (all + broad list), `POST /api/console/condition-programs`
  (upsert one program), `POST /api/console/broad-benefit` (add/remove a slug).
- **Editor** `static/console-support-programs.html` at `/console/support-programs` (mirror
  `/console/ff-drafts`): edit each program's items (slug/name/dose/note/alts, reorder,
  add/remove), the consult flag, and toggle broad-benefit slugs. Product picker validates slugs
  against the catalog.

### Slice 2 — broad_benefit → FF matcher
Wire `broad_benefit.is_broad(slug)` into `_make_ff_items_for`: mark each candidate, surface
"(broadly effective)" to `_ff_llm_rank`'s prompt, and apply a deterministic tie-break so a
broadly-beneficial formulation rounds out the set. Preserves all existing FF safety invariants.

### Slice 3 — client eye-condition tagging
- **Read** a client's eye-condition: map `people.conditions`/`pb:` tags → one of the 9 keys
  (a tag→key normalizer; reuse `biofield_profile` prettify vocabulary).
- **`dashboard/client_conditions.py`** — operator override store `client_conditions(email PK,
  condition_key, set_by, updated_at)`. Console `GET/POST /api/console/client-condition?email=`.
  Card resolution: override wins, else the people-tag mapping, else none.
- Console control (on the client console / a field on the support-programs or people view) to
  set a client's eye condition.

### Slice 4 — the support-program portal card
- **Payload:** `api_client_portal` gains `payload["support_program"]` when
  `SUPPORT_PROGRAMS_ENABLED` and the (member-aware) client resolves to a condition:
  `{condition_key, label, consult_recommended, items:[{name, url, dose?, note?, alts:[{name,url}]}]}`.
  Each `url` = `order_destination.destination_for(slug)`. Flag-off → key absent (byte-identical).
- **Card** in `static/client-portal.html`: "Your <Label> Support Program", lists items with
  **dosing shown**, either/or alternatives ("or <Alt>"), conditional notes, order links, and —
  when `consult_recommended` — a prominent "Book a consultation" CTA (Wet AMD). Gated on a
  `support_program_enabled` payload flag (like the FF button).
- **Add-to-invoice:** `POST /api/portal/<token>/support-program/add-to-invoice` — mirror the FF
  insert-once/never-downgrade money path (unpaid/unpublished `orders` row, deterministic
  `SPINV-<email>-<condition_key>` external_ref, real FF prices via `_ff_line_cents`, gated on
  `_ff_covered`). Reuse that machinery.

## Non-goals
- Auto-populating the 9 eye-condition tags on clients (that is Glen/Rae's tagging step; the
  override store gives them a control). The card fires only for tagged clients.
- Changing the FF-matches card behavior (Slice 2 only reads the broad-benefit flag).
- A separate billing/subscription for programs (uses the existing add-to-invoice → composer).

## Safety / rollout
- Everything behind `SUPPORT_PROGRAMS_ENABLED`, default OFF. Console editor is key-gated.
- The add-to-invoice path inherits the FF money-path guards (covered-only, insert-once,
  never-downgrade, real price). Verified live on a tagged test client before broad enable.
