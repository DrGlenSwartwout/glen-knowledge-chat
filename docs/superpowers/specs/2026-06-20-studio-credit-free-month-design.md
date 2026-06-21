# Studio-Credit Free Month — Design

**Date:** 2026-06-20
**Status:** Approved (brainstorm) — ready for implementation plan
**Scope:** Phase 1 = console side only. Public self-serve claim form deferred to a later phase.

## Problem / Goal

People who get the studio coaching app should be credited with **one free month of the
paid membership**. The signal that someone "got the studio app" is not yet automated — at
minimum, purchasers will email us a copy of their invoice. We want a structured way to
grant the comp today (manual, console-driven) without anything falling through an inbox,
built so an automated trigger can plug in later with no rework.

## Decisions (locked during brainstorm)

1. **Pure comp window**, not a converting trial. 30 free days of membership access, no card
   collected, nothing auto-charges. When it expires the member subscribes the normal way.
   This reuses the existing access-grant path; **no Stripe/billing work**.
2. **Claim → approval queue → grant** architecture (not console-grant-only, not fully
   automated). A durable claim record is the source of truth; approval triggers the grant.
3. **Build the console side first.** The public self-serve claim form is a later phase that
   feeds the same queue. No public-facing surface ships in Phase 1.
4. **Idempotency = one studio credit per email per year.** A new grant is blocked if the
   email received a `studio_credit` grant within the last 365 days (overridable with an
   explicit force). No prior grant, or a prior grant older than 365 days, is allowed.

## Reuse — existing plumbing this builds on

- `_grant_membership(cx, email, days, source)` (app.py ~6201) — inserts a `memberships`
  access-window row (no card, no auto-charge); returns the membership id.
- `admin_membership_grant()` (app.py ~19371) — already permits `source="studio_credit"`,
  defaults to 30 days, mints a magic link via `_mint_membership_magic_link`, sends the
  "coaching access is open" email, and writes a `journey_events` `membership_granted` row.
  The shared grant behavior is factored into one reusable function that both this route and
  the new approve action call.
- `_active_membership_for_email` / `memberships` table — access checks already key off this,
  so a granted studio credit unlocks the member experience with zero extra wiring.
- Console dispatch spine: `dashboard/actions.py` `register_action` / `Action` / `LOW_WRITE`,
  RBAC `OWNER`/`OPS`, mirroring `dashboard/biofield_reveal_actions.py` and
  `/console/biofield-reveals`.

## Architecture

### 1. Data model — new table `studio_credit_claims`

```
id            TEXT PRIMARY KEY   -- uuid
email         TEXT NOT NULL      -- lowercased
invoice_ref   TEXT               -- free text: invoice #, Studio.com order id, or note
proof_note    TEXT               -- where/what the proof is ("emailed invoice 6/20", ref)
status        TEXT NOT NULL      -- 'pending' | 'approved' | 'rejected'
created_at    TEXT NOT NULL
created_by    TEXT
decided_at    TEXT
decided_by    TEXT
decision_note TEXT               -- rejection reason / approve note
membership_id TEXT               -- set on approve; links to the granted memberships row
source        TEXT NOT NULL      -- 'console' (Phase 1); 'self_serve'/'webhook'/'sku' later
```

Idempotent `CREATE TABLE IF NOT EXISTS` migration run at startup alongside the other table
inits.

The `source` column is the seam for automation: a future Studio.com webhook or an in-funnel
SKU just inserts a claim (optionally pre-approved) and runs the same grant path.

### 2. New files (mirror the biofield-reveal pattern)

- `dashboard/studio_credit.py` — store + grant logic (no Flask import; testable standalone):
  - `migrate(cx)` — create table if not exists.
  - `add_claim(cx, *, email, invoice_ref, proof_note, source, created_by)` → claim dict.
  - `list_claims(cx, status=None)` → rows for the console card.
  - `get(cx, claim_id)` → claim dict or None.
  - `_studio_credit_granted_within_year(cx, email)` → bool (most-recent `studio_credit`
    membership for email with `granted_at` inside the last 365 days).
  - `approve_claim(cx, claim_id, *, decided_by, force=False, grant_fn, notify_fn)` →
    result dict. Guards per Decision 4, calls `grant_fn` (the reused grant helper) on
    success, stores `membership_id`, flips status to `approved`, returns
    `{ok, membership_id, magic_link_url}` or `{warning: "active_year", ...}` when blocked.
  - `reject_claim(cx, claim_id, *, decided_by, reason)` → flips to `rejected`; no grant.
- `dashboard/studio_credit_actions.py` — dispatch actions registered via `register_action`,
  `OWNER`/`OPS`, `LOW_WRITE`:
  - `studio_credit.add` — params: email, invoice_ref, proof_note.
  - `studio_credit.approve` — params: id, force(optional).
  - `studio_credit.reject` — params: id, reason.
  - `configure(**kw)` to inject the grant + notify functions (so app.py wires in
    `_grant_membership` + the magic-link/email/journey-event behavior).

### 3. Console page — `/console/studio-credits`

Read-only render (mirrors `/console/biofield-reveals`): an "Add studio credit" form
(email + invoice_ref + proof_note) and a pending-claims list with Approve / Reject buttons.
All mutations go through the dispatch actions above. **Console-only — no new public flag.**

### 4. Grant logic + idempotency (the approve path)

On `studio_credit.approve` for a `pending` claim:
1. If claim already `approved` → no-op, return existing `membership_id` (idempotent).
2. If `_studio_credit_granted_within_year(email)` and not `force` → return
   `{warning: "active_year", granted_at, until}`; **no grant**. Console shows
   "already received a studio credit on <date> — override?" and re-submits with `force=true`.
3. Otherwise: `_grant_membership(cx, email, 30, "studio_credit")`, write a `journey_events`
   `membership_granted` row, send the existing magic-link "coaching access is open" email,
   set claim `status='approved'`, `membership_id`, `decided_at`, `decided_by`.

`reject` sets `status='rejected'` with the reason and sends nothing.

## Testing

Unit tests on `dashboard/studio_credit.py` against a temp sqlite (per the project's
test-isolation pattern — **no `import app`**; seed a tmp DB and pass stub `grant_fn` /
`notify_fn`):

- `add_claim` inserts a `pending` row with the given source.
- `approve_claim` on a pending claim calls `grant_fn` exactly once for 30 days / source
  `studio_credit`, stores `membership_id`, flips to `approved`.
- Double-approve is idempotent (no second grant; same membership_id returned).
- An email granted within the last 365 days is blocked without `force`, granted with
  `force=true`.
- An email whose only prior studio credit is >365 days old is allowed (no force needed).
- `reject_claim` grants nothing and flips to `rejected` with the reason.

## Explicitly NOT in this phase (YAGNI)

- Public self-serve claim form / page.
- Invoice file upload (Phase 1 uses a free-text proof note/ref).
- Studio.com webhook, in-funnel SKU trigger, or any auto-approval.
- Convert-to-paid / card-vault logic (this is a pure comp).

All deferred. The `source` column and the reusable grant function are the seams that let
those plug in later without reworking Phase 1.
