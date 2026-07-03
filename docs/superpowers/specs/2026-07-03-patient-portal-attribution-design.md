# Patient-Portal Column Attribution — Design Draft

**Date:** 2026-07-03
**Status:** design draft, pending Dr. Glen's approval of approach (A vs B)
**Context:** the last deferred column on the Clients-tab dispense table (PR #557 shipped Dispensed + Drop-shipped). "Patient portal" = digital sales made to the practitioner's patients through their own personal ordering page (client-portal reorders).

## The problem

Client-portal reorders land in `orders` under sources `('portal-reorder', 'reorder')` **with real line items** (`items_json`), so the *products* are captured. What is missing is the **practitioner attribution**: those orders are keyed by the client's **email only**, and the `client_portal` table has no `practitioner_id`. Portal publishing is console/owner-only, so nothing stamps "which practitioner serves this patient."

## What already exists

The system has exactly one practitioner↔client link today:
`client_belongs_to_practitioner(pid, email)` (dashboard/practitioner_portal.py) →
`True` iff that email has a row in `dispensary_orders` under the practitioner
(`WHERE practitioner_id=? AND lower(customer_email)=?`). This is the same definition the portal already uses for "your client."

## Two approaches

### Approach A — email-match via the existing dispensary link (no schema change) — RECOMMENDED first

Attribute a `portal-reorder`/`reorder` order to a practitioner when the order's email matches a client they already own via `dispensary_orders.customer_email`. Reuses the existing link and the system's own definition of "your client."

- **Pros:** zero schema change; ships immediately; consistent with `client_belongs_to_practitioner`; naturally covers the common case (a patient you drop-shipped to later reorders through their portal).
- **Cons/limits (documented in the UI):**
  - Only attributes patients who have a **dispensary-order history** under the practitioner — a referral-only or direct-portal client who was never drop-shipped won't attribute.
  - A patient linked to more than one practitioner (same `customer_email` under two `practitioner_id`s) counts for **each** — acceptable and rare, but note it (no single-owner guarantee).
  - Glen's own direct clients (never drop-shipped) don't attribute to a practitioner — correct (they're the house's).

### Approach B — explicit practitioner↔client link (schema change) — later

Add a durable owner: `practitioner_id` stamped on the client (a `client_practitioner(email, practitioner_id, since)` map, or a column), set at the moments a practitioner takes on a patient: portal publish/refer, dispensary onboarding, or a console "claim client" action; then backfill from `dispensary_orders`.

- **Pros:** precise and complete — attributes referral-only and direct-portal clients, resolves multi-practitioner ambiguity to one owner.
- **Cons:** schema + wiring at each attribution moment + a backfill; portal publish is console-only today, so it needs new UX to capture the practitioner at publish time.

## Recommendation

Ship **A** now (immediate, no-schema, reuses the existing link), and layer **B** when precise attribution of referral-only/direct clients becomes worth the schema + wiring. A's limits are the honest edges, not blockers; the UI keeps a short "based on your dispensary clients' portal reorders" note.

## Implementation sketch (Approach A)

New pure/defensive function in `dashboard/dispensary_stats.py`:

```
patient_portal_items(practitioner_id, *, db_path=None) -> {slug: units}
# 1. emails = SELECT DISTINCT lower(customer_email) FROM dispensary_orders WHERE practitioner_id=?
# 2. for orders WHERE lower(email) IN (emails) AND source IN ('portal-reorder','reorder'):
#      sum items_json[slug] -> units   (reuse the existing per-item parser)
# never raises; degrades to {}.
```

Wire it as the third channel in `dispense_stats` (currently passes `{}` for patient_portal), so `rank_dispense_rows(dispensed, dropshipped, patient_portal)` fills the column and Total becomes all three. UI: replace the `soon` Patient-portal cell with the real value; keep a one-line note on the attribution basis. Tests: a seeded portal-reorder order for a dispensary-linked email attributes; an order for a non-client email does not; source-scoped (a non-portal order with the same email is ignored); independent channel degradation.

## Out of scope

Approach B (explicit link + backfill + publish-time capture). SQL for the `IN (emails)` list uses parameter binding (no interpolation); cap the email set defensively if a practitioner has an unusually large client list.
