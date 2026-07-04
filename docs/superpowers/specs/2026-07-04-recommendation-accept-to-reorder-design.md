# Recommendation Accept → Reorder List (fix the accept-before-payment dead-end) — Design

**Date:** 2026-07-04
**Status:** Approved (brainstormed with Glen 2026-07-04)
**Repo:** deploy-chat
**Fixes a fast-follow from:** #572 (doctor continuity tooling C). Touches the patient-portal recommendation accept flow only.

## Problem

C's patient-side "Add to my order" (`api_portal_recommendation_accept`) currently, on the tap: re-prices the items, **mints a QBO invoice + Stripe session up front** (`_portal_reorder_checkout`), and immediately flips the recommendation to `accepted`. Two defects follow:

1. **Dead-end:** the status flip removes the card, so a patient who abandons the Stripe checkout has **no in-portal way to re-initiate the order** they wanted — a silent leak at the exact conversion the recommend loop exists to drive.
2. **Duplicate invoice:** the up-front invoice means a replay/re-tap creates a second QBO invoice + Stripe session. (A prior fix made accept reject replays of a non-`sent` rec, but the root cause — minting the invoice at the tap — remains.)

Portal-reorder payments **do not sync back** (these orders sit unpaid until later reconciliation), so there is no clean "payment succeeded" hook to flip status on. The fix is therefore to **not create an invoice at the accept tap at all.**

## The fix (approach A: defer the invoice to the normal reorder checkout)

Accepting a recommendation should mean "the doctor added this remedy to your reorder list" — not "buy it now." The purchase happens through the patient's existing, retryable reorder checkout.

- **Accept no longer mints an invoice or Stripe session.** `api_portal_recommendation_accept` stops calling `_portal_reorder_checkout`. It marks the recommendation `accepted` and returns success (no `stripe_url`).
- **Accepted-recommendation items merge into the patient's reorder module.** The portal render (`reorder_items`) includes an `accepted` recommendation's items, **deduped by slug** against the existing reorder set. The recommendation card (shown while `sent`, with "Add to my order" / "No thanks") converts, once accepted, to a brief "Added to your reorder list below" confirmation (or disappears), with the items now present in the reorder module.
- **Purchase is the existing normal reorder checkout** (`/api/portal/<token>/checkout`) — one invoice minted there, like any reorder, fully retryable. If the patient abandons, the items remain in the reorder list to check out again. **No dead-end, no duplicate** (nothing is minted at the tap).
- **Accepted items persist in the ongoing reorder list** (Glen's call) — the doctor's recommendation becomes part of what the patient can reorder going forward, alongside their other remedies.

## Status model (unchanged shape, corrected meaning)

`sent` (card with Add / No-thanks) → `accepted` (**items surfaced in the reorder module**; no invoice) → `dismissed` (removed). "Accepted" now means "chosen / added to reorder list," decoupled from payment. The prior replay guard (reject accept of a non-`sent` rec) stays and is now moot for duplication (nothing is minted), but harmless and kept for a clean single accept.

## Components / files

- **`app.py` `api_portal_recommendation_accept`:** remove the `_portal_reorder_checkout` call and the `stripe_url`; keep the token-scoped identity resolution and the `status == 'sent'` guard; set `accepted`; return `{ok, accepted:true}` (no order/invoice). (The now-unused member-pricing call at accept time is removed — pricing happens at the real checkout.)
- **The portal render** (the handler that builds the client-portal payload, where `reorder_items` and `recommendation` are assembled): merge an `accepted` recommendation's items into `reorder_items` (deduped by slug), and only surface the recommendation **card** for `status == 'sent'`.
- **`static/client-portal.html`:** the "Add to my order" handler POSTs accept (no redirect to Stripe), then refreshes the portal so the items appear in the reorder module and the card converts to the confirmation; "No thanks" dismiss unchanged.

## Testing

- **No invoice at accept:** accepting a `sent` recommendation creates **no** QBO invoice / order / Stripe session (assert the order count is unchanged and no `stripe_url` is returned), and flips status to `accepted`.
- **Items land in reorder:** after accept, the portal payload's `reorder_items` contains the recommended items (deduped by slug — a recommended slug already in reorder isn't doubled).
- **Abandon-safe / no dead-end:** the accepted items remain in `reorder_items` across a fresh portal load (nothing consumed them), so the patient can still check out.
- **No duplicate:** two accept POSTs don't create two invoices (there are none) and the second is rejected by the `!= 'sent'` guard.
- **Purchase path intact:** checking out the reorder module (`/api/portal/<token>/checkout`) with the accepted items still prices at member pricing and mints exactly one invoice (the existing behavior, unchanged).
- **Card visibility:** card shows only while `sent`; after `accept` it's the confirmation / gone; `dismiss` removes it.
- Patient-side scoping unchanged (identity from token, never body).

## Out of scope

- Fixing the normal reorder's own unpaid-until-reconciled payment behavior (pre-existing, applies to all reorders — not this fix).
- The other C fast-follows (atomic-accept TOCTOU is now moot since nothing is minted at accept; triage roster; consent back-fill; narrative diff).
