# USPS Shipping Automation & Delivery Tracking — Design Spec

**Date:** 2026-07-18
**Status:** Draft for review
**Author:** Glen + Claude

## Problem / Motivation

Orders ship via USPS, with labels created **by hand in Click-N-Ship**. Two gaps:

1. **No delivery signal** back into the system. This blocks the membership rule (start the coaching month from delivery, not payment) and means customers aren't reliably told their tracking (USPS's own recipient emails are unreliable).
2. **Manual label creation** is slow and error-prone. The intended fix — a third-party shipping platform — is stuck on a Stripe business verification that has failed ~10 times.

Both can be solved **direct to USPS**, no third party, as a bridge until the new platform's API is available. Click-N-Ship itself has no API, but USPS's developer APIs (developer.usps.com) do everything Click-N-Ship does, programmatically.

## Goal

Capture USPS delivery data and (later) create labels automatically from orders, feeding: the membership-start gate, reliable customer tracking notifications, and eventually hands-off fulfillment — direct to USPS.

## Staging

Deliberately two stages so Stage 1 ships value immediately with **no postage-payment setup**, while Stage 2 (which spends money) waits on a USPS payment account.

### Stage 1 — Delivery tracking (read-only, no money) — DO FIRST

Data in, no labels created, no postage spent.

**Sources (works today):**
- **Click-N-Ship confirmation emails** land in Rae's Gmail (`suerae1111@gmail.com`) from `noreply-ecns@usps.com` on every label. Each contains: tracking number (with USPS track link), **scheduled delivery date**, recipient name + full address, ship date, service, cost. (Verified 2026-07-18 against a real confirmation.)
- **USPS Tracking API** (developer.usps.com, free) — given a tracking number, returns current status and the **actual delivered** date.

**Pipeline:**
1. Poll Gmail for new `noreply-ecns@usps.com` Click-N-Ship confirmations → parse tracking #, recipient (name+address), ship date, scheduled delivery date, service.
2. **Match shipment → order** by recipient **name + shipping address** (the email carries no order # or customer email — the "Email" field is Rae's account, not the customer's). Fuzzy-match with a confidence threshold; unmatched shipments go to a review queue, never guessed.
3. Query the USPS Tracking API per tracking # for delivered status/date (until delivered or aged out).
4. Persist per order: tracking #, service, ship date, scheduled delivery, delivered date, raw status.

**Consumers:**
- **Membership-start gate** (ties into [[project_membership_on_invoice]]): start the coaching month from **confirmed delivery** when known; else scheduled delivery; else the existing **5-day + 1-month (35-day) fallback**. Never shorten an already-granted window.
- **Customer tracking email** — send the tracking number to the customer from our own system (reliable), replacing the flaky USPS recipient email.

### Stage 2 — Label automation (write, spends postage) — GATED ON USPS PAYMENT ACCOUNT

**Sources:** USPS **Addresses API** (validate/standardize the ship-to), **Domestic Prices API** (rate/service), **Domestic Labels API** (create the label → returns tracking # + label PDF).

**Prerequisite (Glen-owned setup, not code):** a funded USPS **Enterprise Payment System (EPS)** account (+ Business Customer Gateway CRID/MID). This is what pays for postage on an API-created label. Separate from Stripe.

**Pipeline:** from a ready-to-ship order → validate address → pick service (default Priority Mail, weight/size from the order's packing) → create label via API → store label PDF + tracking # on the order → tracking # flows straight into Stage 1's tracking record (no email harvest needed once this is live).

**Money guardrails (mandatory):** per-label cost ceiling; a create-label action is operator-confirmed or tightly scoped (never a blind loop over orders); idempotency so a retry never buys two labels for one order; a daily spend cap; full audit log of every label purchased.

## Architecture

- **Credentials:** USPS OAuth2 Consumer Key + Secret in Doppler (`USPS_CONSUMER_KEY`, `USPS_CONSUMER_SECRET`); token fetched/cached per the USPS OAuth flow. No secrets in code or chat.
- **Home:** deploy-chat. A poller (folded into an existing cron, like the QBO-heal pattern — no new Render service) runs Stage 1; Stage 2's label creation is an operator-triggered action on an order.
- **Storage:** new shipment fields/table keyed by order (tracking #, service, ship/scheduled/delivered dates, status, label ref), so both stages write one shipment record per order.
- **Gmail access:** existing Gmail integration reads `suerae1111@gmail.com` Click-N-Ship confirmations (Stage 1 only; unnecessary once Stage 2 provides tracking directly).

## Out of scope (v1)

- The third-party shipping platform (blocked on Stripe verification — separate track).
- Non-USPS carriers.
- International labels/customs.
- Returns labels.

## Open decisions (for review)

1. **Match confidence:** how strict on name+address matching before auto-linking vs sending to a review queue? (Proposed: high-confidence auto-link, everything else queued.)
2. **Membership gate precedence:** confirmed delivery > scheduled delivery > 35-day fallback — confirm that ordering, and that we never shorten an already-active grant.
3. **Stage 2 timing:** build Stage 1 now and Stage 2 after the EPS account exists? (Proposed: yes.)
4. **Label service default:** Priority Mail always, or driven by order weight/value?

## Resolved decisions

| Decision | Choice |
|---|---|
| Carrier | USPS direct (developer.usps.com), no third party |
| Approach | Two stages: tracking read-only first, label automation second |
| Stage 1 tracking source | Click-N-Ship confirmation emails (Gmail) + USPS Tracking API |
| Stage 2 postage payment | USPS EPS account (Glen sets up) |
| Membership gate fallback | 5 days + 1 month (35-day) when no delivery date |
