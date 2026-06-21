# Stripe Webhook for $1 Biofield Trial Fulfillment

**Date:** 2026-06-20
**Status:** Approved (design); ready for implementation plan
**Repo:** deploy-chat (Flask, Render `glen-knowledge-chat`, illtowell.com)
**Parent:** Follow-on to Begin #4b (the $1 Biofield-unlock trial -> $99/mo membership). A real $1 end-to-end test (2026-06-20) found that the membership is created ONLY by the Stripe success-redirect to `/begin/checkout-return`; a customer who closes the tab after paying is charged with no membership (stuck on "Your top match is being finalized"). This adds a Stripe `checkout.session.completed` webhook so the membership is created server-side, independent of the redirect.

---

## Problem

The #4b $1 trial creates the paid membership (a `subscriptions` billing row + a `memberships` access grant) inside `begin_checkout_return` (app.py ~4901-4947), which only runs when the browser follows Stripe's success-redirect. If that redirect never completes (the customer closes the tab, a network blip, a slow client), Stripe still captured the $1 but the membership is never created: the customer is charged and gets nothing, and their reveal stays blurred forever. This was reproduced live (a real $1 payment whose membership was missing until the return handler was re-run manually).

## Goal

Create the $1-trial membership from whichever of two independent triggers arrives first - the existing success-redirect OR a new Stripe `checkout.session.completed` webhook - idempotently, so a closed tab never leaves a paid customer without a membership. Scope: the `biofield_trial` checkout kind only.

## Scope

Extract the `biofield_trial` membership-creation block from `begin_checkout_return` into a shared, idempotent `_fulfill_biofield_trial(session_id)`; add a `POST /webhook/stripe` endpoint that calls it on a `checkout.session.completed` event; add a `stripe_pay.verify_webhook` signature helper.

**Out of scope:** webhook fulfillment for the other checkout kinds (reorder/retail orders, subscriptions, studio-claim) - they share the same redirect-dependency but are a later increment once this webhook infrastructure exists; any change to the $1 checkout creation, the pricing/billing engine, the daily charge cron, or the reveal.

---

## Confirmed decisions (Glen, 2026-06-20)

- **Scope: the $1 Biofield trial (`biofield_trial`) only** - the critical live-money case found broken. Other kinds become a follow-on using the same webhook infrastructure.
- **Shared idempotent fulfillment:** the redirect and the webhook call the SAME function; the existing `biofield_trial_grants(session_id PK)` claim-then-create marker dedupes, so exactly one membership is created regardless of which (or both) fire.
- **Security via independent re-fetch:** `_fulfill_biofield_trial` re-fetches the session + PaymentIntent from Stripe and only proceeds on a genuinely paid/succeeded payment with a vaulted card - so correctness holds even before the signing secret is configured; signature verification (when `STRIPE_WEBHOOK_SECRET` is set) is defense-in-depth.
- No emoji, no em dashes.

---

## Architecture

### 1. Shared fulfillment - `_fulfill_biofield_trial(session_id) -> dict` (app.py)
Extract the existing `biofield_trial` branch of `begin_checkout_return` (app.py ~4901-4947) verbatim into this function. It:
- `get_session(session_id)`; read `metadata.kind`, `metadata.email`, `metadata.token`; return `{ok:False, reason:"not_trial"}` if `kind != "biofield_trial"` (so the webhook can call it for any session and it self-filters).
- Confirm the payment: read the PaymentIntent (`get_payment_intent`); proceed only when `status == "succeeded"` with a `customer` and `payment_method` (the vaulted card). Else `{ok:False, reason:"unpaid"}` - no membership.
- Claim-then-create idempotently: `INSERT OR IGNORE INTO biofield_trial_grants (session_id, email, granted_at)` + commit FIRST; only when `rowcount == 1` create the `subscriptions` membership ($9900, next +1mo) + `_grant_membership(email, 31, "biofield_trial")` + mint the `membership_cancel` auth_token. If the marker was already claimed -> `{ok:True, already:True}` (the other trigger created it).
- Best-effort, never raises into callers; returns `{ok, created|already, email}`.
`begin_checkout_return`'s `biofield_trial` branch becomes a call to this function, then the existing redirect to `/begin/biofield/<token>` - behavior-identical to today.

### 2. The webhook - `POST /webhook/stripe` (app.py)
- Read the raw request body + the `Stripe-Signature` header.
- If `STRIPE_WEBHOOK_SECRET` is set: `event = stripe_pay.verify_webhook(body, sig, secret)`; on `None` (bad/missing/stale signature) return `400`. If the secret is unset: parse the JSON body directly (signature check skipped; the re-fetch in fulfillment is the safety guarantee).
- If `event["type"] == "checkout.session.completed"`: pull `event["data"]["object"]["id"]` and call `_fulfill_biofield_trial(session_id)`. Any other event type -> 200 no-op.
- Return `200` for anything handled (ignored type, fulfilled, or already-claimed); `500` only on an unexpected exception during fulfillment - so Stripe's automatic retry is a free, idempotent-safe net.

### 3. Signature helper - `stripe_pay.verify_webhook(payload, sig_header, secret, tolerance=300) -> dict|None`
Implement Stripe's standard scheme: parse `t=<ts>,v1=<sig>` from `sig_header`; compute `HMAC_SHA256(secret, f"{t}.{payload}")` (payload = the raw bytes); constant-time compare to `v1`; reject if the timestamp is older than `tolerance` seconds. Return the parsed JSON event dict on success, `None` on any failure. Pure, no network.

### Reuse / untouched
- Stripe: `get_session`, `get_payment_intent`; `_grant_membership`, `subscriptions.create_membership`, the `biofield_trial_grants` marker, the `membership_cancel` token mint - all reused via the extracted function.
- The $1 checkout creation (`begin_biofield_unlock_checkout`), the pricing/billing engine, the daily charge cron, the reveal render - untouched.
- Mirrors the existing webhook route pattern (`/webhook/practice-better`, `/webhook/scoreapp`, `/webhook/groovekart`).

---

## Data flow
1. Customer pays the $1. Stripe fires the browser **redirect** to `/begin/checkout-return?...session_id=X` AND (async) the **`checkout.session.completed` webhook** to `/webhook/stripe`.
2. Whichever arrives first calls `_fulfill_biofield_trial(X)`; it claims the `biofield_trial_grants(X)` marker (commit first) and only the winner creates the subscription + grant. The other no-ops.
3. Exactly one membership, regardless of redirect, webhook, or both (in any order).
4. The tab-closed customer is now covered by the webhook; deeper remedies release on their next reveal load.

## Error handling
- Bad/missing/stale signature (when the secret is set) -> 400, no fulfillment.
- A non-`biofield_trial` session, or a non-`checkout.session.completed` event -> 200 no-op.
- A non-succeeded PI -> no membership (the re-fetch gates it); 200.
- An unexpected exception during fulfillment -> 500 (Stripe retries; the marker makes retries idempotent).
- The redirect path (`begin_checkout_return`) stays best-effort / never-500 and behavior-identical; the extraction does not change its outcome.
- A forged webhook with a fake session id -> `get_session` fails or returns unpaid -> no membership.

## Testing
`tests/test_stripe_webhook.py` (+ the existing `tests/test_biofield_trial.py` must stay green - the redirect path is behavior-identical):
- **Fulfillment:** `_fulfill_biofield_trial` on a mocked paid `biofield_trial` session -> a `subscriptions` membership (9900, +1mo) AND a `memberships` grant exist; a second call -> no second row (marker idempotent); a non-succeeded PI -> nothing created; a non-`biofield_trial` session -> `{ok:False, reason:"not_trial"}`, nothing created.
- **Webhook:** a `checkout.session.completed` event with a paid trial session -> 200 + membership created (mock `_fulfill_biofield_trial` or the Stripe layer); an unhandled event type -> 200, no fulfillment; with `STRIPE_WEBHOOK_SECRET` set, a tampered/absent signature -> 400; a correctly-signed event -> 200.
- **Signature:** `verify_webhook` accepts a correctly-signed payload, rejects a tampered body, a wrong secret, and a stale timestamp.
- **Idempotency vs redirect:** calling the redirect handler and the webhook for the same session creates exactly one membership.
- Stripe mocked throughout (no live calls); tmp `LOG_DB`; init the subscriptions + memberships + biofield_trial_grants + auth_tokens tables. No emoji; no em dashes.

## Notes
- **No new flag.** The webhook endpoint is inert until Stripe is configured to deliver events to it. Go-live (documented config step): in the Stripe dashboard add a webhook endpoint -> `https://illtowell.com/webhook/stripe`, subscribe to `checkout.session.completed`, copy the signing secret into Doppler as `STRIPE_WEBHOOK_SECRET`.
- Until `STRIPE_WEBHOOK_SECRET` is set the endpoint still fulfills safely (signature check skipped; the re-fetch is the guarantee), but configuring it is what makes Stripe actually send the events and enables signature verification.
- This closes the live-money liability from the #4b $1 trial. The same webhook can later fulfill the other checkout kinds (reorder/retail/subscriptions/studio-claim) once each is made idempotent. Ties to [[project_ascension_pricing_model]].
