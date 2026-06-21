# Biofield $1 Trial: Deliver the One-Click Cancel Link

**Date:** 2026-06-20
**Status:** Approved (design); ready for implementation plan
**Repo:** deploy-chat (Flask, Render `glen-knowledge-chat`, illtowell.com)
**Parent:** Closes the last correctness gap on the #4b Biofield $1 -> $99/mo trial (`_fulfill_biofield_trial`, the Stripe webhook, and `/membership/cancel/<token>`).

---

## Problem

When a customer pays the $1 Biofield trial, `_fulfill_biofield_trial` creates the membership (a `subscriptions` row billing $99/mo + a 31-day access grant) and mints a `membership_cancel` auth token. But the plaintext token is **never delivered** - the fulfillment just returns and the caller redirects. So the one-click cancel route `/membership/cancel/<token>` is built and working, yet **unreachable**: a paying $99/mo member has no working self-cancel path. The grant also sends **no email at all** - no welcome, no receipt, no disclosure that the $1 converts to $99/mo.

Two related defects:
1. The cancel token is minted but the plaintext is discarded, so the member can never use it.
2. Even if delivered, the token's TTL is **60 days**, but the membership recurs monthly indefinitely - after 60 days the link would 404 and the member would again be stranded.

## Goal

At membership creation, send the member one email that (a) confirms the membership and discloses the $1 -> $99/mo auto-convert with the next-charge date, and (b) carries a working one-click cancel link `{PUBLIC_BASE_URL}/membership/cancel/<plaintext-token>`. Extend the cancel token's TTL so the link stays valid for the practical life of the membership. This makes the already-built `/membership/cancel/<token>` mechanism actually reachable.

## Scope

- A welcome/cancel email sent from inside `_fulfill_biofield_trial` (so it rides both the redirect and the webhook fulfillment paths, exactly once).
- Extend the `membership_cancel` token TTL from 60 days to a long window (a named constant).
- Tests covering: end-to-end mint -> email -> cancel; exactly-once across the idempotency boundary; email failure does not break fulfillment; the extended TTL.

**Out of scope:** a member portal / account page; rendering a cancel control on the reveal (a possible fast-follow second channel - the reveal cannot carry this same hashed token); per-monthly-cycle receipt emails with a fresh link (a future robustness improvement); any change to billing, the webhook signature path, or the cancel route's logic.

---

## Confirmed decisions (Glen, 2026-06-20)

- **Channel: email only** (the core fix). One emailed one-click cancel link, no login - the FTC "click-to-cancel" standard. A reveal-page cancel button is deferred.
- The email doubles as the missing **welcome/receipt** and the **auto-convert disclosure** (currently the grant sends nothing).
- **Extend the cancel-token TTL** so the emailed link outlives the first billing cycle (the membership recurs monthly with no other surfaced cancel path).
- Best-effort send: a failed email must never break membership creation.
- Generic greeting ("Aloha,") - no fragile name lookup in this pass.
- No emoji, no em dashes.

---

## Architecture

### 1. Cancel-token TTL constant - `app.py`
Add a module constant near the existing auth-token TTL config (`AUTH_TOKEN_TTL_MIN`, ~line 171):

```python
MEMBERSHIP_CANCEL_TTL_DAYS = 1095  # ~3 years: the emailed one-click cancel link must outlive the recurring membership
```

In `_fulfill_biofield_trial` (~line 4728), replace the hardcoded `timedelta(days=60)` in the `membership_cancel` token insert with `timedelta(days=MEMBERSHIP_CANCEL_TTL_DAYS)`. Nothing else about the mint changes; `/membership/cancel/<token>` already enforces `expires_at`, so the longer window flows through unchanged.

### 2. Welcome/cancel email - `_fulfill_biofield_trial` (`app.py`)
After the token is inserted and committed (still inside the `if claimed:` block, so it fires only for the caller that won the idempotency claim - exactly once across redirect + webhook), send the email best-effort.

- Capture the next-charge date into a local when the membership is created: `next_charge = _bt_subs.add_months(_bt_dt.date.today().isoformat(), 1)` (reuse the value already passed to `create_membership`).
- Build the cancel URL: `cancel_url = f"{PUBLIC_BASE_URL}/membership/cancel/{cancel_tok}"` (plaintext token, used only here).
- Send via the existing `_send_inquiry_email(to_email, subject, body)` (per-recipient SMTP, returns bool, never raises; prints in dev when SMTP env is unset).
- Guard: only send when `bt_email` and `PUBLIC_BASE_URL` are both present; otherwise log and skip (never emit a broken link).
- Wrap the whole send in its own try/except so a failure logs and is swallowed - the membership must already be committed and must not be undone by an email error. (The token commit happens before the send, so the cancel link is valid regardless of send outcome.)

The send is additive: the existing return values (`{"ok": True, "created": True, ...}` etc.) are unchanged.

### 3. Email content (provisional copy; BNSN later)
Plain-text body (no emoji, no em dashes):

```
Subject: You're in - your membership is active

Aloha,

Your $1 unlocked your full Biofield Analysis and started your membership.
Your first monthly payment of $99 will run on {next_charge}. Everything
stays unlocked in the meantime, and you can order your matched remedies
anytime.

No pressure, ever. If you want to cancel before your first payment, it is
one click, no charge, no reply needed:

{cancel_url}

In wellness,
Dr. Glen and Rae
```

`{next_charge}` is the ISO date (`add_months(today, 1)`); `{cancel_url}` is the tokened link. The subject's hyphen is a plain hyphen, not an em dash.

### Reuse / untouched
- `_send_inquiry_email` (SMTP sender), `_hash_token`, `auth_tokens`, the `biofield_trial_grants` idempotency marker, `create_membership` / `_grant_membership`, `add_months`.
- Untouched: `/membership/cancel/<token>` route logic, the Stripe webhook (`/webhook/stripe`) and its signature verification, `begin_checkout_return`, billing/pricing, the cron auto-charge.

---

## Data flow
1. Customer pays the $1 trial; Stripe redirect (or the webhook, whichever lands first) calls `_fulfill_biofield_trial(session_id)`.
2. The caller that wins the `biofield_trial_grants` claim creates the membership + access grant, mints the `membership_cancel` token (now ~3-year TTL), commits.
3. That same caller sends one welcome/cancel email containing `{PUBLIC_BASE_URL}/membership/cancel/<token>` and the next-charge date. The losing caller (`already`) sends nothing.
4. The member clicks the link any time before they want to keep paying; `/membership/cancel/<token>` validates the (unexpired) token and sets their active `kind='membership'` subscription to `cancelled`.

## Error handling
- Email send wrapped in try/except inside the claimed block; failure logs `[biofield-trial] welcome-email failed` and is swallowed (membership stays created, token stays valid).
- Missing `bt_email` or `PUBLIC_BASE_URL` -> skip the send (log), never emit a broken link.
- Idempotency unchanged: the losing caller returns at the `if not claimed` guard before any membership/email work, so no second email.
- TTL is long but finite; the route still rejects an expired token (no behavior change there).

## Testing
`tests/test_membership_cancel_email.py` (keep `test_biofield_trial` green):
- **End-to-end:** a fulfilled trial sends exactly one email whose body contains `/membership/cancel/<token>`; extracting that token and GETting the route cancels the active membership subscription (status -> `cancelled`). Stripe (`get_session` / `get_payment_intent`) mocked to a succeeded paid trial; `_send_inquiry_email` monkeypatched to capture (to, subject, body).
- **Exactly once:** calling `_fulfill_biofield_trial` twice for the same `session_id` sends exactly one email (second call hits the `already` path).
- **Send failure is non-fatal:** `_send_inquiry_email` monkeypatched to raise -> fulfillment still returns created and the membership rows + cancel token still exist.
- **Extended TTL:** the stored `membership_cancel` token's `expires_at` is more than 60 days out (assert against `MEMBERSHIP_CANCEL_TTL_DAYS`).
- LLM/Stripe/SMTP mocked; tmp `LOG_DB` via `DATA_DIR`; no emoji; no em dashes.

Run: `doppler run -p remedy-match -c prd -- env DATA_DIR=$(mktemp -d) ~/.venvs/deploy-chat311/bin/python -m pytest tests/test_membership_cancel_email.py tests/test_biofield_trial.py -v`

## Notes
- This is **live money**: #4b is enabled in prod (`BIOFIELD_TRIAL_ENABLED=true`). Shipping this closes the trust gap where a charged member could not self-cancel. No new flag - the change only adds an email + lengthens a token, both safe to ship on merge.
- A reveal-page "Cancel my membership" button (a second, tokenless channel authed by the reveal magic-link) remains a possible fast-follow if Glen wants in-context cancel as well.
- Ties to [[project_begin_funnel]] (#4b trial + webhook).
