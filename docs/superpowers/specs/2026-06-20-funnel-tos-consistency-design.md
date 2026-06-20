# Funnel ToS Consistency

**Date:** 2026-06-20
**Status:** Approved (design); ready for implementation plan
**Repo:** deploy-chat (Flask, Render `glen-knowledge-chat`, illtowell.com)
**Origin:** A 2026-06-20 audit (during Begin #4a) found that several actionable funnel surfaces deliver individualized advice / transact / grant member benefits WITHOUT requiring ToS-membership, while the sensitive ones (funnel checkout, dispensary, studio claim) already gate. This closes the gaps.

---

## Problem

ToS agreement = Tier-1 membership (`journey_state.tos_agreed_at`, checked via `is_member(session_id, email)`). It is meant to gate individualized advice, condition recommendations, ordering, and member-only benefits. Today it is enforced inconsistently: the hero, funnel checkout, dispensary patient actions, and the studio claim gate correctly, but **becoming an Ambassador, reordering, minting a referral code, and the post-purchase concierge do not**. A person can become an approved affiliate, place repeat orders, mint a referral code, and get individualized post-purchase advice without ever agreeing to the terms.

## Goal

Apply the existing, proven ToS gate to the four ungated **actionable** surface groups, reusing `is_member` + the `need_optin`/`OptinGate` inline-capture pattern (and a visible ToS checkbox on the affiliate form). Already-members pass through unchanged; non-members agree to ToS in place and proceed.

## Scope

Gate these four groups (in build priority):
1. **Affiliate / Ambassador** - `POST /affiliate/apply` (JSON, app.py ~6309) and `POST /affiliate/apply-form` (form, ~6225); `static/affiliate.html`.
2. **Reorder** - `POST /reorder/checkout` (~9891), `POST /reorder/subscribe` (~10013), `GET /api/reorder/items` (~9836); `static/reorder.html`.
3. **Referral** - `GET /api/referral/my-code` (~7999); the page(s) that call it.
4. **Concierge** - `POST /begin/concierge/chat` (~4277), `POST /begin/concierge/add` (~4348); `static/begin-concierge*.html` (the concierge page).

**Out of scope (deliberately left ungated):** journey-signal page-loads that fire `paid_fork` on view (`/begin/path`, `/begin/ascend`, `/begin/ascend/<slug>`) and the `voice-signal` transcript log (`/begin/match/voice-signal`). These are navigation micro-commitments, not transactions/advice; gating them only adds funnel friction, and their real destinations (affiliate signup, intake) are gated or external. Also out: any surface already gated (funnel checkout, dispensary, studio claim).

---

## Confirmed decisions (Glen, 2026-06-20)

- Gate the **actionable** surfaces only (affiliate, reorder, referral, concierge); leave the journey-signal page-loads.
- **Affiliate form gets a visible required ToS checkbox** (agreeing to terms as you become an Ambassador is the natural, visible moment); the other surfaces use the `OptinGate` modal.
- **No feature flag** - these are live correctness fixes. Accept that this changes live behavior for non-members (e.g., a returning buyer who never agreed ToS hits a one-time opt-in on their next reorder). Stage surface-by-surface as each merges; test carefully.
- No emoji, no em dashes.

---

## Architecture

### The gate (uniform)
Each gated server action adds, before it transacts/advises/grants:
```python
_sid = request.cookies.get("amg_session", "")
if not is_member(_sid, email):
    return jsonify({"ok": False, "need_optin": True,
                    "error": "Please agree to our terms to continue."}), 403
```
`email` is the surface's already-resolved identity (the reorder cookie email, the referral caller's email, the concierge order email, the affiliate apply email). The check is **fail-safe**: `is_member` already returns False on internal error, so a failure gates rather than bypasses. Already-members return True and proceed unchanged.

### The inline capture (uniform, front-end)
The page that calls a gated endpoint loads `static/optin-gate.js` (`window.OptinGate`) and, on a `{need_optin:true}` response, calls `OptinGate.onCheckout(data, {... onSuccess: retry})` (or `OptinGate.show(...)`). That renders the name+email+ToS capture, fires `unlock('tos', {email, tos:true})` (-> `record_unlock` sets `tos_agreed_at`), then retries the original action. This is the exact pattern `begin-buy.html` / `studio-claim.html` / `practitioner-client.html` already use. For surfaces whose page does not yet load `optin-gate.js`, add the script tag + the `need_optin` handler.

### Per-surface specifics

**1. Affiliate / Ambassador.**
- `static/affiliate.html`: add a **required ToS checkbox** ("I agree to the Terms" linking the T&C) to the signup form. The form submit must include `tos=true`.
- `POST /affiliate/apply-form` (~6225) and `POST /affiliate/apply` (~6309): before creating the affiliate, require membership OR an agreed ToS in this request: `if not is_member(_sid, email) and not bool(data.get("tos")): -> need_optin (JSON) / re-render the form with the ToS prompt (form)`. When `tos` is present and the email is not yet a member, set it: `record_unlock(cx, session_id=_sid or synthetic, trigger="tos", email=email, tos=True)` BEFORE creating the affiliate, so the affiliate is always a member. Keep the existing auto-approve behavior otherwise.
- Net: you cannot become an Ambassador without agreeing/being a member.

**2. Reorder.** `/reorder/checkout`, `/reorder/subscribe`, `/api/reorder/items`: resolve the reorder email (the existing `rm_reorder_email` cookie / `_reorder_email_from_cookie()` path), add the `is_member` gate; `static/reorder.html` handles `need_optin` via OptinGate (capturing ToS for that email) and retries.

**3. Referral.** `GET /api/referral/my-code`: resolve the caller email (auth user OR reorder cookie, as today), add the `is_member` gate; the calling page handles `need_optin`.

**4. Concierge.** `/begin/concierge/chat` and `/begin/concierge/add`: resolve the order/session email, add the `is_member` gate; the concierge page handles `need_optin` (the SSE chat uses `OptinGate.onSSE`, mirroring `/begin/match/chat`).

### Reuse / untouched
- `is_member` (app.py ~341), `record_unlock`/`unlock('tos')`, `static/optin-gate.js`, the `need_optin` convention.
- Untouched: the already-gated surfaces, the journey-signal page-loads, the pricing/Stripe/order internals (the gate sits in FRONT of them; the order logic itself is unchanged).

---

## Data flow (per gated action)
1. Non-member triggers the action -> server `is_member` false -> `{need_optin:true}` 403 (or the affiliate form rejects without the ToS box).
2. The page shows the OptinGate (or the form's ToS checkbox) -> visitor agrees -> `unlock('tos', {email, tos:true})` -> `tos_agreed_at` set -> member.
3. The action retries (OptinGate `onSuccess`) / the form resubmits -> now `is_member` true -> proceeds normally.
4. A member triggers the action -> passes straight through (no gate shown).

## Error handling
- The gate is fail-safe: `is_member` returns False on error -> gate (never bypass).
- Adding the gate must NOT change the success path for members (a member's order/code/chat behaves exactly as today).
- The affiliate form must reject a submit missing `tos` for a non-member with a clear inline message, not a 500.
- OptinGate failures (visitor closes it) simply leave the action un-performed; no partial state.

## Testing
Per surface, Flask-test-client:
- **Non-member -> gated:** affiliate apply/apply-form without ToS -> not created + 403/`need_optin` (or form re-prompt); reorder checkout/subscribe/items, referral my-code, concierge chat/add as a non-member -> 403 `need_optin`, NO order/code/advice produced.
- **ToS then proceeds:** after `unlock('tos')` (or affiliate form with `tos=true`), the same action succeeds (affiliate created + email now a member; reorder/referral/concierge proceed).
- **Member passes through:** a pre-member email -> the action succeeds with no gate.
- **Affiliate sets membership:** `apply` with `tos=true` for a non-member -> `is_member(email)` true afterward.
- deploy-chat test isolation (tmp `LOG_DB`; `init_journey_tables`; mock GHL onboarding on free-tier transition; mock Stripe/QBO where the order path requires it, or assert the gate returns BEFORE those are called). No emoji; no em dashes.
- Front-end (OptinGate rendering, the affiliate checkbox, retry) = manual visual pass (state it).

## Notes
- **Live behavior change, no flag.** Each surface, once merged, immediately requires ToS from non-members. Stage by merging/deploying surface-by-surface (affiliate first) and watch. `main` auto-deploys.
- The affiliate form ToS checkbox is the only new UI; everything else reuses OptinGate.
- This is the funnel-wide enforcement of the same consent line #1 established; it does not change WHAT ToS means, only WHERE it is enforced.
- Build order = affiliate (critical) -> reorder -> referral -> concierge; each an independent, testable increment.
