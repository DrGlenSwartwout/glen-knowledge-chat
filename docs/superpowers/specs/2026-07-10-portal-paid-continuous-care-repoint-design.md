# Repoint program-page Paid tier at continuous-care — design

**Date:** 2026-07-10 · **Status:** design for review · **Author:** Glen + Claude

The client-portal membership program page (`/portal/<token>/program`, PR #771) currently points its **Paid** tier at the older `live-group` checkout, which is half-wired for provisioning (no access grant at join, no webhook fulfiller). This change repoints Paid at the **complete** continuous-care-monthly flow (immediate grant, webhook-backed, idempotent), via a new portal-token wrapper route, and ships it **dark** behind a new flag so going live stays a deliberate flip.

Background trace: continuous-care checkout = `POST /continuous-care/checkout` (`app.py:3395`), `mode=payment` immediate $99 charge + card vault, fulfilled by `_fulfill_continuous_care_monthly` (webhook `app.py:23321` + return `app.py:3428`), idempotent via `continuous_care_grants(session_id PK)` (`app.py:8203`), grants membership immediately (`_grant_membership`, `app.py:8280`), term cap 6/12.

## Decisions (locked with Glen 2026-07-10)

1. **Wire it dark** behind a new OFF flag — repoint is a safe code change; Paid going live stays a one-line flip.
2. **Term = 12 months** default for the portal Join.
3. Paid keeps rendering **"coming soon"** until the new flag is flipped.

## Why a new route (continuous-care can't be pointed at directly)

The program page button (`static/portal-program.html` `startCheckout`) POSTs `<checkout_path>?token=<token>` with body `{}` and follows `j.stripe_url`. The existing `/continuous-care/checkout`:
- returns `{"url": ...}` not `{"stripe_url": ...}`,
- reads `email` from the POST body (no token/cookie identity resolution),
- requires `term_months in (6,12)` in the body.

So a token-only `{}` POST would fail as `invalid` and never redirect. A thin wrapper route bridges this.

## 1. New wrapper route — `POST /portal/offer/continuous-care/checkout` (app.py)

Mirror the existing live-group route (`portal_group_join_checkout`, ~`app.py:19567`):

- **Guard (all required, else `404 {"error":"not found"}`):** `_program_paid_live_enabled()` **and** `CONTINUOUS_CARE_MONTHLY_ENABLED` (module constant, `app.py:5092`) **and** `_STRIPE_ACTIVE`. This preserves the no-dead-buy-button invariant: the button is never live unless the whole downstream can actually complete.
- **Identity:** `token` from the URL path (route is `/portal/offer/continuous-care/checkout/<token>`? no — keep it tokenless-in-path and read `token` from `request.args`/body like the live-group route); `sess_cookie = request.cookies.get("rm_portal_session","")`; `ident = resolve_identity(cx, token=token, session_token=sess_cookie, client_login_enabled=_client_login_enabled())`; `404` if `ident is None`. Use `ident.email`. (Match the live-group route `portal_group_join_checkout` token handling exactly.)
- **Start checkout** with `term_months = 12` via the shared helper (below): `sess = _continuous_care_checkout_session(ident.email, 12)`, then `return jsonify({"ok": True, "stripe_url": sess.get("url")})`.
- On Stripe failure, mirror the live-group route: `502 {"error": "Could not start checkout. Please reach out and we'll help."}`.

Fulfillment is untouched — the session carries `metadata.kind = "continuous_care_monthly"`, so the existing `/continuous-care/return` handler and the `_fulfill_continuous_care_monthly` webhook + `continuous_care_grants` idempotency provision the member exactly as they do for the existing route.

## 2. Shared checkout-session helper (DRY, money-path safety)

Extract the Stripe-session construction currently inline in `continuous_care_checkout` (`app.py:3414-3421`) into:

Transcribe verbatim from the existing inline block (`app.py:3413-3423`) — do not reword the description or change URLs:

```
def _continuous_care_checkout_session(email, term_months):
    """Create the continuous-care Stripe checkout session (mode=payment, $99 now +
    card vault). Single source of truth for metadata / save_card / success+cancel URLs.
    Returns the session dict (use .get("url"))."""
    from dashboard import stripe_pay as _sp, prepay as _pp
    base = PUBLIC_BASE_URL.rstrip("/")
    return _sp.create_checkout_session(
        _pp.MONTHLY_ANCHOR_CENTS, customer_email=email,
        description=f"Remedy Match Continuous Care - {term_months} month (monthly)",
        metadata={"email": email, "kind": "continuous_care_monthly",
                  "term_months": str(term_months)},
        success_url=f"{base}/continuous-care/return?session_id={{CHECKOUT_SESSION_ID}}",
        cancel_url=f"{base}/",
        save_card=True)
```

Both `continuous_care_checkout` (existing) and the new wrapper call it. `continuous_care_checkout` keeps its own body/validation and its `{"ok":true,"url":...}` response; only the `create_checkout_session(...)` construction moves into the helper. This prevents the metadata / URLs from drifting between the two entry points on a money path. **The existing route's externally observable behavior must be unchanged** — verified by a test.

## 3. `dashboard/program_tiers.py` — Paid tier

- `checkout_path` → `"/portal/offer/continuous-care/checkout"` (was `/portal/offer/live-group/checkout`).
- **Rename** the `program_blocks(...)` parameter `paid_enabled` → `paid_live` (the endpoint is the only caller). Paid's `state` = `_state(paid_owned, paid_live)`. `family_enabled` is unchanged.
- Copy: add one honest note to the Paid benefits (or a `note` field) — "Billed $99 per month for 12 months; your first month is charged today." No em dashes, no ALL CAPS. `price_cents` stays `MEMBERSHIP_PRICE_CENTS` (9900), `period` stays `/mo`, `cta_label` stays `Join`, `cta_kind` stays `checkout_post`.

## 4. `api_portal_program` endpoint (app.py)

Replace the Paid gate input: compute `paid_live = _program_paid_live_enabled() and CONTINUOUS_CARE_MONTHLY_ENABLED` and pass `paid_live=paid_live` to `program_blocks(...)`. Gating on both here means the card never shows a live Join unless continuous-care is actually on (no dead button); `_STRIPE_ACTIVE` is additionally enforced at the wrapper route. So flipping `PROGRAM_PAID_LIVE_ENABLED` on (continuous-care already on in prod) makes Paid live.

## 5. New flag helper (app.py)

```
def _program_paid_live_enabled() -> bool:
    """Whether the program page's Paid tier is a live, sellable Join. Dark by default."""
    return os.environ.get("PROGRAM_PAID_LIVE_ENABLED", "").strip().lower() in (
        "1", "true", "yes", "on")
```

Durable flip = `doppler secrets set PROGRAM_PAID_LIVE_ENABLED=1 -p remedy-match -c prd` (Doppler is source; Render is pruned — see [[reference_prod_flags_deleted_not_off]]).

## 6. Tests

- **program_tiers:** Paid `state == "coming_soon"` when `paid_live=False`; `state == "available"` with `checkout_path == "/portal/offer/continuous-care/checkout"` and the term note present when `paid_live=True`; Free/Family unchanged.
- **wrapper route:** `404` when `PROGRAM_PAID_LIVE_ENABLED` off; with it on (+ continuous-care on, stripe mocked) and a valid seeded portal token → `200 {"ok":true,"stripe_url": ...}`; `404` for an unresolvable token. Mock `create_checkout_session` (or the shared helper) to avoid a live Stripe call.
- **shared helper regression:** the existing `POST /continuous-care/checkout` still returns `{"ok":true,"url":...}` for a valid `email`+`term_months` body (behavior unchanged after extraction).
- Run app-importing tests under `doppler run -p remedy-match -c dev -- python3 -m pytest ...`; pure `program_tiers` test runs bare.

## 7. Verification

- Headless render: with `PROGRAM_PAID_LIVE_ENABLED` off, Paid shows "coming soon"; with it on, Paid shows the "Join $99/mo" button whose `data-checkout` is the new route. ([[feedback_render_the_page_not_the_payload]])

## Out of scope

- No change to the continuous-care fulfillment, webhook, cron, or `continuous_care_grants`.
- No client-facing 6-vs-12 term choice (12 hardcoded for v1).
- Not removing the live-group route (leave it; just no longer referenced by the Paid tier).
- Not flipping `PROGRAM_PAID_LIVE_ENABLED` on — ships dark.
