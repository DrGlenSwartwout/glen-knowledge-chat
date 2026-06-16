# Spec: Role-aware "What's next" offer surface (portal upgrade block)

**Date:** 2026-06-16
**Status:** Approved (design) — pending implementation plan
**Feature:** #2 (integrated sales pages), first slice
**Scope chosen by Glen:** in-portal offer surface (not standalone sales pages); **single next rung** (linear ladder); **build the standalone group-join checkout** so both rungs are purchasable.

---

## Goal

Fill the portal's `upgrade` seam (left as `{enabled:false, placeholder:true}` in slice 1) with a role/tier-aware "What's next for you" surface: for each logged-in client, show the **one** next ladder rung they're eligible for, with a CTA that checks out through the existing pricing + Stripe backend.

Ladder (linear): free → **Live Group $99/mo** → **Biofield $300** → cert (deferred).

## Non-goals (YAGNI)

- Cert rung (enrollment runs through practitioner registration; no $ checkout) — future rung.
- The $149 second group tier — default $99, configurable later.
- Menu-of-all-eligible-rungs — single next rung this slice.
- Standalone `/offer/<x>` sales pages — that was option 2/3, not chosen.

---

## What already exists (reuse)

Confirmed by code map 2026-06-16 (file:line):
- **Seam:** `dashboard/portal_view.py:107` returns the `upgrade` stub; `static/client-portal.html:190-194` renders the quiet `.soon` placeholder.
- **Live group $99:** `dashboard/group_bundle.py` (`MEMBERSHIP_AMOUNT_CENTS = 9900`); membership modeled as a subscription via `dashboard/subscriptions.py` (`create_membership`, `active_memberships_by_email`). Today only sold *bundled* into retail checkout — **no standalone join checkout exists**.
- **Biofield $300:** `_price_biofield()` + `POST /biofield/checkout` (app.py:3010, 3039), dark behind `BIOFIELD_CHECKOUT_ENABLED`; `_has_paid_biofield(email)` (app.py:7799); readiness in `dashboard/biofield_store.py`.
- **Pricing/Stripe:** `dashboard/pricing.compute(...)`; `_stripe_checkout_url_for_retail(...)` (app.py:7005); biofield checkout is the reference pattern for a non-product offer → Stripe.
- **Subscriptions:** `dashboard/subscriptions.py` — `create_membership` (line 290), `active_memberships_by_email` (line 308), own due-charging scheduler; `SUBSCRIPTIONS_ENABLED`.

Eligibility signals exist but are **scattered**; there is no single "what's next for this person" function. That resolver is the one genuinely new piece of logic.

---

## Architecture

Mirror the slice-1 pattern: a small, pure, `cx`-based module that never imports `app`, so it unit-tests in isolation.

### 1. Offer catalog — `dashboard/portal_offers.py` (new)
A declarative, ordered list of ladder rungs. Each rung:
```
{ key, title, blurb, price_cents, period, cta_label, checkout_path, flag, owned(cx, email) }
```
Slice-1 rungs (ladder order):
| key | title | price | period | cta | checkout_path | owned(cx,email) |
|-----|-------|-------|--------|-----|---------------|-----------------|
| `live_group` | Join the Live Group | 9900 | /mo | Join | `/portal/offer/live-group/checkout` (new) | `subscriptions.active_memberships_by_email(cx,email)` non-empty |
| `biofield` | Causal Biofield Analysis | 30000 | one-time | Book | `/biofield/checkout` (existing) | biofield-paid check via `biofield_store` |

`owned(cx,email)` is a per-rung predicate composed from `cx`-based reuse (`subscriptions`, `biofield_store`) — no `app` import.

### 2. Eligibility resolver — `next_offers(cx, email, roles, *, enabled_flags) → [offer,…]`
Pure, `cx`-based, unit-testable. Returns the rungs that are **flag-on AND not already owned**, in ladder order. The view/page shows the **first** (single next rung); `[]` → block hidden. `enabled_flags` is a set/dict passed in by the caller (read from env at the route/view boundary) so the resolver itself stays pure and deterministic in tests.

### 3. Standalone group-join checkout (new route)
`POST /portal/offer/live-group/checkout` — resolves identity via the slice-1 seam (`resolve_identity`: session cookie or path token), starts a $99/mo membership by **reusing the subscriptions system's card-vault + `create_membership` path** (the same mechanism Subscribe&Grow uses), and returns a Stripe URL. Mirrors the `/biofield/checkout` shape.

**Implementation risk (flagged):** the exact subscription-start / card-vault reuse must be confirmed against `dashboard/subscriptions.py`. If there is no clean reuse for starting a membership's first charge + vaulting the card, the `live_group` rung ships behind its flag (dark) until that path exists — the surface still works for the biofield rung.

### 4. Wire into `get_portal_view`
Replace the `upgrade` stub (`portal_view.py:107`) with `{"enabled": True, "offer": <first eligible>}`, or `{"enabled": False}` when none eligible.

### 5. Portal page — `static/client-portal.html`
Turn the `.soon` placeholder (lines 190–194) into a real "What's next for you" card: title, blurb, price/period, and a CTA button that POSTs to the rung's `checkout_path` and redirects to the returned Stripe URL (mirrors the existing reorder button).

### 6. Flags & graceful empty
- Master `PORTAL_OFFERS_ENABLED` gates the whole surface (ships dark; flip when ready — same pattern as `CLIENT_LOGIN_ENABLED`).
- Each rung also respects its own existing flag: `biofield` → `BIOFIELD_CHECKOUT_ENABLED`; `live_group` → subscriptions enabled (`SUBSCRIPTIONS_ENABLED`) + group-join available.
- Master off, or no rung flag-on, or all rungs owned → the block does not render.

---

## Data flow

1. `get_portal_view` → `next_offers(cx, email, roles, enabled_flags)` → first eligible rung → `upgrade` block.
2. Page renders the rung's CTA → POST `checkout_path` → resolves identity → starts checkout (group membership or biofield) → Stripe URL → redirect.

## Error handling

- No eligible rung (all owned / flags off) → block hidden, never errors.
- Checkout unavailable (Stripe inactive) → friendly "temporarily unavailable, reach out" message (mirror existing checkout error paths).
- Identity unresolved on a checkout POST → 404, same as the other portal endpoints.

## Testing

- **Unit (`next_offers`):** linear order; group hidden when membership active; biofield hidden when paid; flag-off rungs excluded; returns first eligible; returns `[]` when none. Injected `enabled_flags`, seeded `cx`.
- **Route:** group-join checkout (session + token identity → Stripe URL, monkeypatched subscriptions/Stripe); biofield rung already covered by existing tests.
- **Integration:** `/api/portal/<token>/view` upgrade block reflects the eligible rung for (a) a bare client (→ live_group), (b) a client with an active membership (→ biofield), (c) a client who owns both (→ no block).
- Invocation + isolation per the deploy-chat conventions (doppler + `deploy-chat311` venv + `DATA_DIR`; mock Supabase; tmp `DATA_DIR`). Work in the session worktree.

## Definition of done

- A logged-in client sees exactly one eligible next rung in "What's next," priced, with a working CTA to Stripe; an in-group client sees the Biofield rung; a client who owns both sees no block.
- `next_offers` is unit-tested; the group-join checkout is route-tested.
- The whole surface is gated by `PORTAL_OFFERS_ENABLED` (dark by default); each rung respects its own flag.
- Full suite green.
