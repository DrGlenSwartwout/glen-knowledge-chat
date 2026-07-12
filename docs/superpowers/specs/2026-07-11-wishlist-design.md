# Wishlist — Design Spec

**Date:** 2026-07-11
**Status:** Approved (design), pending spec review
**Feature flag:** `WISHLIST_ENABLED` (Doppler `remedy-match`/`prd`, default off)

## Goal

Let a visitor save products to a personal wishlist from any `/begin/product/<slug>`
page with **zero friction** (no login), and show that wishlist in the client
portal beside their familiar / previously-ordered remedies. The follow-on to the
Related Products project ([[project_related_products]]); an "Add to wishlist"
action can later live on related-product cards too.

## Non-goals (v1)

- Move-to-cart / reorder-flow integration, "buy all", quantities.
- Sharing a wishlist, multiple named lists, notifications / back-in-stock.
- Practitioner-side views. These are natural follow-ons, not v1.

## Identity model — session-first, merge on identify

A wishlist is **owned** by one of two key namespaces in a single table:
- `sess:<amg_session>` — an anonymous browser session (the `amg_session` cookie,
  already set site-wide, 1-year max-age).
- `email:<email>` — an identified person (lowercased email).

**Add/remove keys to whoever we currently are:** resolve an email first
(`get_authenticated_user(request)` → else the `rm_reorder_email` cookie); if one
exists use `email:<email>`, otherwise fall back to `sess:<amg_session>`. So a
returning customer with the reorder cookie saves straight to their email; an
anonymous browser saves to their session.

**Merge on identify.** `merge_wishlist(cx, session_id, email)` rekeys every
`sess:<session_id>` row to `email:<email>` (INSERT OR IGNORE into the email owner,
then DELETE the session rows — never overwrites an existing email entry, never
downgrades). It fires wherever a request newly ties a session to an email:
- portal load (`/portal/<token>` → token's email),
- the reorder magic-link handler (sets `rm_reorder_email`),
- checkout / prepay (email captured).
Idempotent and safe to call on every such request.

## Data model

LOG_DB table (created idempotently, like the other feature tables):
```sql
CREATE TABLE IF NOT EXISTS wishlist (
  owner      TEXT NOT NULL,   -- 'sess:<id>' or 'email:<addr>'
  slug       TEXT NOT NULL,
  added_at   TEXT NOT NULL DEFAULT (datetime('now')),
  PRIMARY KEY (owner, slug)
);
CREATE INDEX IF NOT EXISTS idx_wishlist_owner ON wishlist(owner);
```

## Endpoints

- `POST /begin/wishlist/toggle` — body `{slug}`. Resolves owner (email-or-session),
  toggles the row (add if absent, remove if present), returns `{saved: bool}`.
  Requires only the `amg_session` cookie (set if missing, as elsewhere).
- `GET /begin/wishlist` — returns the current owner's wishlist as
  `[{slug, name, price, url, image}]` (for the product page to mark saved state
  and, later, any "my wishlist" surface). Union of the session AND, if resolvable,
  the email owner (so a mid-session identify shows both without waiting for a merge).

## Product-page button

On `/begin/product/<slug>` (rendered by `begin-product.html`), add an "Add to
wishlist" toggle in the hero near the Order CTA. On load it checks saved state
(from `GET /begin/wishlist` or a `saved` flag in the page-data payload) and
reflects it (outline heart "Save" ↔ filled heart "Saved"). Click POSTs the toggle
and flips the state optimistically. Gated on `WISHLIST_ENABLED`.

## Portal display

In `client-portal.html`, a "Your wishlist" card placed beside the familiar /
previously-ordered remedies. The portal page-data (keyed by the portal token's
email) gains `wishlist: [{slug, name, price, url, image}]`, read from the
`email:<email>` owner after `merge_wishlist` has run for the load. Each item links
to `/begin/product/<slug>` and has a remove (×) that POSTs the toggle. The card is
hidden when the list is empty.

## Module boundaries (for testability)

`app.py` is not importable in the test env (pinecone), so the pure logic lives in
`dashboard/wishlist.py` and is unit-tested directly:
- `init_wishlist_table(cx)`
- `resolve_owner(email, session_id) -> str` (email wins; else session; else None)
- `toggle(cx, owner, slug) -> bool` (returns new saved state)
- `list_for(cx, owner) -> [slug]` and `list_union(cx, email, session_id) -> [slug]`
- `merge_wishlist(cx, session_id, email) -> int` (rows moved)
Card hydration (slug → {name,price,url,image}) reuses the same helper the related
section uses in `app.py`.

## Testing

- `resolve_owner`: email→`email:`, session-only→`sess:`, neither→None.
- `toggle`: add then remove round-trip; independent per owner.
- `merge_wishlist`: session rows move to email, dedup against existing email rows,
  session rows deleted, never downgrades an existing email entry; empty session = noop.
- `list_union`: de-duplicated union of session + email.
- Integration (flag on/off): toggle endpoint, page-data `saved` flag, portal payload.

## Rollout

Land dark behind `WISHLIST_ENABLED`. Verify locally (button toggles, session→email
merge on portal load, portal card renders), then flip the flag in Doppler.
Same two-deploy caveat as any flag flip ([[feedback_flag_flip_two_deploys]]).

## Open risks

- **Merge coverage:** if a merge trigger is missed, an anonymous list simply stays
  under the session (still usable on-device) until the next identifying request —
  degrades gracefully, no data loss.
- **Card hydration for retired slugs:** resolve `superseded_by` and drop inactive
  slugs when building cards, same as the related section.
