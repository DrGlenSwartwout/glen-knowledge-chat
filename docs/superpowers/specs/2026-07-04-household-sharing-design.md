# Household Communication Sharing & Routing (v1) — Design

**Date:** 2026-07-04
**Status:** Approved (brainstormed with Glen 2026-07-04)
**Repo:** deploy-chat

## Summary

The shipped household feature (#584) links a caregiver to member scan accounts and lets the caregiver **view** any member's scans. This adds the two-sided **consent + routing** layer on top:

- **Member controls sharing (info + comms):** each member can permit or deny sharing their scan info AND their communications with a caregiver. Denying removes them from the caregiver's view *and* stops any cc.
- **Caregiver controls cc:** the caregiver chooses which of their linked members they receive cc'd emails about.
- **Routing:** when a member gets a client-facing notification (report published, invoice sent), the caregiver is cc'd only when **both** switches are on — the member permits AND the caregiver has opted in.

Real use: Sasha (cat) has an inactive E4L email; Karin's active email should receive Sasha's report/invoice notifications. Karin is Sasha's caregiver and operates Sasha's portal, so she sets the permission for the dependent; the cc defaults on for pets.

## Scope

**v1 = the consent + cc layer + routing for reports & invoices.** Both toggles live on the existing `household_members` link. Ships behind `HOUSEHOLD_SHARING_ENABLED` (default OFF). Deferred: cc for other notification types (magic links, practitioner mail, chat), a full member↔multiple-caregiver consent matrix UI beyond the common single-caregiver case, digest/frequency controls, SMS routing.

## Data model

Two columns added to `household_members` (idempotent `ALTER TABLE`, mirroring the existing additive-column pattern):

- **`share_consent INTEGER DEFAULT 1`** — member-controlled. `1` = this member permits sharing their info + comms with this caregiver (the link's `primary_email`). Gates BOTH the caregiver's view and cc. Existing links migrate to `1` (no regression).
- **`cc_enabled INTEGER DEFAULT 0`** — caregiver-controlled. `1` = the caregiver wants cc'd emails about this member. The column default is `0`, but `add_member` sets it from the relationship (below); a one-time backfill sets it for the links that already exist.

**Relationship classification** (drives the cc default):
- `DEPENDENT_RELATIONSHIPS = {"child", "pet", "dependent", "charge", "caregiving-client"}` → cc default **ON**.
- Everything else (`spouse`, `adult-child`, `other-adult`, `""`, unknown) → cc default **OFF**.
- `is_dependent(relationship) -> bool`; `default_cc_for(relationship) -> 0|1`.

`share_consent` is default-ON-revocable regardless of relationship (an adult is shared by default but not cc'd by default; they can revoke sharing entirely).

## The view gate (retrofit onto #584)

- `can_view(cx, viewer, target)` now returns True for `viewer==target` OR a link `(primary=viewer, member=target)` **with `share_consent=1`**. A member who revokes (`share_consent=0`) is no longer viewable by that caregiver — the switcher drops them and `?member=` falls back to the caregiver's own view (existing fail-closed behavior).
- The portal's household list (the switcher) is built from a consent-filtered lookup `viewable_members_for(cx, primary)` (only `share_consent=1`). `members_for` (unfiltered, all columns incl. the two flags) stays for the owner console.
- `same_household(cx, a, b)` is **unchanged** — reassignment is owner-side data correction and must stay consent-independent.
- Because `share_consent` defaults to `1`, this is a no-op for every existing/consented link; only an explicit revoke changes behavior.

## CC routing

- `cc_recipients_for(cx, member_email) -> [caregiver_email, ...]` = `SELECT primary_email FROM household_members WHERE member_email=? AND share_consent=1 AND cc_enabled=1`. The AND is the two-switch rule.
- **Hook points (client-facing notifications only):** the report-ready / portal-report notification (`app.py:_send_reveal_link`, and the publish-portal welcome send) and the invoice send (`dashboard/orders._send_invoice_exec`). At each, after the primary `to_email` (the member) is resolved, compute `cc = cc_recipients_for(cx, to_email)` and add those addresses to the SMTP recipient list AND a `Cc:` header. Best-effort — a cc-resolution failure never blocks the primary send. Gated by `HOUSEHOLD_SHARING_ENABLED`.
- Out of scope: magic-link, practitioner, affiliate, chat, and other non-report/invoice sends.

## Control surfaces

- **Member's own portal** (`client-portal.html`): a "Sharing" control — "Share my scans & messages with **[caregiver]**" toggle per inbound caregiver link (usually one). Sets `share_consent`. For a dependent (pet/child) the caregiver operates this portal, so this is also how the caregiver manages a dependent's permission. Endpoint `POST /api/portal/<token>/share-consent {caregiver_email, consent}`.
- **Caregiver's portal** (the primary's `client-portal.html`): a "Family notifications" list — each member with an "Email me about **[member]**" cc toggle. Sets `cc_enabled`. Endpoint `POST /api/portal/<token>/cc-pref {member_email, cc_enabled}`.
- **Owner console** (`/console/household`): show and toggle both flags per link (extends the existing CRUD page).

Both portal endpoints are token-scoped: the member endpoint only sets consent for links where the token's email is the **member**; the caregiver endpoint only sets cc for links where the token's email is the **primary** (no cross-account writes).

## Data flow

Member opens their portal → payload lists their inbound caregiver link(s) + current `share_consent` → toggle → `POST /share-consent` (token = member) → updates the row. Caregiver opens their portal → payload lists their members + `cc_enabled` → toggle → `POST /cc-pref` (token = primary). A member notification sends → `cc_recipients_for(member)` → cc added. Owner console shows/sets both.

## Components / files

- **`dashboard/household.py`** — the two columns + backfill in `init_household_tables`; `is_dependent`/`default_cc_for`; `set_share_consent(cx, primary, member, consent)`; `set_cc_enabled(cx, primary, member, enabled)`; `viewable_members_for(cx, primary)`; `cc_recipients_for(cx, member)`; `can_view` updated to require `share_consent`; `add_member` sets `cc_enabled` from relationship; `members_for` returns the two flags too.
- **`app.py`** — `POST /api/portal/<token>/share-consent`, `POST /api/portal/<token>/cc-pref`; the switcher payload uses `viewable_members_for` + carries the member's inbound caregiver + each member's cc state; cc hooks at the report + invoice sends; a `_household_sharing_enabled()` flag helper.
- **`static/client-portal.html`** — the member "Sharing" toggle + the caregiver "Family notifications" list.
- **`static/console-household.html`** — show/toggle both flags per link.
- **`dashboard/orders.py`** — cc hook in `_send_invoice_exec`.

No new tables; reuses `household_members`.

## Error handling

- `HOUSEHOLD_SHARING_ENABLED` off → toggles hidden, `/share-consent` + `/cc-pref` inert, cc routing off. `share_consent` stays 1 so views are unchanged.
- cc-resolution or cc-send failure → the primary notification still sends (best-effort try/except); never blocks the member's own email.
- Token scoping: member can only set their own consent; caregiver only their own cc; owner console gated by `_portal_console_ok()`.
- Migration/backfill idempotent; missing columns handled by the additive `ALTER TABLE ... pass` pattern.

## Testing

- **Model:** `default_cc_for`/`is_dependent` classification (pet/child→1, spouse/adult→0); `add_member` sets cc from relationship; migration backfills existing rows; `set_share_consent`/`set_cc_enabled` round-trip; `can_view` returns False when `share_consent=0`, True at 1; `viewable_members_for` excludes revoked; `same_household` unaffected by consent.
- **Routing:** `cc_recipients_for` returns a caregiver only when `share_consent=1 AND cc_enabled=1`; excluded when either is 0; multiple caregivers handled.
- **Endpoints:** member `/share-consent` sets only where token==member (rejects setting another member's); caregiver `/cc-pref` sets only where token==primary; flag-off inert.
- **Send integration:** a report/invoice send to a member with a consented+subscribed caregiver adds that caregiver to the recipient list + Cc header (mock SMTP); no cc when either switch off; a cc failure doesn't block the primary send.

## Out of scope / future

- cc for other notification types (magic link, practitioner, chat, SMS).
- Digest/frequency/quiet-hours controls on cc.
- A richer member↔multi-caregiver consent matrix UI.
- Member-facing "who can see me" audit / activity log.
