# Family / Household Portal-View Accounts (v1) — Design

**Date:** 2026-07-04
**Status:** Approved (brainstormed with Glen 2026-07-04)
**Repo:** deploy-chat

## Summary

A caregiver (parent, pet owner, or a professional caregiver with charges) needs to see the scans of everyone they care for in one place. Today each scan-subject is a **separate account** — E4L requires a distinct email/login per scan account, so every pet, child, dependent, or charge already has its own email, and its biofield reports are already keyed by that email in the portal. This feature lets one **primary/caregiver** account be linked to N **member** accounts and view any member's scans from the primary's own portal, plus lets the owner (console-side) **reassign a mis-attributed scan** to the correct member within a household.

**Portal-view first, not billing.** The link grants *viewing*; each member's own paywall is unchanged, so this adds zero new billing. It is built fresh on the current live `$1 = lifetime membership` paywall — it does NOT revive the retired per-scan-unlock model from the closed PR #522.

## Scope

**v1 = link + view + console-side reassignment.** Explicitly deferred: a combined side-by-side household dashboard (Approach B), caregiver-initiated (in-portal) reassignment, one-payer-covers-all-members billing/entitlement, self-serve family invites, and any change to the upstream E4L / reveal source records.

## The three capabilities

### 1. Linking (data model)

New module `dashboard/household.py` owning two tables in `LOG_DB`:

- **`household_members`** — the caregiver→member links.
  ```
  id            INTEGER PK
  primary_email TEXT   -- the caregiver / account holder
  member_email  TEXT   -- a linked scan account they care for
  label         TEXT   -- display name, e.g. "Mochi", "Kai"
  relationship  TEXT   -- optional: 'child' | 'pet' | 'dependent' | 'charge' | 'self' | '' (free-form allowed)
  created_at    TEXT
  UNIQUE(primary_email, member_email)
  ```
  All emails stored lowercased. The primary always views "self" implicitly (their own email) whether or not a self row exists.

- **`scan_reassignments`** — the reassignment audit log (reversible history).
  ```
  id         INTEGER PK
  scan_date  TEXT
  from_email TEXT
  to_email   TEXT
  by         TEXT   -- console actor (best-effort; e.g. 'console')
  at         TEXT
  ```

### 2. Viewing (member switcher on the existing portal)

- **Payload:** `api_client_portal(token)` (app.py ~13938) resolves the token to the primary's email, loads that email's household, and adds to the JSON payload:
  ```
  "household": [ {"email": <member_email>, "label": ..., "relationship": ...}, ... ]
  ```
  (empty list when the primary has no members). The list is for the switcher; it exposes only labels/relationships, never member scan data.
- **`?member=<email>`:** `api_client_portal` accepts an optional `member` query arg. It is honored **only** if `household.can_view(cx, viewer_email, target_email)` returns True — i.e. `target == viewer` OR a `household_members` row links `primary_email=viewer, member_email=target`. If the arg is absent or unauthorized, the portal serves the **primary's own** view (fail-closed — no IDOR, no error leak). When authorized, `email_for_reports` becomes the target's email and the rest of the portal renders unchanged.
- **Paywall unchanged:** the member view reuses `_portal_biofield_unlocked(target_email)` (app.py:10110), so a member whose own account isn't unlocked shows blurred remedies exactly as it would on that member's own portal. No new billing.
- **UI:** `static/client-portal.html` renders a small member selector at the top when `d.household` is non-empty (the primary/"self" plus each member). Selecting a member re-loads the portal with `?member=<email>`. Absent/empty → no selector, portal byte-identical to today.
- **Flag:** the whole feature ships behind `HOUSEHOLD_VIEW_ENABLED` (default OFF). Flag-off: no `household` key, `?member=` ignored → portal byte-identical.

### 3. Reassignment (console-side)

A mis-attributed scan (performed on the wrong account within a household) is corrected by the owner from the console. Reassignment is an **owner console tool, available independent of `HOUSEHOLD_VIEW_ENABLED`** — the correction is valuable on its own (it fixes the member's own portal too), so only the *viewing* switcher is behind the flag, not the console CRUD/reassign tools.

- **Move logic:** `household.reassign_report(cx, scan_date, from_email, to_email)`:
  1. Guard both emails are in the **same household** (share a `primary_email`, or one is the other's primary). Reject cross-household moves.
  2. Guard the target does **not already** have a `portal_biofield_reports` row for that `scan_date` (the table has `UNIQUE(email, scan_date)`) — refuse with a clear message rather than clobber.
  3. Re-key the `portal_biofield_reports` row: `UPDATE ... SET email=to_email WHERE email=from_email AND scan_date=scan_date`.
  4. Append a `scan_reassignments` audit row.
- **Endpoints (X-Console-Key):** `POST /api/console/household/reassign` `{scan_date, from_email, to_email}`; link CRUD `POST /api/console/household` `{primary_email, member_email, label, relationship}` and `DELETE /api/console/household` `{primary_email, member_email}`; `GET /api/console/household?primary_email=` returns a household's members and each member's scan dates.
- **Console page:** `/console/household` — search/select a household, see members + their scan dates, add/remove links, and reassign a scan to another member. Mirrors the existing `/console/portal-links` page pattern.
- **Re-publish caveat (documented, not fixed in v1):** the upstream reveal still carries the original email, so re-publishing that same reveal could re-create the mis-attribution. v1 documents this; the owner controls re-publishing. (A future "mark reassigned → warn on re-publish" guard or a source-level fix is out of scope.)

## Data flow

`/portal/<token>` GET `api_client_portal` → primary email → `household.members_for(cx, primary)` → `payload["household"]`. If `?member=M` and `household.can_view(cx, primary, M)` → `email_for_reports = M` → existing report/blur/render path runs on M. Console reassignment: `/console/household` → `POST /api/console/household/reassign` → `household.reassign_report` re-keys `portal_biofield_reports` + logs.

## Components / files

- **New `dashboard/household.py`** — tables + pure/tested functions: `init_household_tables(cx)`, `add_member`, `remove_member`, `members_for(cx, primary) -> [ {email,label,relationship} ]`, `can_view(cx, viewer, target) -> bool`, `same_household(cx, a, b) -> bool`, `reassign_report(cx, scan_date, from_email, to_email) -> {"ok":bool,"error":?}`, `list_reassignments(cx, ...)`.
- **`app.py`** — `api_client_portal`: household payload + guarded `?member=` (behind `HOUSEHOLD_VIEW_ENABLED`); new console endpoints + `/console/household` page.
- **`static/client-portal.html`** — the member selector (renders from `d.household`).
- **`static/console-household.html`** (or reuse an existing console shell) — the owner CRUD + reassignment UI.

No change to `portal_biofield_reports` schema, the reveal/publish pipeline, or E4L.

## Error handling

- Unknown or unauthorized `?member=` → serve the primary's own view (fail-closed); never error, never leak that an email exists.
- Missing household tables / any lookup failure → best-effort: omit `household`, portal renders normally.
- Reassignment: cross-household → refused; same-date collision on target → refused (no clobber); both surfaced as messages in the console, not silent.
- `HOUSEHOLD_VIEW_ENABLED` off → feature fully inert.

## Testing

- **`household.py` units:** `can_view` (self→True, linked→True, unlinked→False); `add`/`remove`/`members_for` round-trip; `same_household`; `reassign_report` (moves the row + writes an audit row; refuses same-date collision; refuses cross-household).
- **Route tests:** `?member=<linked>` serves the member's reports; `?member=<unlinked>` falls back to self (no leak); flag-off → no `household` key and `?member=` ignored.
- **Console:** `POST /api/console/household/reassign` moves a report and rejects a collision / cross-household move; link CRUD round-trips; all guarded by X-Console-Key.

## Out of scope / future

- Combined side-by-side household dashboard (Approach B) and a household roster header (Approach C polish).
- Caregiver-initiated (in-portal) reassignment.
- One-payer-covers-all-members billing/entitlement (the deferred "family membership").
- Self-serve family invites / member management by the caregiver.
- Source-level (reveal/E4L) correction so a reassignment survives a re-publish.
