# Provision the Personal Portal on Membership Join (Step 3, core)

**Date:** 2026-06-26
**Status:** Approved (design — Glen chose "core only", no welcome email)
**Author:** Glen + Claude
**Parent:** the personal-portal program. Step 3 makes the portal "always there" for every member by guaranteeing each has a `people` row (so self-login works). Mirrors the 2b-3 affiliate coverage, applied to members.

## Problem

The personal portal is keyed on the `people` table (self-login at `/portal/login` matches `people` by email). A member must therefore have a `people` row to reach their always-there portal. Most members are in `people` (GHL/contacts), but joins don't guarantee it — and we want every member, present and future, to have a reachable portal.

## Goal

On every membership join, ensure the member has a `people` row; backfill existing members. No welcome email (core only). Reuses `customers.find_or_create_by_email`.

## Design

Membership creation funnels through two chokepoints — both get a best-effort `people`-row ensure (mirrors 2b-3):
- `_grant_membership(cx, email, days, source)` (`app.py`) — the access-grant row.
- `subscriptions.create_membership(cx, *, email, …)` (`dashboard/subscriptions.py`) — the billing subscription.

### Component 1 — `subscriptions.backfill_member_people(cx) -> int` (offline-tested)

Ensure a `people` row for every current member missing one. "Member" = an active membership subscription OR an unexpired access grant:
```sql
SELECT DISTINCT email FROM subscriptions WHERE kind='membership' AND status='active'
UNION
SELECT DISTINCT email FROM memberships WHERE expires_at > :now_iso
```
(expires_at is ISO-8601 text with trailing `Z`, so lexical `>` against a current ISO-Z string is correct.) For each email with no `people` row → `customers.find_or_create_by_email(cx, email=email, name="")` (name unknown here; the greeting falls back gracefully and GHL sync fills it later). Idempotent; returns count created; none-raising per email. `now_iso` computed via the module's `_now_iso()` (no override needed).

### Component 2 — console endpoint (`app.py`)

`POST /api/console/backfill-member-people` — `_bos_actor()`-gated (mirror `/api/console/backfill-affiliate-people`), `?dry_run=1` aware. Computes the missing-member emails; dry → reports; real → runs `backfill_member_people`, returns `{ok, created, emails}`. Triggered once at go-live.

### Component 3 — on-join hooks (best-effort)

- In `subscriptions.create_membership(...)`, after the `INSERT INTO subscriptions`, add a best-effort `customers.find_or_create_by_email(cx, email=email)` (local import; try/except → never break the join).
- In `_grant_membership(cx, email, …)` (`app.py`), after the `INSERT INTO memberships`, add the same best-effort call.

So every future member — whether they arrive via a paid subscription or an access grant (incl. the biofield trial) — gets a `people` row at join.

## Non-goals

- Welcome / "your portal is ready" email (deferred; can be a clean follow-up).
- Pre-creating a `client_portal` token row (not needed — self-login works off the `people` row + session).
- Any change to billing, the membership lifecycle, or the portal itself.
- Practitioner-portal dashboard render (separate follow-up).

## Error handling

- `backfill_member_people` none-raising per email (one bad row doesn't abort).
- Both on-join hooks are best-effort (try/except + log → never break create_membership / _grant_membership).
- Console endpoint: 401 unless `_bos_actor()`.

## Testing

**Offline (tmp sqlite) — `backfill_member_people`:**
1. Seed `subscriptions` (1 active membership, 1 cancelled membership, 1 active non-membership) + `memberships` (1 unexpired grant, 1 expired) + a `people` row for one of the member emails. Run → creates `people` rows only for member emails (active membership OR unexpired grant) that are missing one; cancelled/expired/non-membership skipped; already-present skipped. Re-run → 0 (idempotent).
2. The created `people` row carries the email (self-login works).

**Live post-deploy (`app.py`/HTML can't import offline):**
3. `POST /api/console/backfill-member-people?dry_run=1` → reports the member emails missing a `people` row; real run → `created: N`; re-dry → 0.
4. (Hooks verified transitively: a new membership grant / subscription creates a `people` row — confirmed by the dry-run staying at 0 after, or by code review since the hooks are tiny best-effort calls.)

## Rollout

Ships on merge → Render deploy. Go-live: run the backfill (dry then real). Every member then has a reachable personal portal. Welcome email + practitioner dashboard render remain as optional follow-ups.
