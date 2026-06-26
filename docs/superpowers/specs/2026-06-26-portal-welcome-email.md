# Portal Welcome Email on Membership Join (Step 3 follow-up)

**Date:** 2026-06-26
**Status:** Approved (Glen chose: both paths once-per-email excluding $1 biofield trial; self-serve `/portal/login` link; future joins only — no backfill blast)
**Parent:** the personal-portal unification program. Step 3 (#342) guaranteed every member a `people` row so self-login works. This tells a new member, once, that their portal is ready.

## Problem

A new member now has a reachable portal (Step 3) but is never told. We want exactly one "your portal is ready" email per member on join, pointing them to self-login — without spamming existing members and without blocking the join flow.

## Decisions

- **Trigger:** every membership join — both the paid-subscription chokepoint (`create_membership`) and the access-grant chokepoint (`_grant_membership`) — but **once per email** (a send-guard dedups overlap), and **excluding the $1 biofield trial** (`source='biofield_trial'`).
- **Link:** the email button points to `/portal/login` (self-serve magic-link; never expires, always works). No embedded token.
- **Existing members:** future joins only. No mass backfill.
- **Suppression:** this is a proactive send → respect `email_suppression.is_suppressed` (unlike the auth magic-link, which is exempt).
- **Non-blocking:** the join flow must not wait on SMTP. DB work (suppression check, send-guard, name lookup) runs synchronously on the request connection; the network send runs in a daemon thread (existing app pattern).

## Design

### Component 1 — `dashboard/portal_welcome.py` (offline-tested, stdlib only)

- `mark_welcome_sent(cx, email) -> bool` — ensure table `portal_welcome_sent(email TEXT PRIMARY KEY, sent_at TEXT)` exists; `INSERT OR IGNORE` the lowercased email; return `True` iff a row was newly inserted (first time → caller should send), `False` if already present (already sent → skip). This is the once-per-email guard; idempotent.
- `welcome_email_content(name, login_url) -> (subject, text_body, html_body)` — pure builder. Glen's voice/brand; greets by first name when known (falls back to "there"); body explains the portal (biofield reports, reorder, ambassador) and CTA to `login_url`. No secrets, no token.

### Component 2 — `app.py` send function

- `send_portal_welcome_email(to_email, name, login_url) -> (sent_via, err)` — mirrors `send_magic_link_email`'s cascade (GHL workflow → SMTP → console-log fallback), using `welcome_email_content`. Pure network/IO; no DB. (Suppression is checked by the orchestrator before this is called.)

### Component 3 — `app.py` orchestrator + hooks

- `_member_join_welcome(cx, email, source=None)` — best-effort (whole body in `try/except` + log; never raises):
  1. normalize email; if blank → return.
  2. if `source == 'biofield_trial'` → return (excluded).
  3. if `email_suppression.is_suppressed(cx, email)` → return.
  4. if `portal_welcome.mark_welcome_sent(cx, email)` is `False` (already sent) → return.
  5. resolve first name from the `people` row (Step 3 guarantees it; fall back to "").
  6. `login_url = PUBLIC_BASE_URL + "/portal/login"`.
  7. spawn a daemon thread calling `send_portal_welcome_email(email, name, login_url)`.
- **Hook sites** (one uniform line each — exclusion centralized in the orchestrator):
  - inside `_grant_membership(cx, email, days, source)` → `_member_join_welcome(cx, email, source)` (after the people-row hook).
  - after each `create_membership(...)` call site in app.py (4 sites incl. the biofield-trial site, which passes `source='biofield_trial'` so the orchestrator skips it).

### Component 4 — console test endpoint (live verification)

- `POST /api/console/test-portal-welcome?email=<addr>&dry_run=1` — `_bos_actor()`-gated. `dry_run=1` (default) reports what would happen for that email (resolved name, login_url, subject, and any skip reason: suppressed / already-sent / would-send) without sending or marking. Without `dry_run`, sends one welcome to that email (marks it). Lets Glen send himself a live test without a real join, and respects "no mass backfill."

## Non-goals

- No mass backfill to existing members.
- No embedded one-click token (self-serve `/portal/login` only).
- No change to billing, membership lifecycle, or the portal itself.
- No welcome for the $1 biofield trial.

## Error handling

- `_member_join_welcome` best-effort: any failure logs and returns — never breaks `create_membership`/`_grant_membership`.
- Mark-before-send: the guard is set before the thread sends, so a transient send failure does not cause a duplicate later (best-effort: a failed send means no welcome rather than a repeat).
- Suppressed / blank / already-sent → silent skip.

## Testing

**Offline (`dashboard/portal_welcome.py`):**
1. `mark_welcome_sent` → `True` first call, `False` second (idempotent, case-insensitive).
2. `welcome_email_content` → subject non-empty; first name appears when provided and falls back gracefully; `login_url` present in both text and html; no token/secret.

**Live (post-deploy; `app.py` can't import offline):**
3. `POST /api/console/test-portal-welcome?email=<glen>&dry_run=1` → reports would-send + resolved name + login_url. Then real send to Glen's own address → he receives it; re-dry → "already-sent".
4. Hooks verified transitively + by code review (tiny best-effort calls); a real future join sends exactly one.

## Rollout

Ships on merge → Render deploy. No migration (table self-creates on first guard call). No flag. Live-verify via the test endpoint to Glen's own email.
