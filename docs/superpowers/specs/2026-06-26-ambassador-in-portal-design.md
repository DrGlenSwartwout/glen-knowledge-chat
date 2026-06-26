# Ambassador, Embedded in the Personal Portal (Step 1)

**Date:** 2026-06-26
**Status:** Approved (design)
**Author:** Glen + Claude
**Parent program:** unify the affiliate system into the personal portal (Glen 2026-06-26). Step 1 = embed the ambassador section in the portal (this spec). Step 2 = redirect the standalone `/affiliate/portal` into the personal portal + enable client self-login. Step 3 = auto-provision the portal on membership join. This spec is Step 1 only.

## Problem

The affiliate/ambassador system lives in a **separate** portal (`/affiliate/portal`) with its **own** magic-link login. A member already authenticated to their personal portal can't see or use their ambassador links without a second login — so the ambassador system is out of sight during normal interactions.

## Goal

Add an **Ambassador section to the personal portal**, authed by the portal's existing identity (no second login): if the person isn't enrolled it shows a signup call-to-action; if enrolled it shows their referral links inline. Reuses the affiliate engine (`affiliate_signups` + slug); does not rebuild it. Deploys to prod.

## Non-goals (later steps / out of scope)

- Full affiliate dashboard inline (stats, payouts, social tools) — Step 2.
- Retiring/redirecting the standalone `/affiliate/portal` — Step 2.
- Client self-login (`CLIENT_LOGIN_ENABLED`) — Step 2/3.
- Auto-provisioning the portal on membership — Step 3.
- The public recruit hub (`/affiliate/hub/<slug>`) and public apply page — unchanged (outward-facing by design).
- The practitioner portal — its own surface; a fast follow-up after the client portal.

## Design

The role-aware portal payload `get_portal_view` (`dashboard/portal_view.py`) already returns `account` / `orders` / `biofield` / `upgrade`. Add an `ambassador` block, computed from the person's email against the existing `affiliate_signups` table.

### Component 1 — `dashboard/portal_view.py` (pure, offline-tested)

`_ambassador_block(cx, email, quiz_url, public_base_url) -> dict`:
- Look up the person in `affiliate_signups` by `lower(email)` → `(slug, status)`. None-raising: missing table or no row → treated as not-enrolled.
- **status == "approved"** → enrolled:
  ```json
  {"status": "enrolled", "slug": "<slug>",
   "referral_url": "<quiz_url>?utm_source=<slug>&utm_medium=affiliate&utm_campaign=scoreapp-quiz",
   "recruit_url": "<public_base_url>/affiliate?ref=<slug>"}
  ```
  (Same URL shapes as `/affiliate/portal-data`; derived from the slug, so no second login is needed.)
- **row exists, status != "approved"** → `{"status": "pending"}`.
- **no row** → `{"status": "none", "signup_url": "<public_base_url>/affiliate/apply-form"}`.

`get_portal_view(cx, person_id, *, offers_enabled_keys=None, scan_date=None, quiz_url="", public_base_url="")` — add the two keyword params and include `"ambassador": _ambassador_block(cx, email, quiz_url, public_base_url)` in the returned dict (email is already derived in the function).

### Component 2 — `app.py` (prod, live-verified)

In `/api/portal/<token>/view` (`api_client_portal_view`), pass the constants when calling `get_portal_view`: `quiz_url=QUIZ_URL, public_base_url=PUBLIC_BASE_URL` (both already defined in app.py: `QUIZ_URL="https://healing.scoreapp.com"`, `PUBLIC_BASE_URL` env).

### Component 3 — `static/client-portal.html` (prod, live-verified)

Render an Ambassador card from the `/view` payload's `ambassador`:
- **enrolled** → "Your Ambassador links" with `referral_url` (share to earn) and `recruit_url` (invite other ambassadors), each shown copyable (match the page's existing link/button styling).
- **pending** → "Your ambassador application is under review."
- **none** → "Become an Ambassador — earn rewards by sharing" + a button linking to `signup_url` (the public apply form).

## Error handling

- `_ambassador_block` is none-raising (missing `affiliate_signups` table or query error → `{"status":"none", signup_url}`); never breaks the portal view.
- Existing portal sections are untouched.

## Testing

**Offline (tmp sqlite) — `_ambassador_block` + `get_portal_view`:**
1. Approved signup → `status=="enrolled"`, `referral_url`/`recruit_url` built from the slug + the passed bases.
2. Row with `status="pending"` → `{"status":"pending"}`.
3. No row → `{"status":"none","signup_url": "<base>/affiliate/apply-form"}`.
4. Email lowercased on lookup (mixed-case input matches a lowercase-stored signup).
5. No `affiliate_signups` table → `status=="none"` (none-raising).
6. `get_portal_view` includes an `"ambassador"` key (seed a people row + an approved signup → enrolled block present).

**Live post-deploy (`app.py`/HTML can't import offline):**
7. `GET /api/portal/<token>/view` for Karin (not an affiliate) → `ambassador.status == "none"` with a `signup_url`.
8. Render-verify Karin's portal page: the "Become an Ambassador" card shows with a working signup link; zero console errors. (For an enrolled test signup, the card shows the referral + recruit links.)

## Rollout

Ships on merge → Render deploy. Verify on Karin's portal (shows the signup CTA) and, with a seeded approved signup, the links view. Steps 2 (one login / retire standalone portal) and 3 (provision on membership) follow as their own specs.
