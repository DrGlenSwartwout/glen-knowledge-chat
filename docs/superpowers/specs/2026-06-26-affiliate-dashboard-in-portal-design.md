# Affiliate Dashboard in the Personal Portal (Step 2b-1)

**Date:** 2026-06-26
**Status:** Approved (design)
**Author:** Glen + Claude
**Parent:** affiliate→personal-portal unification. 2b-1 = bring the affiliate dashboard *view* into the personal portal (read-only), reusing the affiliate data. 2b-2 = social-links gamification. 2b-3 = retire the standalone `/affiliate/portal`.

## Problem

An enrolled ambassador's full dashboard (stats, links/offers, recent referrals, recruit) lives only in the standalone `/affiliate/portal` (its own login). The personal portal's Ambassador section (Step 1) shows only two links.

## Goal

Show the full affiliate dashboard (minus social-gamification) inside the personal portal's Ambassador section, **authed by the portal identity** (email→slug), reusing the same data the standalone page uses. No second login.

## Design

**Refactor-and-reuse:** extract the data builder so both the standalone route and the portal share it (also prepares 2b-3).

### Component 1 — `dashboard/affiliate_dashboard.py` (new, pure, offline-tested)

`build_dashboard(cx, slug, *, quiz_url, public_base_url) -> dict` — replicates exactly what `/affiliate/portal-data` returns today (so the standalone page is byte-for-byte unchanged):
- Fetch the `affiliate_signups` row by `slug` → `name, organization, short_url, created_at` (status not needed here — callers gate on approved before calling). If no row → return `{}`.
- `tracking_url = short_url or f"{quiz_url}?utm_source={slug}&utm_medium=affiliate&utm_campaign=scoreapp-quiz"`; `recruit_url = f"{public_base_url}/affiliate?ref={slug}"`.
- Queries (all `LOG_DB`, keyed by slug): `stats` (COUNT + MAX(received_at) from `referral_events WHERE utm_source=slug`), `recent` (top-10 `referral_events`: received_at, first_name, last_name, quiz_score), `recruited_count` (`affiliate_signups WHERE referred_by=slug AND status='approved'`), `conversions_count` (`affiliate_conversions WHERE affiliate_slug=slug`), `offers` (`affiliate_offers WHERE active=1 ORDER BY sort_order`), `social` (`affiliate_social_links WHERE slug=slug`).
- Returns the exact dict: `name, organization, slug, tracking_url, recruit_url, total_leads, last_lead, recruited_count, conversions_count, recent[{received_at,name,score}], offers[{name,description,url,instructions}] (url = url_template.replace("{slug}",slug)), social_links[{url,points,views,likes,shares,ts}], member_since`.
- Includes a module-local `_mask_lead_name(first,last)` identical to app.py's (`"Mary","Johnson" → "Mary J."`), used for `recent[].name`.

### Component 2 — refactor `/affiliate/portal-data` (`app.py`)

Keep the token→row lookup + approved gate; replace the inline queries + return with:
```python
return jsonify(affiliate_dashboard.build_dashboard(cx, slug, quiz_url=QUIZ_URL, public_base_url=PUBLIC_BASE_URL))
```
Behavior-preserving (same payload). The old inline `stats/recent/.../social` queries + the big return dict are removed (now in the module).

### Component 3 — extend `_ambassador_block` (`dashboard/portal_view.py`)

In the **enrolled** branch only, add the dashboard:
```python
return {"status": "enrolled", "slug": slug,
        "referral_url": ..., "recruit_url": ...,
        "dashboard": affiliate_dashboard.build_dashboard(cx, slug, quiz_url=quiz_url, public_base_url=public_base_url)}
```
(Existing `status`/`referral_url`/`recruit_url` kept for back-compat; `dashboard` is additive. None/pending branches unchanged. `/api/portal/<token>/view` already passes `quiz_url`/`public_base_url` — no route change.)

### Component 4 — render in `static/client-portal.html`

In the Ambassador card's **enrolled** branch, render the dashboard from `amb.dashboard`:
- **Stats row:** total_leads, last_lead, conversions_count, member_since.
- **Your links / offers:** the existing referral + recruit links, plus the `offers[]` (name, description, the `url`, instructions).
- **Recent referrals:** `recent[]` (masked name + score + date); empty-state if none.
- **Recruit:** recruit_url + `recruited_count`.
- **Do NOT render `social_links`** (that's 2b-2). Keep escaping via the page's `esc()`.

## Non-goals

- Social-links gamification (2b-2). `build_dashboard` returns the data, but the portal doesn't render it yet.
- Retiring/redirecting the standalone `/affiliate/portal` (2b-3).
- The practitioner portal dashboard render — a fast follow-up (data path is reused; just the render).
- Any change to the affiliate engine, enrollment, or the affiliate token.

## Error handling

- `build_dashboard` none-raising: no `affiliate_signups` row for the slug → `{}`; query errors degrade gracefully (wrap the query block; on error return the dict with zero/empty stats). The standalone route still validates token/approved before calling.
- `_ambassador_block` stays none-raising (it already wraps its query); a `build_dashboard` failure inside the enrolled branch must not break the block — wrap it so `dashboard` is omitted on error.

## Testing

**Offline (tmp sqlite) — `build_dashboard`:**
1. Seed `affiliate_signups` (slug, name, short_url, created_at) + `referral_events` (2 rows) + `affiliate_offers` (1 active) + `affiliate_conversions` (1) + an `affiliate_signups` recruit row (`referred_by=slug, status=approved`). Assert `total_leads==2`, `last_lead` = latest, `recruited_count==1`, `conversions_count==1`, `tracking_url` (long form when no short_url), `recruit_url`, `recent` masked names (`"Mary J."`), `offers[0].url` has slug substituted, `member_since`.
2. `short_url` present → `tracking_url == short_url`.
3. Unknown slug → `{}`.

**Offline — `_ambassador_block`:**
4. Enrolled → result includes a `dashboard` dict (with the keys above); none/pending → no `dashboard`.

**Live post-deploy (`app.py`/HTML can't import offline):**
5. `/affiliate/portal-data?token=<approved>` still returns the same shape (standalone unchanged).
6. `/api/portal/<token>/view` for an enrolled portal user → `ambassador.dashboard` present with stats/offers/recent; render-verify the portal shows the dashboard, zero console errors. (Seed an approved `affiliate_signups` row for a test portal email to exercise the enrolled state.)

## Rollout

Ships on merge → Render deploy. Verify the standalone page is unchanged + an enrolled portal user sees the dashboard. 2b-2 (social) and 2b-3 (retire standalone) follow.
