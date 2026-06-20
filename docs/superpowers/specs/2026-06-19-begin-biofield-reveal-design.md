# Begin Page #4a — Biofield Analysis Reveal (Find step 2)

**Date:** 2026-06-19
**Status:** Approved (design); ready for implementation plan
**Repo:** deploy-chat (Flask, Render `glen-knowledge-chat`, illtowell.com)
**Parent:** Begin-page redesign. #4 (Match/ordering + Biofield interpretation) decomposed into **4a (this: the reveal), 4b ($1 tripwire unblur -> $99/mo), 4c (match-order polish)**. Builds on #1 (identity), #2 (journey map), #3 (the reserved `biofield` gate = Find card step 2).

---

## Problem

The Find card's step 2 (`biofield` gate) is reserved but nothing sets it, and there is no funnel surface that turns a visitor's E4L voice scan into a remedy reveal. We want: a scan becomes a **ranked remedy interpretation**; the visitor sees their **top match free**; the **deeper matches stay blurred** (the $1-trial unlock is 4b); and doing so **fills Find step 2**. Two hard requirements from Glen: (1) the free top reveal is **manually reviewed/approved/edited by Glen** before a visitor sees it; (2) showing a person's analysis must not leak one person's health data to another who merely typed their email.

## Goal

Scan -> a locally-produced ranked-remedy **draft** pushed to the server -> Glen **reviews/approves/edits** in a console -> the visitor is emailed a **magic link** that both notifies and **verifies ownership** -> a token-gated reveal page shows the **approved top remedy free + the deeper remedies blurred** -> the `biofield` gate is set. No Stripe, no FMP, no server-side scan interpretation in 4a.

## Scope (#4a)

The **server side** of the reveal: a draft-ingest endpoint, a `biofield_reveals` store, a console review surface (approve/edit), an approve-fires "ready" email with a magic link, a token-verified reveal page (top free + blurred depth + a stub unlock CTA), and setting the `biofield` gate on view. Plus the **JSON contract** for the local matcher push (the matcher itself is Glen's existing local agent, out of scope).

**Out of scope:**
- The **$1 charge / card-vault / unblur of the deeper remedies** -> **4b**. 4a renders the deeper remedies blurred with a stub CTA ("unlocking soon").
- **FMP** (`fmp_biofield`, Supabase causal-chain) -> dropped; FMP is being retired.
- **Server-side scan interpretation / Pinecone ranking** -> the local matcher produces the draft; the server stores/reviews/reveals it.
- The local e4l-scan-remedy-matcher agent + its push step (Glen runs it, like the scan-freshness push); 4a only defines the contract.

---

## Confirmed decisions (Glen, 2026-06-19)

- **Engine = E4L, not FMP.** Ranked remedies come from the E4L scan via Glen's local matcher; the server does not interpret.
- **Local matcher pushes the finished draft** to a new server endpoint (like scan-freshness). Least PHI on the server.
- **Free reveal = the surface / #1-priority remedy**; the deeper remedies are blurred (the blur is the unlock incentive).
- **Manual review:** every top reveal is Glen-approved/edited in a console before a visitor sees it (`ai_draft` -> `confirmed`).
- **ToS required** + **verified ownership via magic link** before the reveal renders (the "it's ready" email's link is the verification).
- **Live page, no feature flag** for the reveal route; the console is staff-gated. No emoji, no em dashes.

---

## Architecture

### 1. Draft ingest — `POST /api/e4l/reveal-draft`
Auth: `X-Cron-Secret` (== `CRON_SECRET`, falls back to `CONSOLE_SECRET`) — same pattern as `/api/e4l/scan-freshness`. Body:
```
{ "email": "...", "scan_date": "YYYY-MM-DD",
  "top_match": { "name": "...", "slug": "...", "meaning": "one warm sentence" },
  "blurred":   [ { "kind": "..." }, ... ],   // the deeper remedies; names optional, count is what 4a shows
  "source": "e4l-matcher" }
```
Validates email + top_match.name; upserts a `biofield_reveals` row (status `ai_draft`) keyed by `(email, scan_date)`. Idempotent: re-pushing the same `(email, scan_date)` updates the draft only while status is `ai_draft` (never overwrites a `confirmed` reveal). Always returns 200 with `{"ok": true, "id": ...}` (or a 400 on a missing email/top_match). Never raises.

### 2. Store — `dashboard/biofield_reveals.py`
New module + table:
```
biofield_reveals (
  id INTEGER PK, email TEXT, scan_date TEXT,
  top_json TEXT,        -- {name, slug, meaning}
  blurred_json TEXT,    -- [{kind}, ...]
  status TEXT,          -- 'ai_draft' | 'confirmed'
  token_hash TEXT,      -- set at approve (the magic link)
  approved_at TEXT, approved_by TEXT,
  created_at TEXT, updated_at TEXT,
  UNIQUE(email, scan_date) )
```
Functions: `init_table(cx)`; `upsert_draft(cx, email, scan_date, top, blurred, source)` (no-op overwrite if already `confirmed`); `list_drafts(cx)` (status `ai_draft`, newest first); `get(cx, id)` / `get_by_token_hash(cx, th)`; `set_top(cx, id, top)` (edit, stays draft); `approve(cx, id, by, token_hash)` (-> `confirmed`, stamps approver + token). Kept separate from `portal_biofield_reports` (the $300-service portal report).

### 3. Console review — `/console/biofield-reveals`
A staff page (`_check_console_auth`, like `/console/sales-pages`) listing `ai_draft` reveals; per row: the top-match name + meaning (editable), the blurred count, the email/scan_date. Actions go through the existing dispatch spine (`dashboard/actions.py` `register_action`/`Action`/`LOW_WRITE`, `rbac` OWNER+OPS, the `/api/action/<key>` route):
- `biofield_reveal.edit` — update `top_json` (name/meaning); stays `ai_draft`.
- `biofield_reveal.approve` — `approve()` -> `confirmed`, mint the magic-link token, then fire the "ready" email (best-effort; approve never fails if the email fails). Modeled on `dashboard/sales_pages_actions.py` (a new `dashboard/biofield_reveal_actions.py` with `configure(**kw)` for injected deps: db path, base_url, send fn).
New `static/console-biofield-reveals.html` modeled on `console-sales-pages.html`; a "Biofield Reveals" sub-tab in the console nav.

### 4. Approve -> "ready" email + magic link
On approve: `token = secrets.token_urlsafe(32)`; store `_hash_token(token)` on the row; INSERT an `auth_tokens` row `(token_hash, email, purpose="biofield_reveal", created_at, expires_at=+30d)` — the same table/pattern as `/reorder/request`. Email the visitor (via `_send_inquiry_email`) a short "Your Biofield Analysis is ready" note with the link `{PUBLIC_BASE_URL}/begin/biofield/<token>`. (Copy provisional; Glen voice; no emoji/em dash.)

### 5. Reveal page — `GET /begin/biofield/<token>`
Verifies the token against `auth_tokens` (purpose `biofield_reveal`, not expired) AND the matching `confirmed` `biofield_reveals` row (`get_by_token_hash`). On success, serves `static/begin-biofield.html` (no-store; PHI-adjacent) rendering:
- the **approved top remedy free**: name + the one-line meaning + a buy link (`/begin/buy/<slug>` if the slug is a catalog product, else the match/explore link);
- the **deeper remedies blurred**: a CSS-blurred stack sized to the blurred count, with the teaser "+N more in your full analysis" and an **"Unlock your full Biofield Analysis"** CTA that in 4a is a stub (a disabled/"unlocking soon" button; 4b wires the $1 flow).
On view, fire `_record_entry_unlock("biofield", email)` (the row's email) -> **Find card step 2 fills**. An invalid/expired/missing token -> a friendly "this link is no longer valid, request a fresh one" page (no PHI, no enumeration leak). The token is NOT consumed on view (the visitor can reopen the link; 30-day TTL).

### 6. Find-step-2 wiring
The reveal sets the `biofield` gate by the row's email; `get_state` unions it onto the visitor's journey by email (as in #3), so the Find card shows step 2 done. The Find-step-2 `href` in `JOURNEY_STEPS` stays `/begin/match` for now (the tokened reveal link is delivered by email, not from the card; a logged-in deep link can come in 4b).

### Reuse / untouched
- `auth_tokens` + `_hash_token` + the magic-link verify pattern (`/reorder`); `_send_inquiry_email`.
- The console dispatch spine + `_check_console_auth` + `/api/action/<key>` (Phase 5).
- `_record_entry_unlock` (#3), `is_member`/ToS, `journey_state`.
- Untouched: `fmp_biofield`, `portal_biofield_reports`, `surface()`/`/begin/explore`, `/begin/match*`, pricing/Stripe.

---

## Data flow
1. Scan completes -> Glen's local matcher builds the ranked draft -> `POST /api/e4l/reveal-draft` -> `biofield_reveals` row `ai_draft`.
2. Glen opens `/console/biofield-reveals`, edits the top match if needed, **approves** -> `confirmed` + token minted + "ready" email sent.
3. Visitor clicks the magic link -> `/begin/biofield/<token>` verifies ownership -> renders the **approved top remedy free + blurred depth + stub CTA** -> sets the `biofield` gate.
4. The journey map's Find card step 2 fills (by email union).
5. The $1 unlock of the blurred depth = **4b**.

## Error handling
- Ingest: missing email/top_match -> 400; never overwrites a `confirmed` reveal; always 200 on a valid push; wrapped so a bad row never 500s the batch.
- Approve: the "ready" email is best-effort; a send failure does not roll back the approval (the visitor can be re-notified). Token mint + status flip happen in one committed step before the email.
- Reveal: invalid/expired/missing token -> friendly non-PHI page; no email enumeration; `no-store` headers.
- `_record_entry_unlock("biofield", ...)` is wrapped/idempotent (#3) -> a gate-set failure never blocks the page.
- If a visitor has multiple confirmed reveals (re-scans), the token resolves to a specific scan's reveal; the newest can be re-sent from the console.

## Testing
- **Ingest:** valid push -> `ai_draft` row by `(email, scan_date)`; re-push updates while draft; a `confirmed` row is not overwritten; missing email/top_match -> 400; auth required (real `CRON_SECRET` path, per the #3 lesson — do not widen auth).
- **Store:** `upsert_draft` / `list_drafts` / `set_top` / `approve` transitions; `get_by_token_hash`.
- **Console action:** `biofield_reveal.approve` flips status to `confirmed`, mints an `auth_tokens` row (purpose `biofield_reveal`), and calls the injected send fn once; `edit` updates `top_json` and stays `ai_draft`; RBAC OWNER/OPS.
- **Reveal route:** a valid token -> 200 serving the page with the top name/meaning present and the blurred count; sets the `biofield` gate (assert `get_state(email=).unlocked_gates` contains `biofield`); an invalid/expired token -> the friendly page, gate NOT set; `no-store` header present.
- Front-end (blur visuals, the stub CTA, responsive) = manual visual pass (state it). Server has unit + Flask-test-client coverage.
- deploy-chat test isolation (tmp `LOG_DB`; `init` the new table + `auth_tokens`; mock the send fn; mock GHL on any free-tier transition). No emoji; no em dashes.

## Notes
- **Live page, no flag** for `/begin/biofield/<token>` (reveal). The console is staff-gated. Manual visual pass before relying on it.
- The **$1 unlock CTA is a stub** in 4a (renders blurred + "unlocking soon"); 4b makes it the tripwire that unblurs + converts to $99/mo.
- All copy provisional (reveal page, email, CTA) -> BNSN site pass later.
- The local matcher -> reveal-draft push is a companion Glen runs; 4a pins only the JSON contract above so the two sides agree.
- Privacy: the magic link is the ownership proof; no stored reveal is ever shown from a merely-typed email. ToS-membership still required.
