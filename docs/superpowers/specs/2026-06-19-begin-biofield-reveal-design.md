# Begin Page #4a — Biofield Analysis Reveal (Find step 2)

**Date:** 2026-06-19 (rev 2026-06-20: interpretation auto-shows; only remedies are gated)
**Status:** Approved (design); reworked from rev 1 before merge.
**Repo:** deploy-chat (Flask, Render `glen-knowledge-chat`, illtowell.com)
**Parent:** Begin-page redesign. #4 decomposed into **4a (this: the reveal), 4b ($1 tripwire unblur -> $99/mo), 4c (match-order polish)**. Builds on #1 (identity), #2 (journey map), #3 (the reserved `biofield` gate = Find card step 2).

---

## Problem

The Find card's step 2 (`biofield` gate) is reserved but nothing sets it, and there is no funnel surface that turns a visitor's E4L voice scan into a Biofield reveal. We want: a scan becomes an **interpretation** (a reading) the visitor sees **automatically**, plus a **ranked list of matched remedies** that start **blurred**; Glen **manually approves/edits** the remedies, which **un-blurs the first (top) remedy free**; the rest stay blurred until the **$1 trial** (4b); and viewing fills **Find step 2**. Two hard requirements: (1) only the **remedies** are gated by Glen's review - the interpretation auto-shows; (2) showing a person's analysis must not leak one person's health data to another who merely typed their email.

## Goal

Scan -> a locally-produced draft (interpretation + ranked remedies) pushed to the server -> the server stores it AND emails the owner a **magic link** (notifies + verifies ownership) -> the visitor opens it and **immediately sees the interpretation** with **all remedies blurred** -> Glen **approves/edits** the remedies in a console, which **un-blurs the top remedy free** -> viewing sets the `biofield` gate. No Stripe, no FMP, no server-side scan interpretation in 4a.

## Scope (#4a)

The **server side**: a draft-ingest endpoint that stores the draft and sends the "ready" email + magic link on arrival; a `biofield_reveals` store (interpretation + ranked remedies + a `first_approved` flag); a console review surface (edit interpretation/remedies, approve = un-blur the top remedy); a token-verified reveal page (interpretation always shown; remedies blurred, top un-blurred once approved; the rest blurred with a stub unlock CTA); and setting the `biofield` gate on view. Plus the **JSON contract** for the local matcher push.

**Out of scope:**
- The **$1 charge / card-vault / unblur of the deeper remedies** -> **4b**. 4a renders all-but-the-approved-first blurred with a stub CTA ("unlocking soon").
- **FMP** (`fmp_biofield`) -> dropped; being retired.
- **Server-side scan interpretation / Pinecone ranking** -> the local matcher produces the draft.
- The local matcher agent + its push step (Glen runs it); 4a pins only the JSON contract.

---

## Confirmed decisions (Glen, 2026-06-20)

- **The interpretation shows automatically** - no approval gate on it. Only the **matched remedies** are gated.
- **All matched remedies start blurred.** Glen's **manual approve/edit un-blurs the top (first) remedy free**; the rest stay blurred until the **$1 trial** (4b).
- **One email, on draft arrival:** the moment the matcher pushes the draft, the server emails the magic link ("Your Biofield Analysis is ready"). No second email on approval (the top remedy just un-blurs on the visitor's next view).
- **Engine = E4L, not FMP**; the **local matcher pushes the finished draft** (least PHI on the server).
- **Free top reveal = the surface / #1-priority remedy.**
- **ToS gate at the reveal page:** the magic link proves ownership; a verified visitor who is NOT yet a member (no `tos_agreed_at`) must agree to ToS before ANY interpretation/remedies render (the interpretation is withheld server-side until member). Agreeing fires `unlock('tos', {email, tos:true})` for their email.
- **The free top-remedy unblur is a one-time offer per member, on request (a button).** After Glen approves the top (`first_approved`), the reveal shows a "Reveal my top match (free)" button ONLY if the member has not already used their one free unblock. Clicking it (the request) un-blurs the top AND records a per-member ledger row (consumes the one-time). Any later reveal for that member shows the top blurred; the only unblur path is then the $1 trial (4b).
- **Server-withholding (anti-bypass):** the reveal page never receives blurred remedy details. The top remedy details are sent only after the free unblock is granted (or, in 4b, after the $1). The deeper remedies' details are withheld entirely until 4b. The page only ever knows a blurred COUNT.
- **Live page, no feature flag** for the reveal route; the console is staff-gated. No emoji, no em dashes.

---

## Architecture

### 1. Draft ingest — `POST /api/e4l/reveal-draft`
Auth: `X-Cron-Secret` (== `CRON_SECRET`, falls back to `CONSOLE_SECRET`) - same un-widened pattern as `/api/e4l/scan-freshness`. Body:
```
{ "email": "...", "scan_date": "YYYY-MM-DD",
  "interpretation": { "greeting": "...", "body": "the reading (markdown/plain)" },
  "remedies": [ { "name": "...", "slug": "...", "meaning": "one warm sentence" }, ... ],  // ranked; [0] is the top
  "source": "e4l-matcher" }
```
Validates `email`, `scan_date`, and at least one of `interpretation` / `remedies`. Upserts a `biofield_reveals` row keyed by `(email, scan_date)`. **On the FIRST insert** for an `(email, scan_date)`: mint a magic-link token (`secrets.token_urlsafe`), store its hash on the row, INSERT an `auth_tokens` row (purpose `biofield_reveal`, 30-day TTL), and **email the owner** the "ready" + magic-link note (best-effort; a send failure never fails the ingest). A **re-push** updates `interpretation`/`remedies` only while `first_approved = 0` (Glen has not yet approved the top remedy); it does NOT re-mint the token or re-send the email, and never overwrites once approved. Always returns 200 on a valid push (or 400 on a missing email/scan_date/content); never raises.

### 2. Store — `dashboard/biofield_reveals.py`
```
biofield_reveals (
  id INTEGER PK, email TEXT, scan_date TEXT,
  interpretation_json TEXT,   -- {greeting, body} (auto-shown)
  remedies_json TEXT,         -- [{name, slug, meaning}, ...] ranked; all blurred until approved
  first_approved INTEGER DEFAULT 0,   -- 1 = top remedy (remedies[0]) un-blurred free
  token_hash TEXT,            -- minted at first ingest (the magic link)
  approved_at TEXT, approved_by TEXT,
  created_at TEXT, updated_at TEXT,
  UNIQUE(email, scan_date) )
```
Functions: `init_table(cx)`; `upsert(cx, email, scan_date, interpretation, remedies, source) -> (id, is_new)` (updates content only while `first_approved=0`; `is_new` True only on first insert so the caller mints token + sends email exactly once); `set_token(cx, id, token_hash)`; `set_interpretation(cx, id, interpretation)` / `set_remedies(cx, id, remedies)` (console edits); `approve_first(cx, id, by) -> bool` (sets `first_approved=1`, stamps approver); `list_pending(cx)` (rows with `first_approved=0`, newest first); `get(cx, id)` / `get_by_token_hash(cx, th)`. Getters use a per-cursor Row factory (never mutate the connection's `row_factory`). Kept separate from `portal_biofield_reports`.

### 3. Console review — `/console/biofield-reveals`
Staff page (inline `CONSOLE_SECRET` gate, matching sibling console routes) listing `list_pending` rows; per row: the interpretation (editable greeting/body) and the ranked remedies (editable; the FIRST is the one that un-blurs on approve), email/scan_date, and an approve button. Actions on the dispatch spine (`dashboard/actions.py`/`rbac.py`, the `/api/action/<key>` route), RBAC OWNER+OPS:
- `biofield_reveal.edit` — update `interpretation_json` and/or `remedies_json` (stays pending).
- `biofield_reveal.approve` — `approve_first()` -> `first_approved=1` (un-blurs the top remedy). NO email here (the "ready" email already went out at ingest).
A new `dashboard/biofield_reveal_actions.py` with `configure(**kw)` and idempotent `register()`. A new `static/console-biofield-reveals.html` modeled on `console-sales-pages.html`; a "Biofield Reveals" console nav sub-tab.

### 4. The "ready" email + magic link (at ingest)
Sent from the ingest endpoint on first insert: mint `token = secrets.token_urlsafe(32)`, store `_hash_token(token)` on the row, INSERT an `auth_tokens` row `(token_hash, email, purpose="biofield_reveal", created_at, expires_at=+30d)`. Email via `_send_inquiry_email`: "Your Biofield Analysis is ready" + `{PUBLIC_BASE_URL}/begin/biofield/<token>`. (Copy provisional; Glen voice; no emoji/em dash.)

### 5. One-time free-unblock ledger (`dashboard/biofield_reveals.py`)
A second tiny table enforces "one free top-remedy unblock per member, ever":
```
biofield_free_unlocks ( email TEXT PRIMARY KEY, reveal_id INTEGER, granted_at TEXT )
```
Functions: `init_free_unlocks(cx)`; `free_unlock_reveal_id(cx, email) -> int|None` (the reveal this member's free unblock was granted on, or None); `record_free_unlock(cx, email, reveal_id) -> bool` (INSERT OR IGNORE; True only on the first grant for that email).

### 6. Reveal page — `GET /begin/biofield/<token>`
Verifies the token against `auth_tokens` (purpose `biofield_reveal`, not expired; **not consumed** - reopenable, 30-day TTL) AND resolves the `biofield_reveals` row via `get_by_token_hash`. Then computes membership + unlock state for the row's email:
- `member = is_member(email=row.email)`.
- `top_unlocked = first_approved and free_unlock_reveal_id(email) == row.id` (this member already spent their free unblock on THIS reveal).
- `free_available = first_approved and free_unlock_reveal_id(email) is None` (free unblock still available -> show the button).

Serves `static/begin-biofield.html` (no-store) with a JSON-escaped `window.__REVEAL__` payload (escape `<`,`>`,`&`; `null` on invalid token):
- **Not a member:** payload `{ needs_tos:true, email }` ONLY - no interpretation, no remedies. The page shows the **ToS gate**; agreeing posts `/begin/unlock(tos, email)` and reloads. The `biofield` gate is NOT set yet.
- **Member:** payload `{ interpretation, top, blurred_count, first_approved, free_available, top_unlocked }`. `interpretation` always shown. `top` = the top remedy `{name, meaning, buy_url}` ONLY when `top_unlocked` (else null). `blurred_count` = remedies still hidden. The page: interpretation; then if `top_unlocked` show the top free; elif `free_available` show the **"Reveal my top match (free)" button**; elif `first_approved` (free already used) show the **$1 unlock** stub; else a calm "Your top match is being finalized" note. The deeper remedies always show as a blurred count with the **"Unlock your full Biofield Analysis"** CTA (stub; 4b). On a member view, fire `_record_entry_unlock("biofield", row.email)` -> **Find step 2 fills**.

An invalid/expired/missing token -> the friendly "this link is no longer valid" page, `window.__REVEAL__ = null`, NO personal data, gate not set.

### 7. Free-unblock request — `POST /begin/biofield/<token>/reveal-top`
The button target. Verifies the token + resolves the row; requires `member` (else `{ok:false, reason:"tos"}`); requires `first_approved` (else `reason:"pending"`); if `free_unlock_reveal_id(email)` is already set (used) -> `{ok:false, reason:"used"}`; else `record_free_unlock(email, row.id)` and return `{ok:true, top:{name, meaning, buy_url}}`. The top remedy details leave the server ONLY through this granted response (anti-bypass). Idempotent: a second call by the same member returns `reason:"used"` (their one grant already recorded).

### Reuse / untouched
- `auth_tokens` + `_hash_token`; `_send_inquiry_email`; the console dispatch spine + `/api/action/<key>`; `_record_entry_unlock` (#3), `is_member`/ToS, `journey_state`.
- Untouched: `fmp_biofield`, `portal_biofield_reports`, `surface()`/`/begin/explore`, `/begin/match*`, pricing/Stripe, `/api/e4l/scan-freshness` and sibling endpoints/auth.

---

## Data flow
1. Scan -> local matcher builds the draft (interpretation + ranked remedies) -> `POST /api/e4l/reveal-draft` -> `biofield_reveals` row + token minted + "ready" email sent (first insert only).
2. Visitor clicks the magic link -> `/begin/biofield/<token>` verifies ownership -> **interpretation shown immediately + all remedies blurred** ("top match being finalized") -> `biofield` gate set -> Find step 2 fills.
3. Glen opens `/console/biofield-reveals`, edits if needed, **approves** -> `first_approved=1`.
4. On the visitor's next view, the **top remedy is un-blurred free**; the rest stay blurred.
5. The $1 unlock of the rest = **4b**.

## Error handling
- Ingest: missing email/scan_date/content -> 400; content updates only while pending; never overwrites after approval; token minted + email sent exactly once (first insert), both best-effort and never failing the 200; wrapped so a bad row never 500s a batch.
- Reveal: invalid/expired/missing token -> friendly non-PHI page (`window.__REVEAL__ = null`); token never consumed; `no-store` headers.
- `_record_entry_unlock("biofield", ...)` wrapped/idempotent (#3) -> never blocks the page.
- Approve flips a flag only; safe to re-approve (idempotent).

## Testing
- **Ingest:** valid push -> row by `(email, scan_date)` with interpretation + remedies; first insert mints an `auth_tokens` row (purpose `biofield_reveal`) and calls the injected send fn ONCE; a re-push updates content while pending but does NOT re-send/re-mint; a re-push after approval does not overwrite; missing fields -> 400; un-widened `CRON_SECRET` auth (set `CRON_SECRET` in the test).
- **Store:** `upsert` returns `is_new` correctly; `set_interpretation`/`set_remedies` update while pending; `approve_first` flips the flag; `list_pending` excludes approved; `get_by_token_hash`.
- **Console action:** `biofield_reveal.approve` flips `first_approved` (no email); `edit` updates content and stays pending; RBAC OWNER/OPS.
- **Reveal route:** a valid token (pending) -> 200 with the interpretation present and ALL remedies blurred (top NOT free); after approval -> the top remedy un-blurred free, rest blurred; sets the `biofield` gate; an invalid token -> friendly page, no interpretation/remedy text, gate NOT set; `no-store` header; `window.__REVEAL__` JSON-escaped.
- Front-end (blur visuals, the "being finalized" note, the stub CTA, responsive) = manual visual pass. Server has unit + Flask-test-client coverage.
- deploy-chat test isolation (tmp `LOG_DB`; init the table + `auth_tokens` + journey tables; mock the send fn; mock GHL on free-tier transition). No emoji; no em dashes.

## Notes
- **Live page, no flag** for the reveal route; the console is staff-gated. Mostly dark in practice until (1) the local matcher pushes drafts and (2) Glen approves the top remedy - the interpretation + blurred remedies show on the emailed link, but a visitor sees nothing until a draft for their email is pushed.
- The **$1 unlock CTA is a stub** in 4a; 4b makes it the tripwire that unblurs the rest + converts to $99/mo.
- All copy provisional -> BNSN site pass later.
- Privacy: the magic link is the ownership proof; no stored reveal (interpretation or remedies) is shown from a merely-typed email. ToS-membership still required.
