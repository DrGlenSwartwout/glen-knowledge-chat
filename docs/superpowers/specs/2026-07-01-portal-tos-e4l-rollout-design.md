# Personal Portals + Combined RM/E4L TOS + E4L Onboarding Rollout

**Date:** 2026-07-01
**Status:** Draft for review
**Owner:** Glen (illtowell.com / deploy-chat)

---

## Goal

Every known person gets a personal portal. First access is gated by a single
Terms agreement that covers **both** Remedy Match and **Energy For Life (E4L)**.
Anyone without an E4L account is offered automated E4L onboarding. Portal links
are emailed out — **today to the Zoom attendees**, then rolled out to the full
roster **at a comfortable, throttled pace**.

## Success criteria

- A returning attendee clicks their emailed link, sees the combined RM+E4L TOS
  on first access, agrees once, and lands in their portal.
- A portal holder with no E4L account is offered a one-tap E4L setup
  (GHL onboarding + prefilled `truly.vip/E4L` link).
- The full roster (FileMaker ∪ e4l.db ∪ inbound_leads, deduped) is provisioned
  with portals and drains out over days via GHL — never a Gmail burst.

## Non-goals (explicitly deferred)

- **Full hands-off E4L account creation** — no E4L create-account API exists.
  Best available = trigger onboarding + prefill; the client completes one step.
- **GHL-tagged PB-account ingestion** — GHL exposes only exact-email lookup
  today, and it is unverified that PB accounts are tagged in GHL. This is a
  later roster tranche (SP0-v2), not a blocker for today.
- **Client login / session auth** — portal token in the emailed link remains the
  credential (mirrors `/invoice/<token>`). `portal_identity` session branch stays
  scaffolded.

---

## Locked decisions

| Decision | Choice |
|---|---|
| Today's send | Zoom attendees only |
| Rollout | Full roster, throttled, over following days |
| Roster v1 source | On-disk: FileMaker CSV ∪ `e4l.db` ∪ `inbound_leads`, dedup by email |
| Roster v2 (later) | GHL-tagged PB accounts (needs GHL contact search + verified tagging) |
| Portal | Bulk-mint no-scan portals via existing `upsert`, idempotent |
| TOS | Combined RM + E4L, `BEGIN_TOS_VERSION` bumped, first-access gate |
| E4L onboarding v1 | Trigger GHL E4L workflow + prefilled `truly.vip/E4L` link |
| E4L onboarding v1.1 | Collect missing details form (fast-follow) |
| Send channel | GHL workflow (portal URL → GHL custom field → throttled workflow) |

---

## Existing building blocks (verified in code)

- **Portal mint:** `POST /admin/portal/upsert` (app.py ~12844), console-key auth
  (`_portal_console_ok`, `X-Console-Key`/`?key` vs `CONSOLE_SECRET`).
  `dashboard/client_portal.upsert_portal(cx, email, name, content)` →
  `client_portals` (token_hash SHA256; raw token cached in
  `portal_notify_state.portal_token`). No-scan default content
  `{"biofield_status":"pending"}`. Reissue: `POST /admin/portal/reissue-link`.
- **TOS gate:** frontend `static/client-portal.html` shows gate when
  `tos_agreed === false`; `POST /api/portal/<token>/agree-tos` →
  `begin_funnel.record_unlock(..., tos=True, tos_version=BEGIN_TOS_VERSION)` →
  `journey_state.tos_agreed_at/tos_version`. `BEGIN_TOS_VERSION="rm-tc-2026-05-28"`
  (app.py ~3052). TOS text is external (`illtowell.com/terms`); **E4L not mentioned.**
- **E4L data:** `~/AI-Training/e4l.db` (env `E4L_DB`), read-only.
  `e4l_clients(client_id, name, email)`, `e4l_scans`, etc.
  `biofield_e4l.scan_context(email, today)` already used in prod portal.
- **GHL onboarding:** `ghl_onboard_contact(email, first, last, ...)` (app.py ~8213)
  → upsert → `ghl_add_to_pipeline` (E4L Onboarding pipeline
  `A6LWJMBoIsOFBMeCa6NY`, stage `397c5fb2...`) → `ghl_enroll_workflow`
  (`0b02dd3e...`). GHL access is v1 exact-email lookup only (`/contacts/lookup`).
- **People hub:** `people(email UNIQUE, name, roles, tags, ...)` +
  `app.upsert_person` enrichment; `portal_identity.resolve_identity` is the choke point.
- **Email:** `_send_full_report_email` (Gmail API → SMTP → log),
  `send_portal_welcome_email` (GHL workflow → SMTP → log).

---

## Sub-projects

### SP0 — Roster assembly (ops script, no prod code change)

**Purpose:** Populate `people` with the full known-person universe, deduped.

- **Input v1:** `~/Downloads/fmp-clients-export.csv` (email, first, last, phone),
  `e4l.db` `e4l_clients` (name, email), `inbound_leads` (email, first/last, source).
- **Transform:** normalize emails (lowercase/trim), dedup by email, merge names
  (prefer non-empty), tag provenance (`tags`: `fmp`, `e4l`, `lead`).
- **Output:** upserts into `people` via `app.upsert_person` (or a thin script that
  hits the DB the same way). Idempotent — re-runnable.
- **Interface:** `scripts/roster_assemble.py` → writes/updates `people`; prints
  counts (new, updated, total, per-source, dropped-no-email).
- **v2 (deferred):** GHL contact search by tag → append PB-account tranche.

### SP1 — Bulk portal mint (ops script, no prod code change)

**Purpose:** Ensure every `people` row has a portal + a resolvable token.

- Loop `POST /admin/portal/upsert` (or call `client_portal.upsert_portal`
  directly if run in-process) for each email; **do not** set `send:true`.
- Idempotent: existing portal keeps its token (upsert, not reissue).
- Store/read the raw token from `portal_notify_state.portal_token` so SP4 can
  build each URL. Emit a manifest: `email, name, portal_url`.
- **Interface:** `scripts/portal_bulk_mint.py --limit N --dry-run`.

### SP2 — Combined RM + E4L TOS (code + PR) — **must ship before any send**

**Purpose:** First-access agreement covers both RM and E4L terms.

- Bump `BEGIN_TOS_VERSION` → e.g. `rm-e4l-tc-2026-07-01`.
- Gate copy: "By continuing you agree to the Remedy Match Terms **and** the
  Energy For Life (E4L) Terms." Two links: `illtowell.com/terms` + E4L terms URL
  (**input needed** — see Open inputs).
- No schema change: `journey_state.tos_version` already stores the version, so
  who-agreed-to-what is auditable by version string.
- **Files:** `static/client-portal.html` (gate copy + links), app.py
  (`BEGIN_TOS_VERSION`). Optional: surface `TOS_URL`/`E4L_TOS_URL` to the page.
- **Tests:** first-access shows gate; agree records new version; returning
  agreed-under-old-version behavior decided (default: old agreement still valid,
  no re-prompt — a re-consent sweep is out of scope).

### SP3 — E4L onboarding on first access (code + PR, fast-follow)

**Purpose:** Offer E4L setup to portal holders with no E4L account.

- **Detect:** email ∉ `e4l_clients` (via `biofield_e4l`, read-only) → "no E4L
  account" state. Email ∈ → skip (they already have scans).
- **v1 action:** button → `ghl_onboard_contact(email, first, last)` (enrolls E4L
  onboarding workflow) **and** open a prefilled `truly.vip/E4L` link with known
  name/email params. One tap; client finishes on E4L.
- **v1.1:** portal form collects missing details E4L needs (confirm the exact
  field set with E4L — likely name, email, phone; DOB/address TBD) before the
  onboarding trigger.
- **Files:** `static/client-portal.html` (E4L card), an API endpoint
  `POST /api/portal/<token>/e4l-onboard`, app.py wiring to existing GHL fns.
- **Tests:** in-e4l email → no card; not-in-e4l → card; onboard endpoint calls
  GHL onboard once, idempotent, never raises to the client.

### SP4 — Staged GHL send (code + PR)

**Purpose:** Email portal links deliverability-safely; attendees today, rest paced.

- Push each `portal_url` into a **GHL custom field** on the contact
  (`ghl_upsert_contact(..., custom_fields={PORTAL_URL_FIELD: url})`).
- Enroll into a **GHL workflow** that emails the portal link (workflow id +
  custom-field id are **inputs needed**).
- **Throttle:** batch by cohort. `--cohort attendees` (explicit list) sends now;
  `--cohort rollout --wave-size N --interval days` drains the remainder. Track a
  `portal_send_log(email, cohort, sent_at)` so no one is emailed twice.
- **Interface:** `scripts/portal_send.py --cohort attendees --emails-file ...`
  then `--cohort rollout --wave-size 150`.
- **Fallback:** if a contact fails GHL enroll, log and skip (don't fall back to a
  Gmail burst).

---

## Sequencing

```
Phase 1 (today):  SP2 (ship) ──▶ SP0-v1 ──▶ SP1 ──▶ SP4 attendees wave
                     │                                    ▲
                     └── gate must be live before ─────────┘ any client email
Phase 2 (days):   SP4 rollout waves (throttled)  +  SP3 v1 (fast-follow)
Phase 3 (later):  SP0-v2 GHL-PB tranche  ·  SP3 v1.1 details form  ·  re-consent sweep (if ever)
```

## Open inputs needed from Glen / to confirm

1. **Attendee list** for today's send (names+emails) — the `--cohort attendees`
   source. (DB segment or pasted list.)
2. **E4L terms URL** for the combined TOS link.
3. **GHL:** the workflow id that emails the portal link + the portal-URL custom
   field id (or create them).
4. **Prod `e4l.db` availability** — confirm the deployed app can read `e4l.db`
   for the SP3 in-e4l check (it already serves E4L scans in prod, so likely yes).
5. **E4L required fields** for account setup (for SP3 v1.1).

## Risks

- **Deliverability:** large sends must go via GHL, throttled; wave 1 today small.
- **GHL assumptions:** PB-account tagging in GHL is unverified → kept out of v1.
- **TOS legal text:** combined terms should be sanity-checked before blast.
- **Idempotency:** SP0/SP1/SP4 all re-runnable; `portal_send_log` prevents double-emails.
