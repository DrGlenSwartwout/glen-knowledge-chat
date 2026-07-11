# Portal scan history + orders, with managed "current" scan — design

**Date:** 2026-07-11 · **Status:** design for review · **Author:** Glen + Claude

The client portal (`/portal/<token>`) today renders one biofield analysis and a small "History & receipts" strip. This change turns the portal into three tabs — **Current Analysis · Scan History · Orders & Invoices** — so every past scan and every invoice is browsable, while a managed "current scan" pointer decides what shows by default. Each new scan auto-becomes current (client can opt out); notifying the client stays a deliberate, human-confirmed action on a safe send channel.

Most of the machinery already exists; this is mostly **surfacing and organizing**, plus a console set-current control, two client preferences, and a backfill.

## Background trace (verified in code, all `app.py` unless noted)

- Portal shell `/portal/<token>` → `client_portal_page()` (`app.py:16179`) serves `static/client-portal.html` (single-page stacked cards).
- Data endpoint `/api/portal/<token>` → `api_client_portal()` (`app.py:16184-16530`).
- Scan bodies are **denormalized one row per `(email, scan_date)`** in `portal_biofield_reports` (`dashboard/portal_biofield_reports.py`, `content_json`, `status`). Legacy fallback = inline `client_portals.content_json` (`app.py:16263-16266`).
- **Scan selection** (`app.py:16244-16267`) precedence: `?scan_date=` query param → `content.current_scan_date` → newest (`list_report_dates()` is `ORDER BY scan_date DESC`, `dashboard/portal_biofield_reports.py:66-70`).
- `current_scan_date` is **a JSON key in `client_portals.content_json`**, not a column. Republish sets it: `POST /api/console/biofield-portal` → `api_console_biofield_publish()` (`app.py:18022`) writes both stores and sets `current_scan_date` (`app.py:18038-18039`).
- Layers un-blur only when the report is `confirmed` AND the client is entitled — `_portal_biofield_unlocked()` (`app.py:16272`, gate `app.py:10710`).
- Portal ↔ scans linked **by lowercased email**. Family view already exists: `?member=<email>` re-points `email_for_reports` when `household.can_view(primary, member)` passes (`app.py:16211-16236`).
- Client-viewable invoices **already exist**: `/invoice/<token>` → `invoice_page()` (`app.py:35421`), records in the `orders` table; `api_client_portal` already emits `payload["invoices"]` + `payload["past_invoices"]` with `/invoice/<token>` links (`app.py:16447-16456`), rendered in "History & receipts" (`static/client-portal.html:875-890`).
- Existing scan-date **switcher pills** `.scantabs`/`.scantab` reload a past scan via `?scan_date=` (`static/client-portal.html:1490-1495`).
- Sub-page precedent: `/portal/<token>/program` (`app.py:17373`, `static/portal-program.html`), flag-gated.

## Decisions (locked with Glen 2026-07-11)

1. **Approach A** — a tab bar over the existing cards in `client-portal.html`, client-side show/hide. No new renderer, no new server pages.
2. **Current-scan model** — single lever `current_scan_date`. Resolution precedence: transient `?scan_date=` → console set-current → client pin (opted out of auto-advance) → newest.
3. **Auto-advance** — per-client flag `auto_advance` (default **on**). A newly ingested confirmed scan auto-becomes current unless the client opted out; opt-out only stops *future* ingests from moving the lever. Console set-current always writes the lever directly.
4. **Notify** — mandatory **confirm-to-send** operator gate (nothing auto-emails an unconfirmed scan) AND per-client `notify_opt_in` (default **on**, opt-out sticks permanently). Sends route through **Mailgun**, never consumer Gmail, and stay dark until Mailgun is wired. See [[reference_gmail_send_cap]].
5. **Family grouping** — Scan History groups by household member via the existing `household.can_view` + `?member=` allowlist (explicit, not fuzzy name/email matching). Solo clients collapse to one ungrouped list. See [[reference_portal_seed_email_vs_contact]].
6. **Orders & Invoices** — reuse the existing payload + `/invoice/<token>` page; wiring only.
7. **Rollout for everyone** — backfill so every client's `portal_biofield_reports` rows populate their history; both flags default on. Whole feature ships behind one OFF flag.
8. **Phase 0** — repoint Sasha's live portal to the newer infoceutical scan first (see §7), as the immediate ask that motivated this.

## 1. Data model — two new preference keys

Add to `client_portals.content_json` (JSON, no schema migration):

- `auto_advance` (bool, default `true` when absent).
- `notify_opt_in` (bool, default `true` when absent).

`current_scan_date` already exists. Read defaults defensively (absent = the default) so un-backfilled portals behave correctly. No new tables; `household` relationships already exist.

## 2. Scan selection — extend the existing resolver (`app.py:16244-16267`)

Unchanged precedence order; only the middle term's meaning is formalized: `current_scan_date` is now written by **three** paths — console set-current (§4), client pin (§5), and auto-advance on ingest (§3). The resolver code does not change; it already honors `current_scan_date` then newest. Add one guard: if `current_scan_date` names a date with no row in `portal_biofield_reports` for the resolved email, fall through to newest rather than the legacy inline body, and log it (prevents a dangling pointer silently showing stale inline content).

## 3. Ingest + auto-advance (`api_console_biofield_publish`, `app.py:18022`)

On publishing/ingesting a scan into `portal_biofield_reports`:

- If the portal's `auto_advance` is on (default), set `current_scan_date = <new scan_date>` (existing behavior at `app.py:18038` becomes conditional on the flag).
- If the client opted out, write the report row but **do not** move `current_scan_date`.
- Enqueue a **pending notification** record (scan `email` + `scan_date`) rather than sending. Sending is §6.

## 4. Console set-current control

Expose an operator action to set `current_scan_date` to any existing `(email, scan_date)` for a portal — authoritative, independent of `auto_advance`. Reuse `api_console_biofield_publish` if it already targets an existing date, or add a thin `POST /api/console/portal/set-current` `{email, scan_date}` that writes only the pointer (no report rewrite) and validates the row exists. Console UI: a "Set as current" affordance on each scan in the operator's per-client view.

## 5. Client preferences + pin

- `GET`-exposed in `api_client_portal` payload: `auto_advance`, `notify_opt_in`, and the resolved `current_scan_date`.
- `POST /api/portal/<token>/prefs` `{auto_advance?, notify_opt_in?, pin_scan_date?}` — token/cookie identity via the existing `resolve_identity`/`_portal_record_for` path (`app.py:15877`). `pin_scan_date` sets `current_scan_date` and turns `auto_advance` off in one action ("keep showing me this one"). `notify_opt_in=false` is permanent until the client turns it back on.
- Client-facing toggles live in the Current Analysis tab footer: "Automatically show my newest analysis" and "Email me when a new analysis is ready."

## 6. Notification — confirm-to-send, Mailgun, flag-dark

- New OFF flag `PORTAL_SCAN_NOTIFY_ENABLED` (see §9). While off, notifications are never sent (pending records accumulate harmlessly).
- Operator **confirm-to-send**: a one-click console action per pending scan. On confirm, if `notify_opt_in` and the flag is on, send the "your new analysis is ready" email **via Mailgun** (`references` the Mailgun bulk path from [[reference_gmail_send_cap]] item #3), not `inbox.send_email`. Link is the portal URL (canonical `myhealingoasis.com` per PORTAL_BASE_URL split).
- No auto-send path exists at all until Mailgun is confirmed wired; document that as the flip prerequisite. This is the guard against repeating today's stale-scan send and the invoice-starving cap incident.

## 7. Phase 0 — Sasha portal repoint (immediate)

Sasha is Karin's cat; her July 2 scan (scan_id 1037250, FF-based) is superseded by a newer infoceutical scan (July 9 seed, scan_id 1035975). See [[feedback_biofield_ff_vs_infoceutical_currency]], [[reference_portal_seed_email_vs_contact]].

Procedure (verify before writing a live portal):

1. Determine which email Karin's portal token resolves to and which email/`scan_date` the infoceutical scan lives under in `portal_biofield_reports` (cat login `permanentlyyours777@hawaiiantel.net` vs Karin's mailbox `permanentlyyours@hawaii.rr.com`). **Open identity question to resolve at execution** — may require household `?member=` routing if the subjects differ.
2. If the infoceutical scan has no `portal_biofield_reports` row in prod, ingest/push it first (E4L scan push scripts — see [[reference_scan_pull_feature]] / [[feedback_e4l_ingest_needs_prod_push]]).
3. Set `current_scan_date` to that scan's date (console set-current).
4. Render-verify the live portal (headless Chrome) before considering it done — see [[feedback_render_the_page_not_the_payload]].
5. Notifying Karin is a separate, later confirm-to-send once §6 is live; do not auto-send.

## 8. Frontend — three tabs (`static/client-portal.html`)

- Add a tab bar (Current Analysis · Scan History · Orders & Invoices); group existing cards under the first, toggle visibility client-side (no router; matches Approach A).
- **Scan History**: promote the existing scan list; group rows by household member (name label per subject) using the `?member=` data already available; badge the current scan; each row reopens via `?scan_date=` (+ `?member=`). Locked scans keep the existing blur/unlock gate.
- **Orders & Invoices**: render `payload.invoices` (open/pay) and `payload.past_invoices` (receipts), each linking to `/invoice/<token>` — reusing the data already in the payload; move the current "History & receipts" content here.
- Empty states: "No past analyses yet" / "No invoices yet."

## 9. Flags

- `PORTAL_SCAN_HISTORY_ENABLED` — gates the whole three-tab UI + prefs endpoints. Dark by default.
- `PORTAL_SCAN_NOTIFY_ENABLED` — gates confirm-to-send emails; stays off until Mailgun is wired.

Durable flip = `doppler secrets set <FLAG>=1 -p remedy-match -c prd` (Doppler is source; Render is pruned — [[reference_prod_flags_deleted_not_off]], [[feedback_doppler_is_source_render_mirrors]]).

## 10. Testing

- Selection precedence: `?scan_date=` > console set-current > client pin > newest; dangling `current_scan_date` falls to newest (not legacy inline) and logs.
- Auto-advance on: new confirmed scan moves the pointer. Off: pointer unchanged, scan still appears in history.
- Notify gate: no send without confirm; no send if `notify_opt_in=false`; no send while flag off; send channel is Mailgun (assert, don't hit Gmail). Opt-out persists.
- Family grouping: a portal never surfaces a scan for an email outside its `household.can_view` allowlist (no stranger/commingled scans).
- Entitlement: historical (non-current) scans respect `_portal_biofield_unlocked` exactly as the current one.
- Backfill idempotency: re-running populates history without duplicating rows or resetting a client's opted-out pointer.
- Existing `/invoice/<token>` and `api_client_portal` externally observable behavior unchanged for a client with one scan and no prefs set.

## Out of scope

- Mailgun integration itself (tracked separately; this feature consumes it once ready).
- Any change to how scans are captured/parsed upstream (E4L ingest).
- Merging distinct subjects — family grouping *displays* multiple subjects under one portal via the existing household allowlist; it never merges identities. See [[feedback_identity_merge_review]].
