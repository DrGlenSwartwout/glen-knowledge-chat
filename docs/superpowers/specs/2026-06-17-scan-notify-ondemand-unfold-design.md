# Spec: Scan notification + engagement-gated on-demand synthesis + live-unfold

**Date:** 2026-06-17
**Status:** Approved (design) — pending implementation plan
**Related:** [[project_ascension_pricing_model]] (speed-to-lead, adaptive-by-action, the teaser) · [[project_e4l_scan_ingestion]] · the blur-reveal feature (PR #156) + multi-scan history (#157) + synthesis fidelity. This **supersedes the current "auto-draft every scan"** (`E4L_AUTODRAFT_ENABLED`) with notify-then-process-on-engagement.

---

## Goal

The moment a client's E4L scan arrives, text + email them that their analysis is being prepared in their Healing Oasis. Only spend AI synthesis on clients who engage: pre-process for the already-engaged (instant on open), synthesize on-demand for cold clicks (the analysis **unfolds in front of them**, ~6s). Respect attention with a 3-message taper and full opt-in/out across SMS, email, and the portal.

## Scope

In scope: notification (Twilio SMS + SMTP email) from the local scan trigger; a unified notification-preference + engagement state; engagement-gated processing (pre-process engaged / on-demand cold) replacing the blanket auto-draft; the live-unfold portal UX. The synthesis itself + the blur-reveal/confirm flow are unchanged (the unfold lands a client at the existing **interested** state: patterns shown, remedies blurred pending Glen's confirm). **Out of scope:** tier-based access-gating / the $99 teaser-vs-full split (a separate layer; here *everyone who scans is notified* and the processing gate is engagement, not tier).

## Architecture

The synthesis stays **local** (it needs `e4l.db` + the FMP `ingredients.db` store + the catalog). The remote portal and the local Mac bridge through two small server-side queues + a shared client state. A per-client **engagement/preference** record drives notification eligibility and processing eagerness.

```
e4l scan email ─▶ LOCAL trigger ─┬─ send SMS (Twilio) + email (SMTP)  [if eligible]
                                 └─ engaged? ─yes─▶ pre-synthesize now ─▶ publish ai_draft
                                              └no──▶ wait
client taps link ─▶ portal opens ─┬─ ready?  ─▶ staged UNFOLD (patterns→layers; remedies blurred)
                                  └─ not ready (cold) ─▶ post process-request ─▶ "preparing…"+poll
LOCAL watcher (tight poll) ─▶ pending process-requests ─▶ synthesize (~6s) ─▶ publish ─▶ mark done
first portal open ─▶ mark engaged
```

## Components

### 1. Notification sender — local trigger (Twilio SMS + SMTP email)
Extend the existing `e4l-email-trigger` (local; it already detects new scans + resolves the client). On a new scan, look up the client's **phone + email** (People hub) and, **if eligible** (below), send **SMS via Twilio immediately** (primary) + **email via the existing SMTP sender** (backup), both carrying the portal link. New local dependency: Twilio creds (`TWILIO_ACCOUNT_SID`, `TWILIO_AUTH_TOKEN`, a from-number) in Doppler. Email includes an unsubscribe link; SMS relies on Twilio's STOP/START.

### 2. Unified notification preference + engagement state — server
One per-client record (table `portal_notify_state`: `email, phone, opt_status (default|in|out), notify_count, engaged (bool), updated_at`). All channels write the SAME `opt_status`:
- **SMS STOP** (Twilio inbound webhook) → `out`; **START** → `in`.
- **Email unsubscribe link** → `out`.
- **Portal opt-in/out button** → `in` / `out`.
`out` hard-suppresses all channels; `in` overrides the taper; `engaged` is set on first portal open.

### 3. Notification eligibility + taper
Before sending, the trigger checks eligibility (a console endpoint reading `portal_notify_state`): send when `opt_status != out` AND (`opt_status == in` OR `engaged` OR `notify_count < 3`). The **3 messages are varied** — (1) "your scan's in, analysis being prepared", (2) a different "ready and waiting" nudge, (3) **last-call**: "this is the last reminder — tap to keep getting these" (the portal opt-in CTA). After a successful send the trigger increments `notify_count`. Engaged or opted-in clients are notified on every scan; everyone else goes quiet after 3.

### 4. Engagement-gated processing — replaces auto-draft-all
- **Engaged + new scan:** the local trigger **pre-synthesizes immediately** (today's importer path, but gated to engaged clients) and publishes the dated `ai_draft` (patterns + interpretation; remedies blurred). Their click is instant.
- **Cold client:** nothing synthesized. The portal click posts a **process-request** (server queue `portal_process_requests`: `email, scan_date, status, requested_at`). A **tight local watcher** (a dedicated launchd job polling every few seconds, separate from the 15-min email-trigger) claims pending requests, runs the ~6s synthesis, publishes the `ai_draft`, and marks the request done.
- `E4L_AUTODRAFT_ENABLED`'s blanket behavior is retired in favor of this.

### 5. Live-unfold — portal UI
On portal open for a scan: if the `ai_draft` is **ready**, play a **staged reveal** — the stress patterns appear first (they come straight from the scan), then the layers assemble one by one (title → the stresses it balances). If **not ready** (cold), show "Preparing your analysis…", post the process-request, and **poll** until ready, then unfold. Remedies remain blurred with the existing **"Request my remedy matches"** → Glen's review → confirm → un-blur. First open marks `engaged`.

### 6. Opt-in/out surfaces
- **Portal button** — `POST /api/portal/<token>/notify-pref {in|out}`.
- **Email unsubscribe** — a tokenized `GET /unsubscribe` link → `out`.
- **Twilio inbound webhook** — `POST /sms/inbound` maps STOP/START → `out`/`in` (Twilio also enforces STOP at the carrier).

## Data flow
scan → eligibility check → SMS+email (varied by `notify_count`) → [engaged: pre-synthesize+publish] / [cold: wait] → client taps → portal: ready→unfold, else process-request+poll→unfold → patterns+layers reveal (remedies blurred) → request → Glen confirms → un-blur. First open sets `engaged`; STOP/unsubscribe/portal-toggle set `opt_status`.

## Error handling
- Twilio send fails → log, still send email (backup); never block ingestion.
- No phone on file → email only. No email → portal-only (link still works).
- `opt_status == out` → send nothing on any channel.
- Synthesis fails (cold request) → portal shows a graceful "we're finishing your analysis, we'll text you when it's ready" + the request stays pending for retry; never a broken page.
- Process-request watcher down → cold clicks queue and resolve when it resumes; the portal keeps polling/raincheck.
- Unconfirmed remedies never leave the server (existing blur), unchanged.

## Testing
- **Eligibility/taper (server, unit):** send-decision given `opt_status`/`engaged`/`notify_count`; `out` suppresses; `in`/`engaged` bypass the cap; quiet after 3.
- **Preference writes (server):** portal toggle, unsubscribe link, SMS STOP/START all converge on one `opt_status`.
- **Engagement:** first portal open flips `engaged`; pre-process only fires for engaged.
- **Process-request queue:** portal enqueues; watcher claims/marks-done idempotently; double-claim safe.
- **Notification sender (local):** Twilio + SMTP calls mocked; varied copy by `notify_count`; phone/email-missing fallbacks.
- **Unfold (portal, manual + JS check):** ready → staged reveal; cold → preparing+poll→reveal; remedies stay blurred.
- Deploy-chat pytest conventions; vault synthesis unit-tested; the live Twilio/SMS path validated with a manual test send.

## Phasing (for the plan)
1. **Notification + preferences + taper** (Twilio/SMTP from the trigger, `portal_notify_state`, opt-in/out across all 3 channels). Independently valuable; ships speed-to-lead.
2. **Engagement-gated processing** (engaged-flag, pre-process for engaged, process-request queue + local watcher, retire auto-draft-all).
3. **Live-unfold UX** (staged reveal + cold "preparing"+poll).

## Definition of done
A new scan immediately texts (Twilio) + emails the client a link to their Healing Oasis; non-engagers get 3 varied messages then quiet; clients can opt in/out via SMS STOP/START, an email unsubscribe link, or a portal button (one unified preference). AI synthesis runs only for engaged clients (pre-processed) or on a cold click (on-demand, ~6s), and the analysis **unfolds** — patterns then layers — with remedies blurred pending Glen's confirm. Everyone who scans is notified; processing is gated by engagement. Phased build; each phase tested and shippable.
