# Spec: E4L auto-draft + two-click blur-reveal + review + followup

**Date:** 2026-06-16
**Status:** Approved (design) — pending implementation plan
**Related:** [[project_ascension_pricing_model]] · [[project_e4l_scan_ingestion]] · [[project_unified_personal_portal]] · the E4L importer (`02 Skills/e4l-portal-import.py`, built) · the console biofield editor (PR #149/#155) · the hardened e4l-email-trigger.

---

## Goal

When a client's E4L voice scan ingests, an AI analysis lands in their portal **immediately** (disclosed as AI-only, remedies blurred). A **two-click** reveal lets them first understand their pattern dynamics, then explicitly **request** the Functional-Formulation (FF) plan — which is the only thing that pulls Glen into review. Behavior at each step emits a signal that fires **supportive** followup. This is the engine that makes the $99/mo analysis (and the free top-match teaser) real, while keeping Glen's review labor tiny and the actionable/liability content behind his confirmation.

## Decomposition — THIS spec is the buildable core

**In scope:** the auto-draft → portal state machine → two-click blur-reveal → Glen's review/confirm → GHL followup signals.
**Deferred (layer on once the membership system exists):** access gating (1/month included, additional $99, free-tier #1-match one-time + order-to-unlock-next), the offer/checkout for additional analyses, the `purchased` state (needs order linkage), auto-publish without the two clicks (AI-maturity gate). Hooks are designed so these slot in without reworking the core.

## State machine (the heart)

Per client (per latest scan), the biofield analysis carries a `status`:
```
ai_draft ── click 1 ──▶ interested ── click 2 ──▶ requested ── Glen confirms ──▶ confirmed
   (auto)   (reveal       (patterns +    (request    (review queue)   (un-blur FFs +
            patterns)     interp shown)  FF plan)                      notify client)
```
- **ai_draft** — auto-generated, in portal, FFs blurred, AI-disclosed.
- **interested** (click 1) — patterns + interpretation revealed; FFs still blurred ("request Dr. Glen's confirmed matches"). One click is fine — they may act on their own / wait for the covered monthly analysis.
- **requested** (click 2) — explicit ask → enters Glen's review queue.
- **confirmed** — Glen reviewed/edited the FFs → un-blurred → client notified.

## Architecture / components

### 1. Content model — `dashboard/client_portal.py`
The biofield analysis already lives in `client_portals.content_json` (`layers`, `greeting`). Add a top-level `content.biofield_status` (default `confirmed` for hand-built/legacy portals so existing portals are unaffected; E4L auto-drafts set `ai_draft`). The layers keep `remedy`/`dosing`; **blur is enforced server-side** (below), not by trusting the client.

### 2. Auto-draft on ingest — extend `02 Skills/e4l-email-trigger.sh` (local)
After a scan ingests (existing parse step), for each newly-ingested client run `e4l-portal-import.py --email <e>` → publish via `POST /admin/portal/upsert` with `biofield_status: "ai_draft"`. The importer already produces the content; this just publishes it draft. Idempotent per scan (skip if a draft for that scan already exists).

### 3. Blur enforcement + reveal endpoints — `app.py`
- `GET /api/portal/<token>/view` (+ the content endpoint): when `biofield_status != confirmed`, **omit `remedy`/`dosing` from the layer payload** and include `biofield_status` + `blurred: true`. The client literally never receives unconfirmed remedies (true blur, not CSS).
- `POST /api/portal/<token>/biofield/interest` — sets `interested` (click 1; idempotent), enqueues GHL tag `e4l:interested`. Returns the patterns+interpretation (titles/meanings), still no remedies.
- `POST /api/portal/<token>/biofield/request` — sets `requested` (click 2), enqueues GHL tag `e4l:requested`, adds to the review queue. (Token resolves identity via the slice-1 seam.)

### 4. Portal rendering (two-click) — `static/client-portal.html`
The biofield block reads `biofield_status`:
- `ai_draft` → CTA "View your scan analysis" (click 1 → `/interest` → re-render `interested`).
- `interested`/`requested` → show patterns + interpretation (titles/meanings); FF area rendered as a **blurred placeholder** ("Your remedy matches are being confirmed by Dr. Glen"). If `interested`, a "Request my remedy matches" button (click 2 → `/request`). If `requested`, "Requested — Dr. Glen is confirming."
- `confirmed` → full layers with remedies (the current rendering).
- Every state clearly labeled **AI-generated, pending clinician review** until `confirmed`.

### 5. Review queue + confirm — `app.py` + `static/console-biofield-portal.html`
- `GET /api/console/biofield/review-queue` (console-key) → clients at `requested` status (email, name, requested_at). A small list in the editor (or a console panel) linking each into the editor pre-filled.
- Glen edits in the existing editor → a **"Confirm & publish"** action sets `biofield_status = confirmed`, enqueues GHL tag `e4l:confirmed`, and notifies the client (email via `_send_full_report_email`; SMS via the GHL write-queue).

### 6. GHL followup signals — reuse the existing `ghl_write_queue`
Each transition enqueues a contact tag (`e4l:interested`, `e4l:requested`, `e4l:confirmed`). GHL workflows (authored in the GHL UI) fire the **supportive, non-FF-first** sequences: 1-click-only → therapies/tools/lifestyle/diet support for their priority functions; requested-no-purchase → same + nurture toward the covered monthly analysis. (The `no-purchase` branch is deferred with the purchase/membership state; the tags it needs are emitted here so GHL can branch later.)

## Data flow
scan ingests → importer → `/admin/portal/upsert` (ai_draft) → client opens portal → click 1 (`/interest`, GHL tag, patterns shown) → click 2 (`/request`, GHL tag, review queue) → Glen confirms in editor (status confirmed, GHL tag, notify) → client sees full remedies.

## Error handling
- Unconfirmed remedies are never sent over the wire (server omits them) — a client can't bypass the blur.
- No scan / importer failure → no draft published (portal unaffected).
- GHL enqueue failure → state still transitions (followup is best-effort; logged).
- Legacy/hand-built portals default to `confirmed` → render exactly as today (no regression).

## Testing
- **Unit/route:** view endpoint omits remedy/dosing when `status != confirmed` and includes them when `confirmed`; `/interest` and `/request` transition status + enqueue the right GHL tag (idempotent); review-queue returns only `requested`; "Confirm & publish" sets `confirmed` + notifies (monkeypatched). Legacy content (no `biofield_status`) treated as `confirmed`.
- **Integration:** full transition ai_draft → interested → requested → confirmed via the endpoints, asserting the payload blurs then reveals.
- Deploy-chat pytest conventions + isolation; the local auto-draft hook covered by the importer's existing tests + a manual run.

## Definition of done
A scan auto-drafts into the portal (AI-labeled, remedies blurred); click 1 reveals patterns+interpretation and signals interest; click 2 requests the FF plan and queues it for Glen; he confirms in the editor → remedies un-blur + client is notified; each step emits a GHL tag for supportive followup. Unconfirmed remedies never leave the server. Access-gating + purchase/offer deferred but hook-ready. Full suite green.
