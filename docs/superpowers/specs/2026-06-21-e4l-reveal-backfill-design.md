# Batch Backfill Begin-Reveal Drafts (sub-project C)

**Date:** 2026-06-21
**Status:** Approved (design); ready for implementation plan
**Repos:** deploy-chat (the ingest `notify` flag) + the AI-Training vault (`02 Skills/`, the `run` change + the backfill script).
**Parent:** Final sub-project of the matcher activation arc ([[project_e4l_reveal_push]]): A (console enrichment) and B (auto-push on scan arrival) are done. C is a one-time, controllable backfill that seeds reveal drafts for recent scans without emailing anyone.

---

## Problem

Sub-project B auto-pushes a reveal only for clients who engage AFTER it goes live. Clients who scanned recently but before B (or who never engage) have no reveal draft. Glen wants to seed the console with the recent cohort so he can review them - but the reveal push emails the client a magic link on every new draft, so a naive batch would be a mass email blast to past scanners.

## Goal

A local backfill that, for each client whose **latest scan is within a recent window** (default 10 days), synthesizes once and **silently** seeds a reveal draft (no email), so the drafts appear in `/console/biofield-reveals` for review. Dry-run by default; outreach happens later, separately.

## Scope

- A `notify` flag on the reveal ingest: `notify=false` stores the draft + mints the token but skips the "your analysis is ready" email.
- `build_payload` + `e4l-reveal-push.py run` thread a `notify` value through.
- A new `e4l-reveal-backfill.py`: select the recent-scan cohort from `e4l.db`, dry-run preview by default, `--push` to silently seed each via `run(..., notify=False)`.

**Out of scope:** any outreach/send mechanism for the seeded drafts (Glen contacts them later, manually or via a future send step); approval-time email; changing B; the engaged-only gate (C is engagement-agnostic - safe because it never emails).

---

## Confirmed decisions (Glen, 2026-06-21)

- **Silent seed: no emails.** The backfill creates drafts only; clients are not notified.
- **Cohort = most recent scan per client, only if that latest scan is within the window** (Glen: "the most recent scan from each client only if it is less than 10 days ago"). Window is a `--days` flag, default 10. (Today's count: 15 clients.)
- Dry-run by default (lists the cohort, synthesizes nothing); `--push` to seed.
- Reuse the proven engine (`e4l-reveal-push.py run`) - the backfill is a thin selection + loop wrapper.
- No emoji, no em dashes.

---

## Architecture

### 1. Ingest `notify` flag - `api_e4l_reveal_draft` (app.py, deploy-chat)
Read `notify = bool(data.get("notify", True))` (near the other `data.get` reads). Gate the existing first-insert email on it: change `if is_new:` (the `_send_inquiry_email(email, "Your Biofield Analysis is ready", ...)` block, ~line 10832) to `if is_new and notify:`. Everything else - the token mint, the auth_tokens row, the upsert, the guardrail - is unchanged, so the draft is fully reachable later; only the email is suppressed. Default `notify=True` preserves today's behavior for the matcher and for B.

### 2. `notify` pass-through (vault)
- `e4l_reveal_lib.build_payload(content, email, scan_date, label_map=None, notify=True)`: include `"notify": bool(notify)` in the returned payload. Default `True` keeps every existing caller unchanged.
- `e4l-reveal-push.py run(email, scan_id=None, layers=6, top=12, push=False, notify=True)`: pass `notify=notify` into `build_payload`. The CLI keeps its current default (notify True); the backfill calls `run(..., notify=False)`.

### 3. Backfill script - `02 Skills/e4l-reveal-backfill.py` (new)
- `select_cohort(cx, days, today=None) -> [(email, latest_scan_date), ...]`: group `e4l_scans` by client joined to `e4l_clients` (email non-empty), take `MAX(scan_date)` per client, keep those with `latest >= (today - days)`. `today` defaults to `date.today()` (injectable for tests). Ordered by latest scan desc.
- `main()`: args `--days` (default 10), `--push` (default off = dry-run), `--top`/`--layers` (passed to `run`), `--sleep` (default ~2s between pushes).
  - **Dry-run:** print each `(latest_scan_date, email)` and the count; synthesize nothing; exit.
  - **`--push`:** fail fast if neither `CONSOLE_SECRET` nor `CRON_SECRET` is set. For each cohort member, call `e4l_reveal_push.run(email, push=True, notify=False, top=..., layers=...)` wrapped in try/except (log + continue on any error), sleeping `--sleep` between. Print a final summary (seeded / skipped counts).
  - Imports `run` from the hyphenated `e4l-reveal-push.py` via `importlib` (as the tests do), or via a thin re-export; the plan picks one.

### Reuse / untouched
- `e4l_reveal_lib`, `e4l-reveal-push.py run`, `e4l_synthesis`, the reveal endpoint's upsert/guardrail/token mint - all reused; only the email is gated by `notify`.
- B's trigger scripts, the portal flow, approvals - untouched.

---

## Data flow
1. `e4l-reveal-backfill.py --days 10` lists the cohort (latest scan within 10 days) from `e4l.db` - no synthesis.
2. `--push`: for each client, `run(email, push=True, notify=False)` synthesizes the latest scan and POSTs the reveal payload with `notify:false`.
3. The ingest stores the draft, mints the token, and - because `notify=false` - sends no email. The draft appears PENDING in the enriched console (sub-project A) for Glen.
4. Glen reviews; outreach to seeded clients is a later, separate step.

## Error handling
- `notify=false` is the guarantee of no email; a test asserts the ingest does not call the sender when `notify=false`.
- Per-client failure (no scan, empty synthesis, network) -> logged, skipped; the batch continues.
- Empty cohort -> "0 clients in window", clean exit.
- Missing secret on `--push` -> fail fast before any work.
- Re-run safe: upsert by (email, scan_date) + `notify=false` -> updates pending drafts, never emails, never duplicates.

## Testing
- **Ingest (deploy-chat, `tests/test_biofield_layers.py`):** a push with `notify=false` stores the reveal but does NOT call `_send_inquiry_email`; a push with `notify` omitted (or true) still sends (existing behavior). (Mock `_send_inquiry_email`; assert call/no-call.)
- **`build_payload` notify (vault, `tests/test_e4l_reveal_lib.py`):** `notify=False` -> `payload["notify"] is False`; default -> `True`.
- **`run` pass-through (vault, `tests/test_e4l_reveal_push.py`):** `run(push=True, notify=False)` posts a payload whose `notify` is `False` (mock `post_reveal_draft`).
- **Cohort selection (vault, `tests/test_e4l_reveal_backfill.py`, new):** a fixture `e4l.db` with clients scanning at varied dates -> `select_cohort(days=10, today=<fixed>)` returns only those whose LATEST scan is within the window (a client with a recent old-plus-newer scan included by the newer; a client whose latest is outside excluded). Dry-run prints them; `--push` calls `run(..., notify=False)` once per client (mock `run`).

Run (deploy-chat): `doppler run -p remedy-match -c prd -- env DATA_DIR=$(mktemp -d) ~/.venvs/deploy-chat311/bin/python -m pytest tests/test_biofield_layers.py -v`
Run (vault): `cd ~/AI-Training && ~/.venvs/deploy-chat311/bin/python -m pytest "02 Skills/tests/test_e4l_reveal_lib.py" "02 Skills/tests/test_e4l_reveal_push.py" "02 Skills/tests/test_e4l_reveal_backfill.py" -v`

## Notes
- The deploy-chat `notify` change is additive and safe to ship on merge (default true = no behavior change). The backfill is a local tool; nothing auto-runs.
- Operational: after the `notify` flag is live, run `e4l-reveal-backfill.py --days 10` (dry-run) to confirm the 15, then `--push` to silently seed them; review in the console.
- Window is a parameter, so older tiers can be seeded later with a larger `--days` (still silent).
- DATA GAP (from A): some E4L item codes have no readable name in `e4l_items`; their stress factors show as codes until those names are filled.
