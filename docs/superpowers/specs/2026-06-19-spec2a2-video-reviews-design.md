# Spec 2a-2 — Video Reviews: record + transcribe + AI-score → points

**Date:** 2026-06-19
**Status:** Approved (design); ready for implementation plan
**Repo:** deploy-chat (Flask, Render `glen-knowledge-chat`)
**Parent:** Spec 2a (reviews). Builds on 2a-1 (merged, PR #179). Next: 2a-2b (auto-trim), 2a-3 (gift), 2b (referral coupons).

---

## Problem

2a-1 captures video reviews (link/upload) but does not score them — video points are not awarded. 2a-2 adds **in-browser recording** and an **automated, Render-side pipeline** that transcribes an uploaded/recorded review video, AI-scores the transcript for quality + compliance, and credits the buyer's video points (up to the 5-point cap). Auto-trim is deferred to 2a-2b.

## Scope (2a-2)

An in-browser recorder on the review form (record/preview/re-record) that submits through the existing 2a-1 upload path; a background worker that, for each upload/record video review, transcribes it via the existing Whisper helper, AI-scores the transcript (0–5), and credits video points so the review total is `min(5, written + video)`. Length is gated: 90s default, 5 minutes after the buyer's first successful video review. All processing is Render-side and reuses existing infrastructure — no new external dependency.

**Out of scope:** auto-trim (→ 2a-2b); AI gift suggestion (→ 2a-3); referral coupons (→ 2b); scoring of **link** videos (we don't hold the file — link videos remain human-moderated, no auto points).

---

## Confirmed decisions (Glen, 2026-06-19)

- **Defer trim** — 2a-2 scores video for points only; auto-trim is 2a-2b.
- **Length:** 90 seconds default; **5 minutes (300s) once the buyer has a prior successful video review** (a video review that earned video points and was approved). Paid-member unlock deferred (the paid tier is not live); the length check is a single server-side function so adding a paid-member branch later is one line.
- **Points math:** video scored **0–5**; review total = **`min(5, written_points + video_points)`** (a great video alone reaches 5). `written_points` = the 2a-1 `ai_score`.
- **Processing:** Render-side background worker (the `_drain_sales_image_queue` + APScheduler pattern). Async — points arrive seconds after submit.
- **Flag:** new **`REVIEWS_VIDEO`** (default off), requires `REVIEWS_ENABLED` on. Lets text reviews go live before video, and gates the OpenAI-cost worker + record UI.

---

## Architecture

### Schema (extend 2a-1 `product_reviews`, additive)
Add columns (lazy `ALTER`, like Phase 5): `video_points INTEGER DEFAULT 0`, `transcript TEXT DEFAULT ''`, `video_status TEXT DEFAULT ''` (`''` | `pending` | `scored` | `failed`). `ai_score` continues to hold the **written** points. New functions in `dashboard/product_reviews.py`: `set_video_result(cx, id, video_points, transcript, status)`, `has_successful_video(cx, email) -> bool` (any of this email's reviews with `video_points > 0` and `status='approved'`).

### Video-job queue — `dashboard/review_video_jobs.py` (SQLite)
`review_video_jobs(review_id INTEGER PRIMARY KEY, status TEXT DEFAULT 'pending', enqueued_at TEXT, done_at TEXT)`. Functions `init_table`, `enqueue(cx, review_id)`, `claim_pending(cx, limit) -> [review_id]`, `mark(cx, review_id, status)`. Idempotent enqueue (PK = review_id; re-submit re-enqueues by resetting status to pending).

### AI video scoring — extend `dashboard/review_scoring.py`
`build_video_prompt(product, transcript) -> (system, user)` (pure) + `score_video(client, product, transcript, *, strip=lambda s: s) -> {"compliance_ok": bool, "reasons": str, "video_points": int(0..5), "recommend_publish": bool}`. Same fail-closed contract as `score_review` (any parse error/exception → safe default with `video_points=0`, `compliance_ok=False`). Quality rewards a clear, specific, authentic spoken experience; the compliance gate rejects disease-cure claims/PII/abuse. Dashes stripped from `reasons`.

### Enqueue (extend `POST /api/reviews`)
When `REVIEWS_VIDEO` is on and the submitted review has `video_kind in ("upload",)` (recorded videos arrive as uploads), after the synchronous written scoring, set `video_status='pending'` and `review_video_jobs.enqueue(rid)`. The response includes `video_status:"pending"`. (Link videos: no enqueue, no auto points.)

### Background worker — `_drain_review_videos()` (app.py, scheduler-registered)
Flag-gated (`REVIEWS_VIDEO`); no-op when off. Registered in `_start_scheduler` on a short interval (e.g. 1 min). Per claimed job (small batch, e.g. ≤3/tick to bound cost):
1. Load the review; resolve the stored file `DATA_DIR/review-media/<slug>/<video_ref>` (skip + mark failed if missing or `video_kind != 'upload'`).
2. Transcribe via `journal_blueprint._whisper_transcribe(path)` → transcript text.
3. `score_video(_cl, product, transcript, strip=_strip_dash)` — the Anthropic haiku client, same as the written `score_review` (Whisper/OpenAI is used only for the transcription in step 2).
4. On `compliance_ok`: `total = min(5, ai_score + video_points)`; credit `delta = max(0, total - ai_score)` dollars under `order_ref=f"review:{rid}:video"`, `reason=f"review:{slug}"` (idempotent via `points.has_entry`); `set_points(rid, total)`. On gate-fail: `video_points=0`, no credit.
5. `set_video_result(rid, video_points, transcript, 'scored')` (or `'failed'` on exception); `review_video_jobs.mark(rid, 'done'|'failed')`. Wrap each job so one failure never aborts the sweep; all Whisper/AI calls stay in the worker (never a web request).

### Recording UI (front-end, `static/begin-product.html` review form)
`MediaRecorder`-based recorder: Record / Stop / Preview / Re-record, producing a `.webm` (or `.mp4` on Safari) blob posted to `/api/reviews` as the `video` file part. The allowed length comes from a new `GET /api/reviews/limits?slug=&email=` (or is embedded in page-data) returning `{max_seconds}` from a server-side `_allowed_video_seconds(email) -> 90 or 300` (300 when `has_successful_video`). The recorder auto-stops at `max_seconds`. Upload-path videos remain bounded by the 2a-1 100 MB size cap. NO emoji.

### Length helper (server-side)
`_allowed_video_seconds(email) -> int`: `300 if product_reviews.has_successful_video(cx, email) else 90`. (Future: `or _is_paid_member(email)`.) Used by the limits endpoint and as a soft guard.

---

## Data flow
1. Buyer records (≤ their allowed length) or uploads a video on the review form → `POST /api/reviews` (verified buyer, 2a-1 path) stores the file, scores any written text synchronously, and (if `REVIEWS_VIDEO`) enqueues a video job; response says scoring is pending.
2. The scheduler drains the job: Whisper transcript → `score_video` → credit the video delta (idempotent) → store transcript/score → mark done.
3. The console `/api/console/reviews` row now shows the transcript + video score for moderation; the page total/points update.

## Error handling
- Missing file / non-upload kind → mark the job failed, `video_status='failed'`, no points.
- Whisper or AI failure → job failed, no points, logged; never crashes the sweep.
- Points credit wrapped (best-effort) — a ledger error never blocks job completion; idempotent `order_ref` prevents double-credit on re-run.
- Re-submitting a review re-enqueues (status→pending) and re-scores, but the `:video` idempotency key means points are not re-credited (consistent with 2a-1's edit-after-credit behavior; the displayed total recomputes deterministically from `ai_score + video_points`).
- Flag off → no recorder, no enqueue, worker no-ops.

## Testing
- **Schema/data:** `set_video_result`, `has_successful_video` (true only with video_points>0 AND approved); additive columns present.
- **Queue:** enqueue/claim/mark; idempotent enqueue.
- **`score_video` (fake client):** quality→points(0-5); disease-claim transcript gate-fails (video_points 0); dashes stripped; never raises.
- **Worker (mock `_whisper_transcribe` + mock scorer):** scored path credits `min(5, written+video)` delta idempotently (no double-credit on re-run), gate-fail credits nothing, missing file → failed; flag-off → no-op.
- **Enqueue on submit:** an upload-video review enqueues a job + `video_status:"pending"`; a link video does not; written-only does not.
- **Length helper:** 90 default, 300 after a successful video review.
- Follow deploy-chat test isolation (tmp `$DATA_DIR/chat_log.db`; mock Supabase; importorskip playwright; `importlib.reload` + idempotent registration). Recorder UI = manual visual pass. NO emoji; no em dashes in generated text.

## Flags
`REVIEWS_ENABLED` (existing) + new **`REVIEWS_VIDEO`** (default off; requires REVIEWS_ENABLED). With `REVIEWS_VIDEO` off, 2a-2 is fully inert (no recorder, no enqueue, worker no-op) — text reviews behave exactly as 2a-1.

## Notes
- Reuses `journal_blueprint._whisper_transcribe` (Whisper-1, accepts the video file directly), the OpenAI client, the points ledger, the 2a-1 review schema/upload/serve, and the APScheduler worker pattern. No new pip/external dependency for 2a-2 (the bundled-ffmpeg dep enters in 2a-2b for trim).
- Whisper cost is ~$0.006/min; the 90s default + ≤3-jobs/tick worker bound cost.
