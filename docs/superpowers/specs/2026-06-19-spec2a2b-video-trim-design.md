# Spec 2a-2b — Auto-trim review videos

**Date:** 2026-06-19
**Status:** Approved (design); ready for implementation plan
**Repo:** deploy-chat (Flask, Render `glen-knowledge-chat`)
**Parent:** Spec 2a (reviews). Builds on 2a-2 (merged, PR #180). Next: 2a-3 (gift), 2b (referral coupons).

---

## Problem

2a-2 records, transcribes, and scores review videos but plays them untrimmed — with dead air / false starts at the head and tail. 2a-2b auto-trims each scored video to its first/last detected speech using the word timestamps Whisper already returns, producing a clean clip for display. It adds a bundled-ffmpeg dependency, gated behind its own flag.

## Scope (2a-2b)

A trim step in the existing `_drain_review_videos` worker that, after a video is scored and points are credited, computes a trim window from the Whisper word timestamps, re-encodes a frame-accurate trimmed copy via a bundled ffmpeg binary, and points the review's served video at the trimmed file (keeping the original). Gated behind a new `REVIEWS_VIDEO_TRIM` flag (default off, requires `REVIEWS_VIDEO`).

**Out of scope:** changing transcription/scoring/points (2a-2); AI gift (2a-3); referral coupons (2b). Trim never affects the points already credited.

---

## Confirmed decisions (Glen, 2026-06-19)

- **Re-encode** (frame-accurate; clips are short, CPU is fine) rather than stream-copy (`-c copy`, keyframe-bound/imprecise).
- **Padding:** LEAD **0.3s** before the first word, TRAIL **0.7s** after the last word (breathing room).
- **Separate flag `REVIEWS_VIDEO_TRIM`** (default off, requires `REVIEWS_VIDEO`) — an independent kill-switch for the new ffmpeg dependency + CPU, so trim can be toggled without disabling video scoring.
- **Non-destructive:** keep the original file; write the trimmed copy; point `video_ref` at the trimmed file.

---

## Architecture

### Dependency
Add `imageio-ffmpeg` to `requirements.txt`. It bundles a static ffmpeg binary (incl. libx264) reached via `imageio_ffmpeg.get_ffmpeg_exe()` — no system install, no Render dashboard change. Imported **lazily** (inside the trim helper) so the module import never fails in environments without it (tests, trim-off).

### `dashboard/video_trim.py`
- `compute_trim_window(words, duration, *, lead=0.3, trail=0.7, min_removed=1.0) -> (start, end) | None`
  - `words` = Whisper `words[]` (each `{"word","start","end"}`); `duration` = clip length (seconds).
  - `start = max(0, words[0]["start"] - lead)`; `end = min(duration, words[-1]["end"] + trail)`.
  - Returns `None` (skip trim) when: `words` is empty, `end <= start`, or the total removed `(start) + (duration - end) < min_removed` (not worth re-encoding).
- `_ffmpeg_exe() -> str` — lazily `import imageio_ffmpeg; return imageio_ffmpeg.get_ffmpeg_exe()` (tests monkeypatch this).
- `trim_video(src_path, dst_path, start, end, *, runner=None) -> bool` — builds the ffmpeg command `[exe, -y, -ss, str(start), -to, str(end), -i, src, -c:v, libx264, -c:a, aac, -movflags, +faststart, dst]`, runs it via `runner` (default `subprocess.run`, capture, timeout ~300s), returns `True` when the process exits 0 and `dst_path` exists, else `False`. Never raises (wraps exceptions → `False`).

### Schema (extend `product_reviews`, additive)
Add `video_orig_ref TEXT DEFAULT ''`. New function `product_reviews.set_trimmed(cx, review_id, trimmed_ref)`: if `video_orig_ref` is empty, set it to the current `video_ref` (preserve the original), then set `video_ref = trimmed_ref`. Idempotent re-trim keeps the same `video_orig_ref`.

### Worker extension (`_drain_review_videos`)
1. Keep the **full** Whisper result, not just `.text`: change `transcript = (...).get("text","")` to capture `whisper = _jb._whisper_transcribe(str(path)) or {}`; `transcript = whisper.get("text","")`; `words = whisper.get("words", []) or []`; `duration = whisper.get("duration") or 0`.
2. Score + credit exactly as today (unchanged — trim runs after, so a trim failure never costs the buyer points).
3. If `_REVIEWS_VIDEO_TRIM`: in a wrapped try/except, `win = video_trim.compute_trim_window(words, duration)`; if `win` is not None, build `dst = <src_stem>-trim.mp4` under the same `_REVIEW_MEDIA_DIR/<slug>/`, and on `video_trim.trim_video(src, dst, *win)` success call `product_reviews.set_trimmed(cx, rid, dst.name)`. Any failure logs and leaves `video_ref` (and points) untouched. The serve route `review_media` already serves any sanitized filename in the dir.

### Flag
`_REVIEWS_VIDEO_TRIM = os.environ.get("REVIEWS_VIDEO_TRIM","") in (...)`. Trim is fully inert when off (no ffmpeg import, no re-encode). Requires `REVIEWS_VIDEO` on (trim only runs inside the video worker, which is itself `REVIEWS_VIDEO`-gated).

---

## Data flow
1. The video worker transcribes (now keeping `words` + `duration`), scores, and credits points (2a-2, unchanged).
2. If `REVIEWS_VIDEO_TRIM`: compute the speech window → re-encode a `<name>-trim.mp4` → repoint `video_ref` to it (original preserved in `video_orig_ref`).
3. The page/console serve the trimmed clip via the existing `/review-media/<slug>/<file>` route.

## Error handling
- No words / negligible trim (< `min_removed`) → `compute_trim_window` returns `None` → keep the original, no re-encode.
- ffmpeg failure / exception / missing binary → `trim_video` returns `False` → `video_ref` unchanged, points unaffected, logged; the sweep continues.
- Trim wrapped separately from scoring so it can never undo or block a credited score.
- All ffmpeg runs in the background worker (never a web request). Re-running a done job is bounded by the queue (jobs marked done aren't re-claimed); `set_trimmed` is idempotent on `video_orig_ref`.

## Testing
- **`compute_trim_window`:** trims to first/last word ± padding clamped to the clip; returns `None` for empty words, for `end<=start`, and when removed `< min_removed`.
- **`trim_video`:** builds the expected ffmpeg arg list (assert via a fake `runner` + monkeypatched `_ffmpeg_exe`); returns `True` only when the runner exits 0 AND `dst` exists (have the fake runner create `dst`); returns `False` on non-zero exit / exception (never raises). No real ffmpeg needed.
- **`set_trimmed`:** sets `video_orig_ref` to the prior `video_ref` once, repoints `video_ref`; second call keeps the original `video_orig_ref`.
- **Worker (mock `_whisper_transcribe` returning words+duration, mock `score_video`, monkeypatch `video_trim.trim_video`):** with the flag on and a non-None window + trim success → `video_ref` becomes the `-trim.mp4` and `video_orig_ref` holds the original; trim failure → `video_ref` unchanged + points intact; flag off → no trim attempted; no-words → no trim.
- Follow deploy-chat test isolation (tmp `$DATA_DIR`; mock Supabase; importorskip playwright; `importlib.reload`). NO emoji; no em dashes in any generated text.

## Flags
`REVIEWS_ENABLED` + `REVIEWS_VIDEO` + new **`REVIEWS_VIDEO_TRIM`** (default off). With trim off, 2a-2 behavior is unchanged (videos play untrimmed).

## Notes
- `imageio-ffmpeg`'s bundled binary includes libx264 + aac, so the re-encode needs no system codecs. First worker run downloads/locates the binary; subsequent runs reuse it.
- Output is `.mp4` (h264/aac, `+faststart`) for universal playback regardless of the source container (webm/mp4). The serve route accepts any sanitized filename.
- Re-encode CPU is bounded by the ≤3-jobs/tick worker cadence and the ≤5-min clip cap; trim is the only CPU-heavy step, hence its own flag.
