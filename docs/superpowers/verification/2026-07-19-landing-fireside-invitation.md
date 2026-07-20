# Landing fireside invitation — Verification Record

**Date:** 2026-07-19
**Branch:** `sess/b9535446`
**Spec:** `docs/superpowers/specs/2026-07-19-landing-fireside-invitation-design.md`
**Plan:** `docs/superpowers/plans/2026-07-19-landing-fireside-invitation.md`

Follows the precedent of `tests/fireside_render_verify.md`: state what a machine
asserted, what a human still must, and never blur the two.

## 1. Automated suites

```
node --test tests/begin_js/*.test.mjs
  → 14 pass, 0 fail

doppler run --project remedy-match --config dev -- python3 -m pytest \
  tests/test_begin_invitation_wiring.py tests/test_fireside_routes.py tests/test_chat_tts.py -q
  → 27 passed, 1 failed
```

The single failure is **`tests/test_fireside_routes.py::test_manifest_served`**, and it is
**pre-existing and unrelated**. It asserts `manifest["intro_video"].endswith("intro.mp4")`
while the manifest actually holds `/static/fireside/video/intro-read.mp4`. Confirmed
independently two ways: this branch never touches the manifest or that test
(`git diff --stat $(git merge-base main HEAD) HEAD -- static/fireside/fireside-manifest.json
tests/test_fireside_routes.py` is empty), and the test fails identically when checked out
at the merge-base `8487f056` in a clean worktree. Not introduced here; worth a separate fix.

The `--project remedy-match` flag is required in this worktree; bare `--config dev` fails
with "must specify a project" because no `.doppler.yaml` is version-controlled. The `prd`
config was deliberately not used — it holds live payment credentials.

## 2. Runtime verification in a real browser

Three separate task reviews flagged that the wiring tests assert only text presence in
served HTML and could not catch a logically broken listener. That gap is closed here for
the landing page.

**Method.** Static files were served directly (`python3 -m http.server` at the repo root)
rather than booting the Flask app, because importing `app` starts its cron scheduler
against dev credentials. Real Chrome was then driven against
`http://localhost:8791/static/begin.html`, with instrumentation on `Audio.prototype.play`,
on the chat iframe's `postMessage`, and on every `a[href="/begin/fireside"]` click, using
genuine user-gesture clicks rather than synthetic events.

| Behavior | Result |
|---|---|
| Button mounts from a real manifest fetch | ✅ present, `hidden` removed |
| Rendered and sized as a tap target | ✅ visible, exactly 44×44 |
| Not nested inside the anchor | ✅ `anchor.contains(button) === false` |
| Positioned against the new wrapper | ✅ `offsetParent === .avatar-wrap`, overlaps avatar top-right |
| Exactly one fireside door on the page | ✅ `a[href="/begin/fireside"].length === 1` |
| Click does **not** navigate | ✅ 3 clicks, 0 navigation attempts, path unchanged |
| Plays the right voice-over | ✅ `…/static/fireside/audio/intro-welcome-3.mp3` |
| Posts the unlock with the exact payload | ✅ `{type:"begin:audio-unlocked"}` to `http://localhost:8791` |
| Unlock is idempotent across replays | ✅ 2 play calls, 1 unlock post |
| Icon and aria-label swap on play | ✅ `🔈`/"Hear Dr. Glen's invitation" → `⏹`/"Stop the invitation" |
| Toggle stops and replays | ✅ |

## 3. NOT verified — human passes still required

Do not read section 2 as covering these.

- **Fullscreen on `/begin/fireside`.** Could not be exercised. Fullscreen is blocked in
  this automation context: an independent control button calling
  `document.getElementById('fireside').requestFullscreen()` inside a genuine click also
  failed, with the promise neither resolving nor rejecting. That rules the environment
  out, not the code. Static checks did confirm `#fsBtn` renders, is feature-detected as
  supported on this desktop, targets `MAIN#fireside`, and sits 8px clear of `#muteBtn`.
  **A human must click it and press Escape in a normal browser.**
- **Audio actually being audible.** `play()` was called with the right file; whether sound
  reached a speaker was not observed.
- **The receiving end of the unlock.** Only the *posting* side ran, because the static
  server does not serve `/embed`. That a reply then speaks itself is still covered only by
  text-presence assertions. **A human must confirm end-to-end on a real reply.**
- **iOS Safari.** Nothing was run there. The `#fsBtn` hiding path and muted-video behavior
  are unexercised.
- **The never-tapped control case.** Confirming an untouched page behaves exactly as before
  was not run end-to-end against the live app.

## 4. Incidental findings

- **`#fsBtn` sits beneath `#begin-overlay`** (z-index 2 vs 5) during the "Sit by the fire"
  entry gate, so it is visible but unclickable until `overlay.remove()` runs at
  `static/begin-fireside.html:421`. **Not a defect:** the pre-existing `#muteBtn` has the
  identical z-index and behaves the same way. Noted so nobody rediscovers it as a bug.
- **The fullscreen handler swallows promise rejections.** `fsEnter.call(fsTarget)` sits in
  a `try/catch`, but `requestFullscreen` rejects asynchronously, so a denial is silently
  discarded rather than caught. Harmless today (the intent is to stay windowed), but it
  means a real failure leaves no trace — which is exactly what made diagnosing section 3
  slow. Consider a `.catch()` that logs.
- **`document.querySelector('[data-fi-sub]')` at `static/begin.html:988`** is dead, orphaned
  by `ec233b56`. Null-guarded and harmless; left alone as out of scope.
