# Fireside Reaction-Library — Verification Record (Task 9)

How the reaction-library + director (Track A) is verified. There is no committed
browser harness in this repo, and automated headless Chrome cannot reliably
*decode* video (readyState stays 0), so DOM/pixel playback is **not** asserted by
machine — the state-machine *logic* is verified deterministically in Node, the
served wiring is verified by an HTTP smoke check, and the *visual* quality is
reviewed by Glen against the real clips (Track B).

## 1. Pure-logic unit tests (Node, zero deps)

Run: `node --test tests/fireside_js/*.test.mjs`
(Note: pass the explicit `*.test.mjs` glob — a bare directory arg mis-discovers `.mjs`.)

Covers: manifest normalizer + graceful degradation, typing→reaction heuristics
(gaze/family/intensity), clip selection + family-affinity fallback, rate/interjection/
interruption governance + ambience-gap math, spark particle spawn. **28 tests green**
(19 module tests + 9 Director integration).

## 2. Director integration — the full dance (Node fake-DOM)

`tests/fireside_js/director-integration.test.mjs` drives the real `Director`
against a fake video/DOM, asserting no-throws + correct video-state transitions:

- **Graceful degradation:** with an empty manifest (`reactions:[]`), every Director
  method (`toResting/onType/onSubmit/maybeInterject/maybeInterruption/onReplyReady`) is
  a no-op that never throws and never sets a video src. However, the Director IS active
  on prod even with `reactions:[]` because `normalizeManifest` falls `resting_loops`
  back to `pondering_loops` (non-empty in prod), so the Director is mounted and is the
  sole video controller once initialized — the inline `pondering()`/`speakingLoop()`
  helpers no-op when `director` is present. True full inertness only occurs if module
  init never runs (manifest fetch fails / retry exhausted), in which case the inline
  helpers retain control and the page behaves as before the Director was added.
- **Dance:** `toResting` → looping resting clip; `onType` → crossfade to a reaction
  clip, then `onended` returns to resting; rate-gating suppresses a 2nd reaction
  within 5s; `onSubmit` → hero clip, then pondering on its end; `onReplyReady` →
  speaking loop.
- **Once-per-client interruption:** `maybeInterruption` fires once (sets
  `_interruptionSeen`, plays the off-screen-voice audio) then is suppressed forever;
  the gate rejects turn 1.
- **Missing-clip safety:** reactions present but no hero → `onSubmit` falls through
  to pondering without throwing.

## 3. Served-wiring HTTP smoke check (app under doppler)

Boot: `doppler run -p remedy-match -c prd -- python3 /tmp/fireside_rv_server.py`
(sets `FIRESIDE_ENABLED=true`, overrides `DATA_DIR`), then curl. Verified
2026-06-30:

- `GET /begin/fireside` → **200**, markup contains `stageA`+`stageB` (stacked
  crossfade), `#sparks` canvas, `#muteBtn`, a `type="module"` script, and imports of
  `manifest-normalize.js` / `director.js` / `ambience.js`.
- `GET /static/fireside/fireside-manifest.json` → has `resting_loops`, `reactions`,
  `ambience:{bed:null,bed_volume:0.18,oneshots:[]}` (safe-empty → prod unchanged).
- `GET /static/fireside/director.js` → **200**, `content-type: text/javascript`
  (browser ES-module loading works).

## 4. Browser-only items — Glen's manual review with REAL clips (Track B)

Machine checks can't judge these; confirm them when the real assets land
(`FIRESIDE_ENABLED` is already on in prod, so clips go live the moment the manifest
references them):

- [ ] Crossfade is visually smooth (~250ms opacity dissolve, no black flash) on the
      immersive page; both `<video>` stay `position:absolute` / same stacking context.
- [ ] Resting-loop seam hidden by the dissolve (else trim resting clips to matched ends).
- [ ] Ambience bed + one-shots audible-but-low; one-shots duck under his voice; mute
      button silences everything and toggles 🔊/🔇.
- [ ] Spark ember-burst lands over the hearth (sparkXY ≈ 22%/62%) and tracks resize.
- [ ] Full conversation loop has **zero console errors**; no journey-shell ribbon
      (page is excluded in `shell_nav.py`).
- [ ] Once-per-client interruption fires at a natural pause a few turns in, then never
      again across reloads (localStorage `fireside_interruption_seen`).
