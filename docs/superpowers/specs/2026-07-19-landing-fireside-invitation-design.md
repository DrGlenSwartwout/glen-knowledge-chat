# Landing-page fireside invitation + spoken replies

**Date:** 2026-07-19
**Status:** approved, ready for implementation plan
**Related:** `2026-06-29-glendalf-fireside-conversation-design.md`, `2026-06-30-glendalf-reaction-library-and-director-design.md`

## Problem

The landing page bot (`/`, `/begin`) already accepts voice input and can speak its
replies, but neither is discoverable. The mic is a small glyph in the input row, and
spoken replies require clicking "🔊 Listen" on every message, so most visitors never
hear Dr. Glen's voice at all.

Separately, `/begin/fireside` is a fully built video-avatar conversation that the
landing page never points at. It is live in production but effectively unreachable.

## What already exists (verified live 2026-07-19)

Nothing in this spec requires new video, new audio assets, or new server endpoints.

| Capability | Where | State |
|---|---|---|
| Mic → Whisper | `static/embed.html:1402-1497` → `POST /transcribe` (`app.py:37082`) | live, button visible on `/embed` |
| Reply → ElevenLabs | `static/tts-output.js` → `POST /chat/tts` (`app.py:29418`) | live, manual click per reply |
| Fireside conversation | `static/begin-fireside.html`, `static/fireside/director.js` | live, `/begin/fireside` returns 200 |
| Welcome clips | `fireside-manifest.json` → `intro_welcomes` (3), `intro_poster` | present, served 200 |
| Invitation reaction clip | `fireside-manifest.json` → `reactions[id=invitation]`, open-palm, silent | present |

`/chat/tts` already falls back to browser `speechSynthesis` on any non-200, and
`embed.html` already hides the mic when `MediaRecorder`/`getUserMedia` are absent.
Both degradation paths are inherited, not rebuilt.

## Design

### 1. Welcome tile

A new element in `static/begin.html`, directly above the `#begin-chat` iframe so it
reads as part of the same conversation surface.

- Poster: `intro-poster.jpg`. Clip: one of `intro_welcomes`, chosen at load.
- Autoplays **muted**, looping, with a visible "tap to hear" affordance.
- Deliberately small — roughly a 160px rounded tile, not a hero video. It must not
  push the chat input below the fold on a phone.
- On tap: unmute, restart, play the invitation with audio. On end, settle into a
  resting loop and reveal two buttons: **"Come sit by the fire"** and
  **"Ask here instead"**.
- If the visitor never taps, the page behaves exactly as it does today.

### 2. Audio unlock

Browsers block sound without a user gesture but permit muted autoplay. The tap on the
welcome tile is that gesture, and it is reused to unlock spoken replies for the rest of
the session.

- The tap sets a session-scoped flag in `begin.html` and `postMessage`s it into the
  `#begin-chat` iframe (same origin, so no cross-origin handling is needed).
- `embed.html` listens for that message and records that audio is unlocked.
- Once unlocked, each completed reply calls `window.TTS.attachAndSpeak` instead of
  `window.TTS.attach`.
- The "🔊 Listen" button remains rendered in both states. Unlocking adds automatic
  playback; it never removes the manual control.

### 3. Handoff to fireside

"Come sit by the fire" is a plain navigation to `/begin/fireside`.

Version one deliberately does **not** carry the landing-page conversation across. The
fireside agent has its own session store (`dashboard/fireside_store.py`) and its own
opening beat, and threading prior context in means reconciling two different system
prompts. That is a separate project, to be specced once this door shows it converts.

### 4. Fullscreen on fireside

`begin-fireside.html` currently approximates immersion with `position:fixed`,
`height:100dvh`, and a `visualViewport` resize handler. It contains no calls to the
Fullscreen API.

- Add a button calling `requestFullscreen` on the stage container, with the
  `webkitRequestFullscreen` prefix for Safari.
- Listen for `fullscreenchange` to swap the enter/exit icon.
- iOS Safari does not support fullscreen on arbitrary elements. Feature-detect and hide
  the button there; the existing `100dvh` + visualViewport treatment remains the
  fallback and already handles that case well.

## Out of scope

- Conversation continuity between the landing bot and fireside (follow-on project).
- Any change to `/chat`, `/chat/tts`, `/transcribe`, or `/begin/fireside/agent`.
- Live generated avatar video (HeyGen or similar) on any funnel surface.
- Replacing fireside's `webkitSpeechRecognition` input with the Whisper path used by
  the chat widget. The two differ today; unifying them is not required here.

## Testing and verification

This change is frontend-only. No server route is modified, so `tests/test_chat_tts.py`
continues to cover the TTS endpoint unchanged.

Verification is therefore behavioral:

1. Headless render of `/begin` confirming the tile appears, autoplays muted, and does
   not push the chat input below the fold at 390x844.
2. Headless render confirming the tap reveals both CTAs.
3. Manual pass on desktop Chrome: tap unlocks audio, the next reply speaks without a
   click, the "🔊 Listen" button still works.
4. Manual pass on iOS Safari: muted autoplay works, the tap unlocks audio, and the
   fullscreen button is absent rather than present-and-broken.
5. Manual pass with audio never unlocked, confirming today's behavior is unchanged.

Steps 3 and 4 are manual because autoplay policy differs per browser and per
engagement history, which is precisely what a unit test cannot reproduce.

## Risks

- **Attention competition.** The tile sits next to the chat panel's opening prompts.
  Keeping it small is the mitigation; if it measurably suppresses typed questions, it
  moves or shrinks.
- **Autoplay policy drift.** Muted autoplay is widely permitted but not guaranteed. If
  it is blocked, the tile shows the poster frame and the tap still works, so the
  failure mode is a still image rather than a broken element.
- **Unexpected speech.** A visitor who taps for the invitation then gets every
  subsequent reply spoken aloud. This is the approved behavior — silence after an
  explicit request for voice reads as broken — but the manual Listen control and normal
  browser tab-mute remain available.
