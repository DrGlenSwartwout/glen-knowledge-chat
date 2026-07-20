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
| **Hero avatar (the existing fireside door)** | `static/begin.html:761` — `<a class="avatar" href="/begin/fireside">` wrapping an autoplaying muted looping `/static/media/glendalf-welcome.mp4`, caption "Sit by the fire with Dr. Glen →", click handler at `:1429` records engagement | live |
| Invitation voice-over | `static/fireside/audio/intro-welcome-3.mp3` (4.3s), also `intro-welcome.mp3`, `intro-welcome-2.mp3` | present, `intro_welcome_audio` in manifest |

**Correction, 2026-07-19.** An earlier draft of this spec claimed the landing page had
no video and proposed adding a standalone welcome tile. That was wrong on both counts.
The hero avatar above is already a muted autoplaying Glendalf video linking to fireside,
and commit `ec233b56` ("remove standalone fireside section — hero avatar is the single
fireside entry") deliberately consolidated the page to exactly one fireside door. A
second tile would undo that decision. The design below therefore hangs the invitation
off the existing avatar instead of adding a sibling to it.

`/static/media/glendalf-welcome.mp4` carries **no audio track** (verified with ffprobe:
one h264 video stream, nothing else). Unmuting it would play silence, so the spoken
invitation must come from a separate audio file.

`/chat/tts` already falls back to browser `speechSynthesis` on any non-200, and
`embed.html` already hides the mic when `MediaRecorder`/`getUserMedia` are absent.
Both degradation paths are inherited, not rebuilt.

## Design

### 1. Speaker button on the existing hero avatar

No new tile. A small speaker button is added as a **sibling** of the existing
`<a class="avatar">` at `static/begin.html:761`, inside a wrapping `.avatar-wrap`
element, positioned in a corner over the looping video.

- Tapping the speaker plays the invitation voice-over (`intro_welcome_audio` from
  the manifest) while the avatar video keeps looping muted underneath. The button
  swaps to a stop state while playing and back when it ends.
- The speaker must **not** navigate. Its handler calls `preventDefault()` as a
  guard against any future re-nesting; `stopPropagation()` is unnecessary because
  the button is a sibling inside `.avatar-wrap`, not a descendant of the anchor,
  so the click cannot reach the anchor's handler in the first place.
- Tapping anywhere else on the avatar still goes to `/begin/fireside` exactly as it
  does today, including the existing engagement handler at `:1429`.
- No second CTA is added. The avatar's own caption already reads
  "Sit by the fire with Dr. Glen →".
- If the manifest fetch fails or yields no audio, the button is never shown and the
  page behaves exactly as it does today.

While editing this file, also delete the orphaned CSS left behind by `ec233b56`
(`#fireside-invite`, `.fi-inner`, `.fi-title`, `.fi-sub`, `.fi-cta`, and
`@keyframes fiGlow`). It styles a section that no longer exists.

### 2. Audio unlock

Browsers block sound without a user gesture. The tap on the speaker button is that
gesture, and it is reused to unlock spoken replies for the rest of the session.

- The tap `postMessage`s an unlock into the `#begin-chat` iframe (same origin, so no
  cross-origin handling is needed).
- `embed.html` listens for that message and records that audio is unlocked.
- Once unlocked, each completed reply calls `window.TTS.attachAndSpeak` instead of
  `window.TTS.attach`.
- The "🔊 Listen" button remains rendered in both states. Unlocking adds automatic
  playback; it never removes the manual control.

### 3. Handoff to fireside

Unchanged from today: the existing hero avatar anchor navigates to `/begin/fireside`.
This design adds no new door and removes none.

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

1. Headless render of `/begin` at 390x844 confirming the speaker button appears over
   the avatar and does not obscure its caption.
2. Manual pass on desktop Chrome: tapping the speaker plays the voice-over and does
   **not** navigate; tapping elsewhere on the avatar still opens `/begin/fireside`.
3. Manual pass on desktop Chrome: after the speaker tap, the next reply speaks without
   a click, and the "🔊 Listen" button still works.
4. Manual pass on iOS Safari: the speaker plays, the tap unlocks audio, and the
   fireside fullscreen button is absent rather than present-and-broken.
5. Manual pass with the speaker never tapped, confirming today's behavior is unchanged.

Steps 2 through 4 are manual because autoplay policy and audio-unlock behavior differ per
browser and per engagement history, which is precisely what a unit test cannot reproduce.

## Risks

- **A lost unlock is silent.** The unlock is posted into the `#begin-chat` iframe, a large
  document that loads in parallel with the small manifest fetch. An early tap can land
  before that iframe registers its listener, and `postMessage` discards it with no error.
  The mitigation is to re-post on every play and again on the iframe's `load` event,
  keeping the receiver idempotent. Without that, auto-speak dies for the whole session
  with no signal anywhere.
- **Tap-target crowding.** A control overlaying a link risks mis-taps on a phone. The
  mitigation is a minimum 44x44px target placed in a corner, clear of the caption.
- **Unexpected speech.** A visitor who taps for the invitation then gets every
  subsequent reply spoken aloud. This is the approved behavior — silence after an
  explicit request for voice reads as broken — but the manual Listen control and normal
  browser tab-mute remain available.
