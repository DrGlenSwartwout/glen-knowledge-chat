# Glendalf Reaction Library & Real-Time Director — Design

**Date:** 2026-06-30
**Status:** Design (brainstormed) — ready for implementation plan
**Builds on:** [`2026-06-29-glendalf-fireside-conversation-design.md`](./2026-06-29-glendalf-fireside-conversation-design.md) (shipped as PR #430, dark behind `FIRESIDE_ENABLED`)
**Related memory:** `project_guided_adventure_glendalf`, `reference_audio_production_workflow`

---

## 1. Summary

The shipped fireside plays a single listening loop, a single speaking loop, one pondering loop, plus pre-cached audio fillers/interjections in Glen's cloned voice. This project gives Glendalf a **palette of expression** and a **real-time director** that selects from it: as the traveler types (or, later, speaks), Glendalf reacts the way a present, attentive healer would — a nod, a curious eyebrow, a soft "ahh…", a knowing recognition, a delighted laugh, a glance up-and-to-his-right as he pictures what they're describing.

The reaction is produced **in video** (Kling/HeyGen renders), not a rigged real-time puppet. So the system is a **flat library of self-contained, tagged clips** and a **director** that picks the single best-matching clip and crossfades to it from a resting loop, then back. Gaze and hand-gesture are *selection tags*, not runtime-composited layers.

**Scope of THIS spec = reactions only** — Glendalf's responses *while the traveler communicates* (the listening state, the pondering gap, and break-in interjections). The emotion-matched **speaking** library (his reply delivery) is **Phase B**, deferred to a later spec (§11).

---

## 2. Goals / Non-Goals

**Goals**
- A curated, named, metadata-tagged **clip library** that drops into the existing `fireside-manifest.json` with **no app.py route change**.
- A client-side **director** that turns the traveler's typing into near-real-time reactions, cheaply.
- Reuse the fireside's **already-shipped Haiku per-turn analysis** for the one big "hero" reaction per turn — no new model cost.
- Encode Glen's **NLP eye-accessing** model so gaze is responsive to the representational mode in play.
- Stay entirely behind `FIRESIDE_ENABLED`; ship dark; degrade gracefully to today's single-loop behavior if any asset is missing.

**Non-Goals (this spec)**
- Speaking-state emotion matching (Phase B).
- Voice input / STT (forward section §10; the tap-to-talk slice is separately specced).
- True runtime layer compositing / a Live2D-style real-time puppet (§11, "someday").
- Rendering the full Cartesian product of all channels — we curate high-value combinations.

---

## 3. The Reaction Palette

Four channels. The director composes a reaction by choosing a clip whose tags best match the moment.

### 3.1 Channel 1 — Emotional families (13)

Each family has a **silent** form (expression + head/body only) and a **vocalized** form (a non-verbal backchannel in Glen's cloned voice). Expressions **may open the mouth** where the emotion demands it (laugh, surprise) — they are *not* constrained to mouth-closed.

| # | Family | Silent form | Vocalized form |
|---|---|---|---|
| 1 | **Attending** (continuer) | slow nod, attentive stillness, slight lean-in | "Mm-hmm…", "Mm." |
| 2 | **Affirming** | warmer nod + slight smile | "Yes…", "That's it.", "Good." |
| 3 | **Empathic concern** | brow softens, head tilt, eyes lower | low "Ahh…", soft sigh |
| 4 | **Curiosity / intrigue** | eyebrow raise, head tilt, eyes brighten | "Oh?", "Hmm?" |
| 5 | **Surprise** (mouth opens) | eyebrows up, eyes widen, slight recoil | "Oh!", "Ah!", small gasp |
| 6 | **Delight / shared joy** (mouth opens) | broad smile, eyes crinkle | warm chuckle, gentle laugh |
| 7 | **Recognition / insight** | slow knowing nod, eyes narrow→open, finger lifts | "Ahhh…", "Mm, yes." |
| 8 | **Pondering** | strokes beard, gazes into fire, reaches for book | "Hmm…", "Let me consider…" |
| 9 | **Reassurance / grounding** | settling breath, steady gaze, open palm | "It's alright.", soft exhale |
| 10 | **Gentle gravity** (severity, no alarm) | stillness, slight head shake, measured gaze | low "Mmm.", "I understand." |
| 11 | **Awe / wonder** | eyes lift, soft open expression, stilled | quiet "Ohh…", breath of wonder |
| 12 | **Invitation-to-go-deeper** | both palms up, slow beckon, lean-in | "Tell me more…", "Say more of that…" |
| 13 | **Lightness** (playful / teasing / wry) | half-smile, raised brow, a wry tilt | soft "Heh.", light "Ah, you…" |

### 3.2 Channel 2 — Hand actions

A separate channel of silent communication and realism. **Silent gestures** carry no sound; **audible actions** carry a real in-room sound and are their **own category** — the director fires an audible action *instead of* a vocalization, never both (keeps the soundscape clean).

**Silent gestures** (no sound):
| Gesture | Reads as | Pairs with |
|---|---|---|
| Steepled fingers | contemplation | Pondering |
| Stroke beard / cup chin | weighing it | Pondering |
| Open palm extended | offering, safety | Reassurance |
| Hand to heart/chest | sincerity | Empathic concern |
| Both palms up, slight lift | "tell me…" | Invitation |
| Slow beckon | "say more" | Invitation |
| Single finger lifts | "ah — a point" | Recognition |
| **Pointing** (at traveler / book / upward) | direct address, emphasis | Recognition, Affirming |
| Palm-down gentle press | "settle, it's alright" | Gentle gravity / Reassurance |
| Hand rests/pats the book | grounding punctuation | Attending |
| Lifts & examines a vial | curiosity about *them* | Curiosity / Awe |
| Smooths robe / settles | idle realism | (idle filler) |

**Audible actions** (gesture + sound; own category, no vocal combining):
| Action | Sound | Reads as | Pairs with |
|---|---|---|---|
| Soft single clap | *clap* | quiet delight | Delight, Affirming |
| Slap the knee | *slap* | hearty laugh | Delight (high) |
| Tap/slap the table | *tap* | emphasis | Affirming, Recognition |
| Snap fingers | *snap* | "of course!" | Recognition (high) |
| Knuckle tap on table | *tok* | gentle punctuation | Attending |
| Rub palms together | *shhk* | anticipation | Invitation, Curiosity |
| Throat clear | *ahem* | pre-speech beat | (bridge → speaking) |
| Set book down | *thud* | decisiveness | Recognition, Gravity |
| Turn a page | *rustle* | pondering action | Pondering |
| Lift glass / sip | *clink* | thinking-gap beat | Pondering (idle) |

### 3.3 Channel 3 — Eye-gaze micro-layer (NLP rep-system)

Six directions from **Glendalf's own perspective** (his left/right). **Render note: his right = the viewer's left of frame** — render prompts must specify "Glendalf's right (screen left)".

| Gaze (his POV) | Rep-system | When |
|---|---|---|
| Up + **right** | Visual **constructed** | listening — picturing what they describe |
| Lateral + **right** | Auditory **constructed** | listening — composing the sound of it |
| Down + **right** | **Kinesthetic** | listening — feeling into it |
| Up + **left** | Visual **remembered** | pondering — recalling an image he's seen |
| Lateral + **left** | Auditory **remembered** | pondering — recalling words/sounds |
| Down + **left** | Internal dialogue (Ad) | pondering — talking it through to himself |

**Key rule (Glen's refinement):** while **listening**, Glendalf is building a picture of *the traveler's* world → he favors the **right (construct) triad**. While **pondering**, he draws on *his own* memory/knowledge → the **left (remembered) triad**. So *listening = right, pondering = left.*

Staging: **separable standalone glance clips first (A), gaze baked into hero reactions later (B).**

### 3.4 Channel 4 — Interjections (cross-cutting)

Short break-in clips played *while the traveler is still typing*, rate-capped (the single biggest "he's alive" lever; magic when rare, maddening when overused). Already shipped as audio in the manifest; this project adds matching short **video** beats. v1 set: "Ah—", "Yes, go on…", "Mm—", "Oh…", "Now that's the heart of it—".

---

## 4. Production Model — Flat Tagged Clip Library + Crossfade

**Decision: Model A.** Because Glendalf is rendered video, we **cannot** composite a gaze-glance or a hand-gesture as a live layer over a base loop without it looking wrong. Therefore:

- Every reaction is a **self-contained clip** (a complete render).
- Each clip carries **metadata tags** (family, form, gaze, hand-action, intensity, duration, loopable).
- The director **selects** the single best-matching clip and **crossfades** from the resting loop into it, then crossfades back.
- Gaze and gesture are **tags that aid selection**, not runtime-composited layers.
- The library is a **curated set of high-value combinations**, not the full Cartesian product (which would be thousands of clips).

Consequence: a "separable gaze layer" is realized through **clip selection + crossfade**, not video compositing.

---

## 5. Clip Metadata Schema (manifest extension)

Extends the existing `static/fireside/fireside-manifest.json` (which already has `intro_video`, `speaking_loop`, `pondering_loops`, `fillers`, `interjections`). Add a `reactions` array and a `resting_loops` array. **No app.py change** — the page already fetches the manifest.

```jsonc
{
  // ...existing fields unchanged...
  "resting_loops": [
    "/static/fireside/video/listen-rest-1.mp4",
    "/static/fireside/video/listen-rest-2.mp4"      // 2-3 variants to avoid robotic repetition
  ],
  "reactions": [
    {
      "id": "concern-silent",
      "family": "empathic_concern",     // one of the 13
      "form": "silent",                 // "silent" | "voice" | "audible_action"
      "gaze": null,                     // null | "up_right" | "lat_right" | "down_right" | "up_left" | "lat_left" | "down_left"
      "hand": null,                     // null | gesture id (e.g. "hand_to_heart", "snap", "table_tap", "point")
      "intensity": "med",               // "low" | "med" | "high"
      "tier": "backchannel",            // "backchannel" (small, while typing) | "hero" (big, on submit) | "ponder" | "gaze" | "interjection"
      "duration_s": 2.4,
      "loopable": false,
      "file": "/static/fireside/video/react-concern-silent.mp4",
      "audio": null                     // optional separate SFX/voice track if not baked into the clip
    }
  ]
}
```

Selection keys the director uses: `tier` (which moment), `family` (emotional match), `form` (silent/voice/audible), `gaze`, `intensity`. Unknown/extra fields are ignored — forward-compatible. **Graceful degradation:** if `reactions`/`resting_loops` are absent or a file 404s, the page falls back to today's single `speaking_loop`/`pondering_loops` behavior.

---

## 6. The Director (client-side, hybrid)

**Decision: Model A — hybrid.** Lives in `static/fireside/` JS, loaded by `begin-fireside.html`. Governs three moments:

### 6.1 Resting state
The default. A `resting_loops` clip plays on loop (random variant on each return, so he doesn't repeat robotically). Everything else crossfades out of and back into this.

### 6.2 While the traveler types — client heuristics (instant, free, no server round-trip)
On a debounced sample of the **partial input text** (e.g. every ~400ms, and on punctuation):
1. **Rep-system predicate scan → gaze.** Visual predicates ("see, look, picture, clear, dark, bright, focus") → `up_right`; auditory ("hear, sounds, told, loud, quiet, said") → `lat_right`; kinesthetic ("feel, heavy, tense, pain, gut, tight, warm") → `down_right`. (Listening = right/construct triad.) Pick a `gaze` backchannel clip.
2. **Emotion keyword + punctuation/length → light backchannel.** e.g. "!" or excitement words → Curiosity/Surprise eyebrow; sorrow words → Empathic-concern silent; "?" → Attending nod. Pick a low/med-intensity `backchannel` clip.
3. **Rate-limit (anti-twitch):** at most one backchannel every N seconds (tunable, ~4–6s); never two reactions overlapping; always return to resting between.
4. **Idle pause → interjection / invitation.** On a typing pause ≥ threshold (reuse the shipped idle≥3.5s, ≤3/session, never turn 1 rule), optionally fire an interjection or an Invitation-to-go-deeper beat.

### 6.3 On submit — the hero reaction (accurate, reuses shipped Haiku)
When the traveler submits, the fireside already runs a **Haiku per-turn analysis** (ASH coverage). Extend its output (or piggyback a tiny field) with a **dominant-emotion label** mapped to one of the 13 families. During the **pondering gap** (the ~4s while the LLM + TTS render), the director fires the matching **`hero`** reaction (the big considered one — Empathic concern, Delight/laugh, Surprise, Recognition, Awe…), then settles into a pondering clip until the voice reply is ready. **No new model call** — it rides the analysis already happening.

### 6.4 Selection algorithm (per moment)
```
candidates = reactions.filter(tier == currentMoment)
if gaze wanted:   prefer candidates with matching gaze
filter family == wantedFamily (fallback: nearest family in a small affinity map)
filter form (typing→silent/voice small; never audible_action mid-typing unless intensity high)
pick by intensity match; tie-break random; crossfade in; on end → resting
if no candidate: stay in resting (never error)
```

### 6.5 Rate / taste governance (carry fireside's discipline)
- Backchannels: ≤1 per ~5s while typing.
- Interjections: idle≥3.5s, ≤3/session, never turn 1 (reuse shipped constants).
- Hero: exactly one per submitted turn, during the pondering gap only.
- Audible actions: rare, high-intensity moments only; never during the traveler's typing flow if it would talk over them.

---

## 7. v1 Curated Clip Set (~32 clips, render order)

Render in sub-waves so the loop comes alive early, then deepens:

**Wave 1.0 — make him present (~13):** 2 resting loops; silent backchannels for Attending(nod), Curiosity(eyebrow), Empathic-concern, Recognition, Surprise (5); hero reactions for Empathic-concern, Delight/laugh, Surprise, Recognition, Awe (5); 1 extra pondering loop (beard-stroke).

**Wave 1.1 — give him voice & hands (~12):** vocalized backchannels: "mm-hmm", "oh?", "ahh…", "oh!", warm chuckle, "hmm" (6); audible actions: snap, table-tap, soft clap (3); pondering beats: reach-for-book, sip, page-turn (3).

**Wave 1.2 — give him eyes (~6):** gaze glances, **right triad first** (up_right, lat_right, down_right), then left triad (up_left, lat_left, down_left).

**Wave 1.3 — interjection video beats (~5):** match the shipped interjection audio ("Ah—", "Yes go on…", "Mm—", "Oh…", "Now that's the heart of it—").

Total ≈ 36; trim to the strongest ~32 after review. Expand toward the full palette over time.

---

## 8. Integration with the shipped fireside

- **No route change.** `begin-fireside.html` already fetches `fireside-manifest.json`; the director is additive JS in `static/fireside/`.
- **Hero-emotion plumbing:** the only server touch is exposing the dominant-emotion family from the existing Haiku analysis to the client (a field on the SSE/analysis response). If that field is absent, the director simply skips the hero reaction (degrades to a pondering clip).
- **Flag:** entirely within `FIRESIDE_ENABLED`. Real clips swap in by replacing/extending manifest entries — placeholders remain valid fallbacks.
- **Asset hygiene:** all clips `-movflags +faststart`, consistent dimensions (1280×720), short (≤~3s for backchannels, ≤~5s heroes), clean loop points where `loopable`.

---

## 9. Production Pipeline

- **Source still:** the locked master `n11-N1.jpg` (Glendalf, his real likeness) is the start image for every Kling i2v render → consistent character across the whole library.
- **Render:** `kwaivgi/kling-v1.6-standard` image-to-video, per-clip prompt encoding expression + head/camera motion + (optional) gaze direction + (optional) hand gesture; `negative_prompt` to suppress talking-mouth where the family is silent. Camera moves (dolly/close-up) used for hero/emphasis variety, as validated in the proof batch.
- **Voice/SFX:** vocalized backchannels rendered in Glen's clone (ElevenLabs `jFxSqMckq2I4mET3C5QC`), trimmed; audible-action SFX either captured in-clip or synced as a separate `audio` track. Non-verbal sounds (laugh, gasp, "ahh") may need careful TTS or light sound design.
- **Post:** ffmpeg faststart + uniform encode; name by `id` from the manifest; small inline-GIF proof for review before locking.
- **Naming:** `react-<family>-<form>[-<gaze>][-<hand>].mp4`, gaze clips `gaze-<dir>.mp4`, resting `listen-rest-N.mp4`.

---

## 10. Voice input (forward — not in v1)

Typing is the shipped modality, so v1 reactions are real-time. When tap-to-talk lands:
- **Batch STT** (today's `/begin/doorway` Whisper flow): word-based gaze/content reactions lag ~1–3s (fire *after* they finish speaking). Mitigate with **acoustic** emotional backchannels driven by prosody (laughter, distress, excitement) straight from the audio — no transcription needed, near-real-time.
- **Streaming STT** (Deepgram/Whisper-streaming): ~0.3–0.8s partial transcripts → near-parity with typing, at the cost of a new dependency. Adopt only if the lag proves to hurt.

The director's selection layer is input-agnostic; only its *signal source* changes.

---

## 11. Phase B & beyond (deferred, separate specs)

- **Speaking-state emotion library:** emotion-matched, mouth-hidden speaking loops chosen by the *emotional tone of Glendalf's own reply* — turns the reply delivery expressive (the other half of the video library).
- **Gaze/gesture baked into hero reactions** (the "B" of A→B staging): signature clips with gaze + gesture pre-paired for the highest-value moments.
- **Live-composited puppet ("someday"):** rebuild Glendalf as a rigged 3D/2.5D real-time avatar for true infinite combination — a major pipeline change, only if the curated-video ceiling is ever hit.

---

## 12. Testing

- **Manifest/director unit tests:** schema parse, graceful degradation (missing `reactions` → single-loop fallback; 404 clip → resting), selection algorithm (family/gaze/intensity match + fallbacks), rate-limit governance (≤1 backchannel/5s, ≤3 interjections/session, never turn 1, one hero/turn).
- **Heuristic tests:** predicate→gaze mapping (visual/auditory/kinesthetic strings → up_right/lat_right/down_right), emotion-keyword→family.
- **Render-verify (headless):** page loads with the extended manifest, resting loop plays, a simulated typed message triggers a crossfade reaction and returns to resting, no console errors, no chrome ribbon (page stays excluded from journey shell).
- **Taste pass (human):** Glen reviews each wave as inline GIFs before locking — likeness, motion naturalness, no unintended talking-mouth on silent clips.

---

## 13. Open Questions / Decisions Locked

**Locked in brainstorming:** scope = reactions-first then whole library (B later); gaze = separable A then baked B; hand audible actions = own category, no vocal combining; production = Model A (flat tagged clips + crossfade); director = Model A (hybrid heuristics + existing Haiku); listening gaze = right/construct triad, pondering = left/remembered triad.

**Open for the plan:**
1. Exact hero-emotion field name and where it rides on the existing analysis response.
2. Crossfade mechanism in the client (two stacked `<video>` elements with opacity transition vs. a single element with a brief dissolve) — pick the one that doesn't flash on the immersive page.
3. Non-verbal vocalization production: which backchannels (laugh, gasp, "ahh") come from ElevenLabs vs. light sound design.
4. Whether `resting_loops` variants need matched first/last frames for seamless crossfade, or whether the dissolve hides the seam.
