# Glendalf Fireside Conversation — Design Spec

**Date:** 2026-06-29
**Status:** Design approved (brainstorm). Ready for implementation plan.
**Part of:** the larger "guided adventure" re-architecture of the illtowell.com onboarding. **This is the FIRST slice** — the emotional core. Follow-on specs (separate): the Remedy Match reveal + $1 E4L trial binding; the adventure shell (map-as-stage, parchment-scroll banners, per-stage background art, video→chat→map sequencing); the emotion-matched video library; voice-in (tap-to-talk).
**Builds on:** `2026-06-27-ash-health-ally-foundation.md` (the conversational ally brain + 12-dimension ASH map), the live `/begin/doorway` voice pipeline, and the existing `/chat/tts` cloned-voice infrastructure.

---

## 1. What this is

An intimate, full-screen **conversation with Glendalf** (Dr. Glen Swartwout as the wizard) by his fire. The user **types**; Glendalf **replies in Glen's cloned voice with on-screen subtitles**. No map, no header, no chrome — a single, immersive, *defined pathway individualized by what the user says* (not branching choices). The conversation gets them feeling genuinely *heard*, surfaces an insight about what they're carrying, and **ends on a hook**: *"I think I know what your body is asking for… shall we go and find it?"* — the doorway to the (separately-specced) Remedy Match.

This re-presents the existing consultative funnel chat / ASH health ally as a voice+video fireside, reusing that same chat brain — it is a **presentation + multi-turn realization**, not a new conversational purpose.

**Decisions locked (brainstorm):** input = **type-in** (tap-to-talk is a fast-follow); output = **Glendalf in Glen's cloned voice + subtitles**; video = **B: one rendered lip-sync intro + one mouth-hidden speaking loop + one "pondering" loop** (emotion-matched library deferred); **no map/header**; conversation **ends on the hook** (no Remedy Match / no payment in this slice); identity **anonymous session** for v1.

---

## 2. The experience (user flow)

1. **Arrive** at `/begin/fireside` (full-screen, warm, dark, firelit). Glendalf sits by his fire.
2. **Intro** — a short **rendered, lip-synced** video: Glendalf greets them in Glen's voice and invites them to speak their heart (the one-time "wow"; this single clip is the only lip-synced asset).
3. **Listening** — Glendalf settles into a calm **listening loop**. A minimal text line waits at the bottom.
4. **They type** a message and send.
5. **The pondering beat** (latency cover, see §5): the instant they send, a brief **"Hmmm…"** in Glen's voice plays and Glendalf shifts to a **pondering loop** (a draw on the pipe + a magical smoke ring / a sip from the glass / reaching for a book / turning to poke the fire). His reply **streams in as subtitles immediately**; the spoken voice follows a beat later.
6. **He speaks** — the reply plays in Glen's cloned voice; subtitles complete. A small **🔊 replay** affordance is available.
7. **Back to listening**, awaiting their next line. Loop steps 4–7.
8. **The hook** — when Glendalf has heard enough to land a meaningful reflection, he closes on the hook line and the slice ends (a single forward affordance — e.g. a glowing "Yes" — that, in this slice, simply marks the end; wiring it to the Remedy Match is the next spec).

---

## 3. Architecture — reuse vs. build

### Reuse (do NOT rebuild)
- **Cloned voice-out:** `POST /chat/tts` (`app.py` ~17167) → pre-rendered MP3 in Glen's ElevenLabs clone (`ELEVENLABS_VOICE_ID` = `jFxSqMckq2I4mET3C5QC`, model `eleven_turbo_v2_5`), LRU-cached + per-IP rate-limited, with browser `speechSynthesis` fallback. Client helper `static/tts-output.js` (`TTS.attach` / `attachAndSpeak`).
- **Streaming pattern:** the SSE architecture of the existing `/chat` endpoint (`app.py` ~3197) — mirror it so Glendalf's reply text streams to the subtitles instantly.
- **Per-turn signal read (optional):** the doorway's `/journal/analyze` (Whisper → Haiku-4.5 structured analysis: elements/treasures/polyvagal/themes) — for v1, only relevant if a turn is voice; with type-in, a lightweight text analysis or skipping it is fine.
- **Session + storage:** `amg_session` cookie + `chat_log.db` sqlite.

### Build new
- **The Glendalf agent** — a multi-turn conversational LLM endpoint (the ASH ally brain, §4).
- **The fireside UI** — `static/begin-fireside.html` served at `/begin/fireside` (§5).
- **`fireside_sessions`** table (§6).
- **Video + audio assets** — the intro, the two loops, the "Hmmm…" filler (§7).

---

## 4. The Glendalf brain

The conversational brain **is** the `2026-06-27-ash-health-ally-foundation.md` ally, realized here as a real **multi-turn agent** (only the one-shot doorway exists today). New server endpoint (SSE-streamed, mirroring `/chat`):

- **Persona:** Glendalf — warm, wise, unhurried; Glen's voice and clinical lens (Wellness Whispering = the listener who translates the body's signals). System prompt extends the #403 ally persona + NLP/mirroring principles.
- **Behavior:** reads the running turn history + the latest user message; **follows the user's thread** while gently cultivating the key ASH dimensions; reflects back what it heard ("I hear you say…"); asks one good next question. It tracks a running **ASH coverage map** per session.
- **Ending:** the agent decides when it has surfaced enough to land the reflection, then emits the **hook** close. v1 may use a soft heuristic (min turns + coverage of the must-reach dimensions) plus the agent's judgment; the exact trigger is tuned in the plan.
- **Model:** a conversational model (Haiku 4.5 for cost/latency, or Opus for depth — decide in the plan; per-turn analysis can stay Haiku). Responses **streamed** for instant subtitles.

---

## 5. The presentation dance (the make-or-break)

Voice is **pre-rendered (~2–4s)**, not streamed — so the UX is engineered to never feel like dead air:

On **send**:
1. **Instant acknowledgment** — immediately play a short, pre-cached **filler phrase** in Glen's voice (zero-latency, so there's never silence at t=0). Drawn from a small **repertoire** of natural conversational fillers, e.g.: *"Hmmm…"*, *"Well…"*, *"I see."*, *"That's very interesting."*, *"Let me consider that."*, *"You've given me much to contemplate there."*, *"Here's what I think…"* — a mix of acknowledgments, thinking beats, and bridges-into-the-answer. v1 picks at random (avoiding immediate repeats); **contextual/emotion-matched selection** of the filler — like the video library — is a follow-on. These are a fixed set, pre-rendered once (recorded by Glen or generated via the clone) and cached.
2. **Pondering loop** — switch the video to a mouth-hidden pondering action (pipe + smoke ring / glass / book / poke-the-fire). "Poke the fire" can even begin under the start of speech since the mouth isn't visible.
3. **Subtitles stream immediately** — the agent's reply text arrives via SSE and renders word-by-word as it generates (fast).
4. **TTS in parallel** — the reply text is sent to `/chat/tts`; the MP3 renders while 1–3 play.
5. **Voice plays** when the MP3 is ready; switch to the **speaking loop**; subtitles finish in sync (or are already shown — acceptable for them to slightly lead).
6. **Return to listening loop**; await next input. **🔊 replay** available (free, from `tts-output.js`).

This sequence is the heart of the build — get it smooth and the conversation feels alive.

### Strategic interjections ("breaking in")
A real conversation isn't strict turn-taking. **Judiciously and rarely**, Glendalf should *break in* — play a short spoken interjection **while the user is still typing** ("Ah—", "Yes, go on…", "Mm—", "Oh…", "Now *that's* the heart of it—"), as if he's listening so intently he can't hold back. This is the single biggest "he's *alive*" lever — but it's a delight only when **strategic**: rare, well-timed (at a natural typing pause), and never stepping on every message. Overuse is maddening.
- **v1 (light):** occasional interjection triggered by a typing pause (input focused + content present + a brief idle), capped to a low rate (e.g. at most once every few turns), from the interjection subset of the repertoire. Returns to listening immediately after.
- **Follow-on:** *content-aware* interjections — analyze the partial input and break in on an emotionally-charged or resonant phrase. This pairs with the emotion-matched video/filler work.

---

## 6. Data / persistence

New `chat_log.db` table `fireside_sessions` (follow the existing sqlite store patterns, e.g. `journal_store.py`):

| field | purpose |
|---|---|
| `id` (PK) | fireside session id |
| `amg_session` | anonymous session cookie linkage |
| `user_email` / `user_name` | null in v1 (anonymous); reserved for the next slice's capture |
| `started_at` / `last_turn_at` / `turn_count` | lifecycle |
| `transcript` (JSON) | full ordered turns `[{speaker, text, ts}]` |
| `ash_coverage` (JSON) | per-dimension state (untouched/opened/explored) + the excerpt that opened each |
| `signals` (JSON, optional) | per-turn analysis snapshots if computed |

Persisted so a returning visitor can resume ("when we last spoke, you mentioned…") and so the conversation seeds the later Remedy Match. Anonymous-session only in v1.

---

## 7. Assets (v1)

- **1× rendered lip-synced intro video** — Glendalf by the fire, in Glen's voice, ~10–20s greeting (HeyGen-style; the only lip-synced asset).
- **1× speaking loop** — mouth hidden (pipe/hand/beard), seamless, ~6–12s.
- **1× pondering/listening loop** — the characterful gap-cover actions: pipe + magical smoke ring, sip from the glass, reaching for/opening a book, turning to poke the fire (can be one montage loop or a couple short clips chosen at random). Doubles as the "thinking" cover and the idle "listening" state.
- **Filler-phrase repertoire** (~8–12 short clips) in Glen's clone, pre-cached for instant playback: *"Hmmm…", "Well…", "I see.", "That's very interesting.", "Let me consider that.", "You've given me much to contemplate there.", "Here's what I think…"*, etc. — a mix of acknowledgments, thinking beats, and bridges. Picked at random in v1 (contextual selection = follow-on). Includes a short **interjection subset** for the "breaking in" feature (§5): *"Ah—", "Yes, go on…", "Mm—", "Oh…", "Now that's the heart of it—"*.
- All assets are placeholders-swappable (manifest/asset-driven where practical); the **emotion-matched library is the follow-on**.

---

## 8. Scope, flag, and verification

- **Flag:** ships dark behind a new flag (e.g. `FIRESIDE_ENABLED`, default off); `/begin/fireside` 404s when off. Does not alter the live `/begin/doorway` or the journey quest.
- **Out of scope (follow-on specs):** Remedy Match reveal + $1 E4L trial; the adventure shell (map/scrolls/background art/sequencing); emotion-matched video selection; voice-in (tap-to-talk); identity/email capture; longitudinal cross-session analysis.
- **Verification:** unit tests for the agent endpoint (streamed turns, session persistence, ASH-coverage update, hook trigger) + the `fireside_sessions` store; **render-verify** the fireside UI in a headless browser (intro plays, type→send→subtitles→voice→loop transitions, "Hmmm" + pondering loop cover the gap, zero console errors) — per the project's render-verify discipline.

## 9. Open items to resolve in the plan
1. Conversational model (Haiku vs Opus) + the exact hook-trigger heuristic.
2. Whether to run per-turn `/journal/analyze` on typed input (probably a lighter text-only signal pass) to feed the ASH coverage map.
3. Subtitle timing: word-stream-with-text vs reveal-in-sync-with-audio (lean: stream text immediately, let voice catch up).
4. The pondering loop as one montage vs a few short randomly-chosen clips.
5. The strategic-interjection cadence + trigger (§5): how rare, what idle-pause threshold, the per-session rate cap — tune to feel alive without being intrusive.
