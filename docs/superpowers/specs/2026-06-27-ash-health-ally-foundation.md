# ASH Health Ally — Foundation & Roadmap

**Date:** 2026-06-27
**Status:** Foundation (living) — distilled from an extended brainstorm with Glen
**Repo:** deploy-chat (illtowell.com)
**Supersedes:** the "MindingBody quiz migration" specs v1–v4. The work began as "move the ScoreApp
quiz to illtowell.com"; it converged into building a voice-first AI health ally. That earlier spec
file and the two ScoreApp capture docs remain as **references** (ScoreApp drove people to the E4L
scan — which is exactly the keystone below).

---

## 1. North star

Understand each person's **uniqueness** in order to **serve their needs toward a higher quality of
life** (health, performance, experience). The inner aim: have them *experience* an **ally and
partner** — relationship, meaning, purpose, trust, empowerment. The clinical work — *the keys that
pick their locks, the little hinges that swing the big doors* — is in service of that, not the
point. **We help them write and implement the lost owner's manual for their own body.**

We optimize for **depth and accuracy of understanding + the felt relationship.** Conversion, data,
and routing fall out of that — not the other way around.

## 2. The person we serve

They start at a **mismatch**: a symptom/condition they have vs. the healing/wholeness they want.
They **intuitively know there's an answer, and that it isn't in conventional care/drugs.** They
already sense they can heal; they're missing the pieces and a guide.

## 3. The belief shift (the core reframe — their own Treatment dimension)

Make conscious the contrast between two roads, so their mind is **freed to choose**:
- **The conventional road** — *Surgery · Suppression · Substitution* (the lower three Levels of
  Therapy; the conscious mind's trained offering, ~50–120 bits/s) — leads *further toward* what
  they're trying to escape.
- **Our road** — *Support · Stimulation* (the upper two; vital-force-increasing) — partners with the
  body's intelligence, **orders of magnitude greater**, now amplified by **AI**.

Then we hand them the missing pieces: **listen to the body's messages → translate them to action →
supply the materials, energies, and information/meaning the body is asking for, to complete the
healing it is already trying to do — now.** A symptom is not the enemy; it's a message.

## 4. What we're building — an ally, not a quiz

The "quiz" dissolves. We're building the **first turn, and then the ongoing turns, of a voice-first
AI health ally.** The person never sees a form or a grid. The 12 ASH dimensions are the ally's
**private map** for coming to know the whole person over time.

## 5. Experience principles

- **Voice-first, voice-out (key).** They speak; the ally speaks back **in Glen's own cloned voice**
  (ElevenLabs `jFxSqMckq2I4mET3C5QC`). Type is an option, but voice is more natural, lower-effort,
  more revealing — and the **voice itself carries the body's message** (stress, word-vs-energy
  incongruence).
- **The ally opens and invites.** It starts the conversation and invites them to share.
- **Meet them where they are.** Some unload; some test the water. Both are welcome; no forcing.
- **Depth from the first breath** — in the quality of presence, not deferred to "later sessions."
- **NLP / mirroring** — learn and reflect *their* language and model of the world.
- **Listen for congruence** — words vs. energy; the mismatch is often where the real message lives.
- **Empathic entry & self-discovery** — enter *their* world; arrange the conditions so **they
  discover the path themselves**. Knowing them lets us set the stage for their adventure.

## 6. The relationship model

A **persistent, multi-session** relationship, not an event. We remember, per person, **what we've
covered and what we haven't** — so the conversation is **guided by their answers**, surfacing
3rd/4th-order detail (their story, their words, the felt texture) no category grid can hold. The
ASH coverage map fills in **quietly behind the scenes**, deeper where they light up.

**Navigation policy — follow them; the map only opens doors they crack.** The coverage map is not an
agenda. Its only job is to give the ally **peripheral awareness of what's still unexplored**, so that
when the person opens a door — a mention of a parent, of early childhood, of a past treatment — the
ally can step through it *naturally* into the uncovered dimension (e.g. that parent-mention →
Inheritance: *"that makes me wonder what patterns came down through your family — any challenges you
know of?"*). It **paces, then leads — always through their own openings. No non-sequiturs.** For
must-reach dimensions (Terrain, Tissue, Regulation) the ally is patient: over the relationship it
*cultivates* adjacent openings rather than forcing a jump. A topic is only ever raised when the
thread the person is already pulling makes it feel inevitable.

**The engine loop (every interaction):** **hear → feel → see where they're stuck → offer a path.**

Continuity: starts in the funnel (`/begin`), continues in the portal post-opt-in, across sessions
with felt continuity ("last time you named…").

## 7. The keystone: their first voice scan

**The single biggest step is the first E4L voice scan** — and the whole doorway exists to lead them
there. The elegant collapse: **the conversation *is* the scan.** They don't brace to "go take a
scan"; they just talk to something that finally listens — and that *is* the read. The hardest
commitment becomes the most human moment.

**Two-scan model:**
- **Native doorway scan (the first scan):** the existing voice-journal pipeline, reframed as the
  ally's listening — instant, on-site, ungated, reuses everything. (See SP1.) **This ships now.**
- **External E4L Biofield Analysis (the deeper, paid next level):** truly.vip/E4L → partner →
  `/portal/<token>` biofield report. The upsell tier.

**Track B (aspirational upgrade — invisible real E4L scan):** capture the person's *count-1-to-10*
clip on-platform and submit it to E4L behind the scenes, so they get the REAL bioenergetic scan
with no account/leaving the page, and the ally's first questions ride the real energy data. Then
the two-scan model could collapse into one.
*Feasibility (researched 2026-06-27): chosen approach = browser automation (no E4L API exists).*
The E4L integration is browser-automation today (login `portal.e4l.com` as practitioner via
`E4L_USERNAME`/`E4L_PASSWORD` in Doppler `remedy-match/prd`, `LoginAs/{client_id}` to impersonate,
scrape result PDFs via `/Scans/ScanPDF/...`); **result retrieval already works**. Protocol confirmed
("30s, count 1 to 10", `app.py:1328`); portal has `openVoiceScanForUser(id,name)`.
**Submission mechanism (the new part):** there is no file-upload and no API, so we feed our recorded
WAV as the *microphone* via headless Chromium flags
`--use-file-for-fake-audio-capture=<count.wav> --use-fake-device-for-media-stream` (bit-exact, zero
loss if sample-rate matches), driven by Playwright: create client → `openVoiceScanForUser` → fake-mic
plays the count → scrape result → run OUR ASH/TCM AI on top of E4L's. New client accounts created on
the user's behalf using **E4L's own Generate-password button + "Email login details to client"
checkbox** — E4L mints a unique password and emails it directly to the client, so **we never
generate, store, or handle it** (never in repo/`e4l.db`). The client owns it; `LoginAs`
impersonation means our automation never needs it.
**Authorization: CONFIRMED (Glen, 2026-06-27)** — running scans for clients through the practitioner
account is sanctioned and already wired in for in-person clients. Our only delta is the voice
*input* (pre-recorded WAV via fake-mic vs. live in-room mic) — same authorized act, the client's own
voice; user consent handled via the doorway TOS.
**SPIKE RESULT (2026-06-27) — fake-mic path is VIABLE.** Inspected the live portal
(`portal.e4l.com`, Glen-authed, read-only). Two distinct scan modalities exist; don't confuse them:
- **Device scan = "BioSync," a downloadable DESKTOP app** (Win `.exe` / Mac `.dmg`,
  `/Root/DownloadSoftware`) that drives a physical instrument reading the body field. NOT our target;
  not browser-automatable. (This is what threw the first read.)
- **Voice scan = browser `getUserMedia` capture.** On the **Clients page** the probe found
  `getUserMedia: true`, `mediaDevices: true`, `openVoiceScanForUser` is a live function, "voice scan"
  present, no Flash/applet. The practitioner runs a client's voice scan from the portal via
  `openVoiceScanForUser(client_id, name)`, which opens a browser mic recorder. `MediaRecorder:false`
  + `mediaDevices:true` ⇒ likely Web-Audio/AudioContext analysis fed from the mic stream.
**So Track B works as originally designed:** create the client → call `openVoiceScanForUser` → feed our
count-1-to-10 WAV via Chromium `--use-file-for-fake-audio-capture` (16-bit PCM WAV) → the portal's
`getUserMedia`/AudioContext receives it like a real mic → scan submits → result syncs and we scrape it
(retrieval already works) → run our ASH/TCM AI on top. All headless-browser, server-side — no desktop
app, no virtual audio device.
**Confirmed entry point (spike):** portal **Scans tab → "BWS Voice Scan" button** (BWS = Bioenergetic
Wellness Scan; also a "BWS Foundation" scan type) → `openVoiceScanForUser` opens the browser
`getUserMedia` recorder. Feed our WAV via fake-mic here.
**Live flow confirmed (2026-06-27):** Glen ran a real BWS Voice Scan (test client, live mic) — accepted.
**Playwright recon (2026-06-27, headed Chromium + fake mic, login via Doppler):** the BWS Voice Scan
recorder is an **in-page flow** (rendered as an in-page modal under automation — no separate window to
juggle). Confirmed `getUserMedia: true`, **AudioContext present, `MediaRecorder: false`** ⇒ Web-Audio
capture fed from the mic stream → **fake-mic injection is architecturally sound** (our WAV feeds that
`getUserMedia`). `getUserMedia` fires on record-START (didn't capture in recon since we didn't start).
Flow controls seen: a **"test data client" quick-create form** (gender/region/species/country → Create)
= a safe throwaway client with no real PII; recorder buttons **"Repeat Recording" / "Proceed"**; confirm
dialogs incl. a **"Notify Client"** that automation MUST avoid. WAV ready = Glen's real count
(`/tmp/count.wav`, 48kHz/16-bit, 12.7s, loops to fill 30s).
**Lesson:** scope any portal automation to the scan dialog only — a broad DOM dump surfaced the live
client roster (PII); discarded, not stored.
**✅ PROVEN END-TO-END (2026-06-27).** A pre-recorded count-1-to-10 WAV, fed via Chromium fake-mic
(`--use-file-for-fake-audio-capture`) into a manually-driven BWS Voice Scan on a test-data client,
produced a **real E4L scan result.** Track B is fully validated: an invisible E4L voice scan from a
pre-recorded clip works — no app, no live mic, server-side-automatable. The single biggest technical
risk in the vision is retired with a working demo. Repeatable: `bash ~/e4l-fakemic.sh` (script + WAV in
vault `02 Skills/e4l-fakemic-test.py` / `e4l-fakemic-count.wav`).
*Validity note:* `--use-file-for-fake-audio-capture` REPLACES the mic device at the browser level, so
the recorder receives the WAV, not the live mic — the result is from the injected file even if the
operator also spoke aloud. **Airtight confirmation DONE (2026-06-27): re-ran with the operator
completely SILENT — a scan result still returned.** Proves the injected WAV alone drove the scan, zero
live-voice contamination. Track B definitively confirmed.
**For the Track B BUILD (own SP):** wrap this into full headless automation — login → create client
with required PII (name/gender/email/DOB/address; E4L Generate+email password) → BWS Voice Scan with
fake-mic injection of the person's WAV → scrape the result → our ASH/TCM AI on top. Runs as an
out-of-band worker on a machine with a browser. **Never click "Notify Client" in automation.** Cleanup:
archive the test-data client created during the proof.
**E4L Add Client form fields (captured by Glen 2026-06-27):** species (Human, default), first name,
last name, gender, email, language (English, default), **password (Generate button + "Email login
details to client" checkbox)**, country, address (up to 3 lines), city, state, zip, telephone, a
consent checkbox ("I have permission from my client to store their personal data and have made them
aware of our privacy policy"), and "save as delivery address" checkbox. The two checkboxes "Email
login details to client" and "Save as delivery address" are **optional, but we set BOTH** (client
owns/receives their account so we never handle the password; address saved for future remedy orders).
**Required for a valid scan/account (Glen, corrected):** first name, last name, gender, email,
**DOB**, and **full address** (country, address lines, city, state, zip). DOB + address are NOT
optional. (DOB wasn't in the initial field list above — confirm in the spike whether it's a field on
this form or collected at the scan step.)
**Intake reality (corrected):** a full intake is required up front — first/last name, gender, email,
**DOB, full address**. This is heavier than email-only, so it pushes against the "just talk" feel and
makes **where capture lands in the doorway** a real design decision: place the intake **after the
ally has earned trust and landed the first true hinge** (so they're motivated and feel met before we
ask), and gather it as gracefully as possible — DOB and the personal bits by voice/conversation, the
address via a light auto-filled form. Minimum to *start the conversation* is still just showing up;
the full intake gates the *scan*, not the first exchange.
**Consent:** the E4L permission checkbox is satisfied by **folding E4L's privacy policy into our
doorway TOS** (Glen's call) so accepting our TOS = the client made aware. (Confirm with E4L/legal
that referencing their policy in ours suffices.) TOS also covers "we run a bioenergetic scan via
partner E4L on your behalf."
SP1 still ships the native doorway scan regardless; Track B layers on once the spike + E4L blessing
clear. Evidence: `02 Skills/scrape-e4l-http.py`, `e4l-daily-watch.sh`, `e4l.db`,
`00 Integration Partners/e4l-energy-4-life.md` (interop is an open TODO).

## 8. The ASH 12-dimension private map (the ally's hidden coverage rubric)

Twelve dimensions, each a five-fold (authoritative; from `ash-certification` + `01 Clinical/`):
Body (States of Matter) · Mind (5 C's) · Spirit (5 Elements) · Inheritance (5 Generations) ·
Personal History (5 Penetration) · Epigenetics (5 Infoceuticals) · Symptoms (5 Cardinal Signs) ·
Terrain (5 R's) · Diagnosis (5 Pathology) · Treatment (5 Therapy) · Regulation (5 Levels) ·
Prognosis (5 Stages). Plus the supporting **Tissue** lens (5 embryological layers). Full five-folds,
depth orderings, and the felt-sense framing live in the captured reference doc. Each category carries
a depth level (1 = deepest root → 5 = surface); the ally can read both **strongest need** (what's
most alive) and **deepest root** (the lowest-level thing they flagged — the hinge). The three
pivotal dimensions to always reach over time: **Terrain, Tissue, Regulation.**

## 9. Roadmap

- **SP1 — the voice-first doorway that *is* their first scan (build first; replaces ScoreApp).**
- **SP2 — the persistent deepening:** the ASH coverage map as durable per-person state; answer-guided
  next-question policy (follow their thread vs. quietly fill the map); multi-session continuity in the
  portal; the always-reach trio.
- **SP3 — the ongoing ally:** cross-session AI synthesis (the personalized owner's-manual writeup),
  remedy/action plans, and the chat that keeps writing the manual with them.

## 10. SP1 — first build (concrete)

**Goal:** a native, voice-to-voice ally on `/begin` whose conversation culminates in the person's
first (native) voice scan, captures them as a lead, lands one true hinge, and invites them onward —
fully replacing the external ScoreApp quiz.

**Reuse (≈the whole analysis stack already exists — `journal_blueprint.py`):** Web-Audio capture,
Whisper transcription, lexical features, **Haiku TCM/Three-Treasures/polyvagal/congruence analysis**,
ada-002 embedding, Pinecone remedy match (`e4l-protocols` + `specific-formulations`), affirmation
generation; `dashboard/journal_store.py` storage; `begin_funnel.record_unlock()`; portal token auth.

**Net-new:**
- **Conversational voice-to-voice wrapper:** the ally opens (voice), invites sharing, listens
  (record → analyze), reflects what it hears/feels, sees the stuck point, offers a path — across
  multiple turns. System persona embodies §3 belief shift + §5 principles.
- **Voice-out:** ElevenLabs TTS in Glen's clone for the ally's spoken turns.
- **Lead-capture parity:** the journal is member-only with no GHL; the doorway must capture the lead
  (email/account) and port the `/webhook/scoreapp` behaviors onto opt-in (GHL onboard + tags + note,
  referral/UTM, `_record_entry_unlock`).
- **Session grouping / persistence:** group multi-turn voice turns into one "doorway scan" record
  (add `session_id`/context to `journal_entries`, or a `conversation_scans` table).
- **Funnel cutover:** `_ACTIVE_QUIZ_ID`/quiz land + affiliate `healing.scoreapp.com` links →
  internal doorway; leave `/webhook/scoreapp` dormant.

**The doorway arc:** meet the mismatch (their words, voice) → honor the intuition → make the contrast
conscious → reframe the symptom as a message → **hear & feel them via the scan** → land one true
hinge (a matched remedy / affirmation / insight) → invite to continue + capture (so the relationship
persists).

**Verification:** end-to-end local run (speak → transcribe → analyze → ally voice reply → remedy →
capture); render-verify the `/begin` doorway (mic capture, voice-out plays, zero console errors);
confirm `/begin` + a sample affiliate link resolve internally (no `healing.scoreapp.com`); opt-in
emits the expected GHL tags/note.

## 11. To refine (open)

1. Confirm **native scan as the doorway** (recommendation A/C) with external E4L Biofield as the
   paid deeper tier.
2. SP1 doorway scope: how many turns before the opt-in/capture? Where exactly does capture land in
   the arc (after the first reflected hinge feels right)?
3. Voice-out latency/UX (TTS streaming) and the mic/turn UX on mobile.
4. RESOLVED — navigation policy (§6): the ally always follows them; the map only opens doors they
   crack; must-reach dimensions are reached by cultivating adjacent openings over time, never by
   non-sequitur. (Session one: follow them; capture the read; touch the map only where they open it.)
5. The persistent coverage-map data model (SP2) — sketch early so SP1 stores compatibly: per-person,
   per-dimension/category coverage state (untouched / opened / explored / deep) + the verbatim
   excerpts that opened each door.

## 12. References (committed alongside)
- `2026-06-27-mindingbody-capture-questions-scoring.md` — ScoreApp source (questions/scoring/tiers).
- `2026-06-27-mindingbody-result-copy-verbatim.md` — ScoreApp result copy.
- `2026-06-27-mindingbody-quiz-migration-design.md` — superseded v4 quiz spec (ASH structure + the
  funnel-cutover mechanics, still useful for the link-repoint detail).
- ASH 12-dimension structure: extracted from `ash-certification` skills + `01 Clinical/`.
