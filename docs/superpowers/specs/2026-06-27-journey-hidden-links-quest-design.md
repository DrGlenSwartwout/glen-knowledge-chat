# Journey "Find the Hidden Links" Quest — Design Spec

**Date:** 2026-06-27
**Status:** Design approved (iterative, via playable prototype). One reward-model detail flagged for confirmation.
**Supersedes/extends:** `2026-06-26-journey-trademark-scene-design.md` (that spec covered a static named scene; this one turns the scene into an interactive, gamified, ordered onboarding quest). Builds on the navigation shell (`2026-06-24-navigation-shell-journey-map-design.md`).
**Flag:** ships behind the existing `JOURNEY_SHELL_ENABLED`. Interface-only — no funnel engine/gate/points/routing change to the underlying `begin_funnel` keys (`scan/find/heal/give`).

---

## 1. What this is

The journey shell's expandable map becomes a **gamified onboarding quest** called **"Find the Hidden Links."** Instead of handing the user a nav menu, we make them *discover* the brand journey by exploring an illustrated Shire scene — which teaches the five stages, rewards curiosity, and turns wayfinding into a small delight (and is genuinely fun for kids).

**The blessed master scene** is `5-journey-unified-v12-trumpet.png` (1328×800, 16:9): a hobbit-hole home (lower-left) → a winding path → a glowing white-gold elvish cathedral (far right, by a lake), with the wizard **Glendalf** seated on a rock cupping his ear (listening) and handing a dark-violet remedy bottle to a hobbit woman. The scene art is final. (Source/iteration history lives in `~/Downloads/journey-ribbon-samples/landmarks/`; the playable interaction prototype is `hunt-prototype.html` in that folder.)

**Top design principle (Glen, explicit): sound quality matters more than the visuals.** The audio *is* the experience. Production must use real, high-quality recorded cues — and the spoken lines in Glen's own voice clone (the existing ElevenLabs clone `jFxSqMckq2I4mET3C5QC`, Doppler `remedy-match/prd`) — not the synthesized Web-Audio stand-ins used in the prototype.

---

## 2. The five stages (display names = trademarks; engine keys unchanged)

| engine key | trademark name | hidden hotspot in the scene |
|---|---|---|
| `home`* | Home | the whole green hobbit-hole door |
| `scan` | Wellness Whispering | Glendalf's right hand & ear |
| `find` | Remedy Match | the remedy bottle & the woman's hands |
| `heal` | Accelerated Self Healing™ | the winding path into the distance |
| `give` | Healing Oasis | the whole cathedral |

\* `home` is a wayfinding entry, not a funnel engine step. The four engine keys (`scan/find/heal/give`) and their real hrefs/actions are unchanged; this layer only adds discovery + display.

The journey reads left→right and is played **in order**: Home → Wellness Whispering → Remedy Match → Accelerated Self Healing™ → Healing Oasis.

**Hotspot coordinates** (as % of the 1328×800 scene, center x / center y / width / height — from the prototype, to be fine-tuned):
- Home: 14 / 57 / 13 / 23
- Wellness Whispering: 43 / 35 / 12 / 19
- Remedy Match: 64 / 56 / 13 / 17
- Accelerated Self Healing™: 62 / 44 / 14 / 13
- Healing Oasis: 72 / 18 / 16 / 27

Hotspots are **invisible** — no dots, no outlines, no labels on the art. Discovery is by **audio proximity** (§4), preserving the clean scene.

---

## 3. The mechanic

### 3.1 Three parallel "adventure paths," one progress state
A user can engage each stage three ways, all of which count toward the same progress and light the same header link + chip:
1. **Chat** — interact with the assistant.
2. **Video** — watch that stage's video.
3. **Hunt** — find the stage's hidden link in the scene.

Different users prefer different rails (kids love the hunt). The rails are **additive for rewards** (§5): each rail's *first* touch of a stage pays its own reward.

### 3.2 Two required steps per stage (DECISION (a), locked)
A stage is **cleared** (its header link permanently activates, and the next stage becomes findable) only when **both** of its steps are done:
1. **Find** the hidden spot in the scene, **and**
2. **Engage** that stage's content — satisfiable by **either** video **or** chat.

"Find" alone, or "engage" alone, is not a clear — but each still earns its own reward when first done. Ordering is enforced: you cannot find stage N+1's spot until stage N is cleared.

### 3.3 Entry flow (first run)
1. **Clean landing** — no header ribbon, no scene image.
2. The moment the user does **either** first action (plays the welcome video **or** touches the chat), the **journey UI appears all gray/locked** (header ribbon + the scene below it).
3. The first hunt target is **Home**: the user must find and click the **door** to activate the Home button + icon (the icon is a small crop of the hobbit home — see §6).
4. Proceed in order through the remaining four.

### 3.4 First-run onboarding, with an escape hatch (guardrail)
The quest is a **first-run onboarding ritual**, not a permanent cage. Once the quest is completed — or for a known/returning member, or via an explicit "skip the tour" — the header reverts to a normal, fully-active nav. **The game must never permanently block a real customer from reaching a real page.** (Reconciles "any path unlocks" with "don't trap people.")

### 3.5 Always-visible "you are here"
The header always shows, for the user *and* for our analytics, what is **done**, what is **current** (highlighted/pulsing), and what is **locked**. This maps directly onto the shell's existing "next" concept.

---

## 4. Audio design (the heart of the experience)

Each stage has a **two-phase** sound: an **approach** ambient and an **arrival** moment.

- **Approach:** the current target's signature sound is **faintly audible across the whole scene and swells as the cursor nears** the hidden spot ("getting warmer" by ear). This *replaces* any visual proximity cue, keeping the art clean. In the prototype only the **current** target's sound plays so the hunt stays solvable by ear; a fuller multi-spot ambient bed is a production option once stages are discovered.
- **Arrival:** on the click that finds the spot, the approach sound **transforms** into a distinct arrival cue (often a spoken line in Glendalf's voice).

| Stage | Approach (swells with proximity) | Arrival (on the find/click) |
|---|---|---|
| **Home** | muffled talking / singing / dinner behind the closed door | door **creaks open** + a hearty **"Welcome, friend!"** in Glendalf's voice (also replays on every later Home click) |
| **Wellness Whispering** | **172 Hz Tibetan singing bowl**, louder toward the ear | **counting "1 … 2 … 3 … up to 10"** at the ear |
| **Remedy Match** | **succussion** of a homeopathic remedy (≈once/sec) + **mortar & pestle** grinding | the sound of relief — **"Ahhhh!"** |
| **Accelerated Self Healing™** | **footsteps** walking on the dirt path | **"And I'm going with you."** (Sam's line, LOTR) in Glendalf's voice |
| **Healing Oasis** | **breeze + songbirds** | adds **water fountains + wind chimes** on arrival |

**Production notes:**
- All cues to be produced as real recorded/curated audio assets (see §8). Synthesized Web-Audio versions in the prototype are placeholders for shape/timing only.
- Spoken lines ("Welcome, friend!", the count, "Ahhhh!", "And I'm going with you") in Glen's voice clone.
- **Rights gut-check** before launch on the LOTR line "And I'm going with you" (recognizable IP in a paid product); swap to an original line if needed.
- Audio requires a user gesture to start (browser autoplay policy) — satisfied because the UI only appears after the user's first video/chat action.
- Provide a global mute/volume control and respect `prefers-reduced-motion` / a sound-off preference; the hunt must remain completable with a visible fallback hint when sound is off (e.g., the guided hint line, or a "reveal" affordance) so audio-off / deaf users aren't blocked.

---

## 5. Reward model — progressive, never penalized by sequence

- **Discount grows with breadth, caps at 15%:** engaging a stage via 1 / 2 / 3 of the paths yields **5% / 10% / 15%**. More rails always *raise* the %, so order never penalizes the user.
- **Coupon #1 (for yourself):** issued at the user's **first stage completion** (so they hold it early); its value climbs as more paths are completed.
- **Coupon #2 (to give away):** issued at **full quest completion** — thematically mirrors **Healing Oasis = "give,"** the final stage.
- **Per-pathway first-touch reward:** the *first* time each rail (chat/video/hunt) brings the user to a given stage, it fires at least a visual + auditory gamification reward. A single stage can therefore pay out up to three times.
- **Finding a hidden link always pays out** — even if the stage was already opened via chat or video. The hunt is never "used up" by another rail.

**Visual reward:** on-screen celebration toast per find/clear; a bigger finale when all five are lit.
**Auditory reward:** the stage's signature fanfare (per §4) plus a short rising flourish; a fuller fanfare on full completion.

### OPEN DECISION (needs Glen's confirmation)
Is the **5/10/15** counted **per-stage** (each stage can reach 15% on its own by doing all three rails there) **or once across the whole journey** (a single personal coupon climbs 5→10→15 as paths accumulate anywhere)?
**Proposed default:** *single growing personal coupon* (simpler to reason about, simpler to redeem, and matches "get the coupon for yourself with the first stage … % can still increase with path finding"). The prototype implements this default. **Confirm or override before implementation.**

---

## 6. Header ribbon states

- **Locked** (default): grayscale + reduced opacity, not clickable, a small lock glyph.
- **Current**: the next-to-clear stage, gold-tinted with a gentle pulse ("you are here / go find this").
- **Unlocked**: full color, clickable link into the real stage action; subtle gold glow. Clicking an unlocked link replays its signature sound (e.g., Home re-creaks).
- **Icons**: each header item's icon is a small **live crop of that hotspot region from the scene** (Home icon = the hobbit door, etc.), grayscale when locked, full color when unlocked. (In production these can be pre-rendered thumbnails per the asset plan, mirroring `thumb-{key}.webp`.)
- The **funnel chips** activate identically and in lockstep with the header links.

---

## 7. Persistence, scope, platforms

- **Persistence:** per-stage `{found, done}` plus per-rail first-touch flags and the coupon state, in `localStorage` (and, where a known member, mirrored to their server profile so progress follows them across devices). Once a link is unlocked it stays unlocked.
- **Relationship to the funnel engine:** "engage content" maps onto the real `begin_funnel` stage actions the engine already tracks; the quest is a presentation/onboarding layer over real state, not a separate source of truth. The hunt + rewards are the new layer.
- **Mobile / touch (no hover):** the header tap opens the scene; tap to hunt. Because the audio "warmer" proximity cue is weaker without a moving cursor, mobile gets a **subtle guided hint** (e.g., the current target name in the header + a one-time "explore to find it" line). (Glen: subtle hint on mobile — yes.)
- **Accessibility:** every hotspot is a real focusable control with an `aria-label` (name + status); keyboard users can tab the targets in order; sound-off path remains completable (§4).

---

## 8. Assets

- **Scene:** `5-journey-unified-v12-trumpet.png` → optimized `static/journey/scene.webp` (~1600px, <300 KB) per the prior build-assets plan.
- **Icons:** five hotspot-region crops → `static/journey/thumb-{home,scan,find,heal,give}.webp`.
- **Audio (10 cues + fanfares):** approach + arrival per stage (table §4), plus reward flourishes and a completion finale. Its own production workstream ("work on all the sounds" — Glen). **Confirmed for real recordings:** Glen's spoken voice lines (via the voice clone, or live VO) **and** the 172 Hz Tibetan bowl, at minimum — likely others too. Remaining cues may use curated/recorded library sound where a synthesized stand-in won't sell the experience (sound quality is the top priority). The Web-Audio versions in the prototype are placeholders for shape/timing only.

---

## 9. Out of scope / future

- Replacing the static per-station detail art (the four blessed landmark images) — unchanged.
- A full multi-spot ambient sound bed (all five sounds layered across the scene at once) — production option, not v1.
- Server-side coupon issuance/redemption plumbing beyond recording entitlement — coordinate with the existing membership/coupon mechanisms separately.
- Any change to `begin_funnel` engine keys, routes, or points.

---

## 10. Open items to resolve before/within implementation

1. **Reward scope:** per-stage vs single growing coupon (§5 OPEN DECISION) — proposed default = single growing coupon.
2. **"And I'm going with you"** rights check (§4) — keep or swap to an original line.
3. **Coupon redemption mechanics** — how coupon #1 and the giftable coupon #2 are actually issued/redeemed in the existing store/membership flow.
4. Final **hotspot coordinate** tuning on `scene.webp` (Phase-1 gate).
5. Final **audio asset** production (separate sound workstream).
