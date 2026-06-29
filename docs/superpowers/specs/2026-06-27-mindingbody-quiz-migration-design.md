# MindingBody™ → ASH Self-Assessment — native build on illtowell.com (v4: 12-dimension hierarchical)

**Date:** 2026-06-27
**Status:** Design / spec (pending review)
**Repo:** deploy-chat (illtowell.com)
**Supersedes:** v1 (ScoreApp replication), v2 (4-dim category), v3 (5-dim felt-sense). Glen
reframed the model 2026-06-27 around his full ASH 12-dimension framework, hierarchical/adaptive.

## Context

The "Where to begin" funnel sends quiz traffic OUT to an external ScoreApp quiz
(`healing.scoreapp.com`, "MindingBody™"); ScoreApp posts completions back via `/webhook/scoreapp`
→ GHL. illtowell.com already has a native quiz funnel (`/begin/quiz` → answer → opt-in → result;
`quiz_engine.py`, `data/quizzes.json`, `begin-quiz*.html`) but it runs a Neuro-Magnesium
"eye-brain" quiz and the live funnel + affiliate links point at ScoreApp. Original goal: get off
ScoreApp onto illtowell.com. The model has since grown into **Glen's ASH self-assessment**.

ScoreApp source content is captured in two committed reference docs
(`2026-06-27-mindingbody-capture-questions-scoring.md`, `-result-copy-verbatim.md`) — now mostly
superseded by the ASH structure below; reuse only for tone/voice.

## The ASH 12 dimensions (authoritative; each a five-fold)

Extracted from Glen's `ash-certification` skills + `01 Clinical/`. Each dimension is one ASH
module; its five categories have an inherent ordering (deepest root / most upstream → surface).

| # | Dimension | Five-fold (ordered) | Deepest-root end | Depth-scored? |
|---|---|---|---|---|
| 1 | **Body** (States of Matter) | Solid·Liquid·Gas·Plasma·Condensate | Condensate (quantum) | yes |
| 2 | **Mind** (5 C's) | Context·Container·Content·Connection·Communication | Context (outer field) | yes |
| 3 | **Spirit** (5 Elements) | Wood·Fire·Earth·Metal·Water | cycle — no single root | profile-only |
| 4 | **Inheritance** (5 Generations) | Grandparents·Parents·Self·Children·Grandchildren | Grandparents (past) | yes |
| 5 | **Personal History** (5 Penetration) | Gut/Env·Blood·ECF·Cytoplasm·Nucleoplasm | Nucleoplasm (genetic core) | yes |
| 6 | **Epigenetics** (5 Infoceuticals) | Terrains·Source·Organs·Meridians·Systems | Terrains/Source | yes |
| 7 | **Symptoms** (5 Cardinal Signs) | Rubor·Calor·Dolor·Tumor·Functio laesa | Functio laesa | yes |
| 8 | **Terrain** (5 R's) | Recharge·Rejuvenate·Regenerate·Reclaim·Regulate | Recharge (low-energy) | yes |
| 9 | **Diagnosis** (5 Pathology) | Hypertrophy·Hyperplasia·Metaplasia·Dysplasia·Neoplasia | Neoplasia (least reversible) | yes |
| 10 | **Treatment** (5 Therapy levels) | Surgery·Suppression·Substitution·Support·Stimulation | vital-force axis — no root | profile-only |
| 11 | **Regulation** (5 Levels) | Blocked·Negative·Mixed·Positive·Optimum | quality spectrum — no root | profile-only |
| 12 | **Prognosis** (5 Stages) | Self-Limiting·Serious·Degenerative·Life-Threatening·Certain Death | Certain Death (gravest) | yes |

**Supporting lens (not one of the 12):** **Tissue** = 5 Embryological Tissue Layers
(Compression·Connection·Conversion·Communication·Containment; deepest → surface). Used inside
Symptoms, and a mandatory follow-up dimension (below).

Notes: "Concern" (from ScoreApp) dissolves — it was a compression of modules 5/7/9/10/12.
"System" (earlier design) = the Tissue lens, not a top-level module. "Penetration" (earlier) =
module 5. "Phase" (earlier) = module 8 Terrain (5 R's). Profile-only dims contribute a resonance
profile, not a deepest-root (Glen to confirm which are profile-only).

## Flow — hierarchical / adaptive, opt-in between tiers

- **Tier 1 — lead magnet (pre-opt-in):** present the **12 ASH areas**; the user rates **any
  issues/challenges they want to heal or change, 1–10** (felt-sense slider; rate only what
  matters, leave the rest at rest). Relatable one-line prompt per area, **no academic names
  shown** (zero priming); ASH names revealed in the result. Fast (~1 min).
- **Opt-in gate** (email + TOS, existing `/begin/quiz/opt-in`) → **immediate result**: reveal the
  areas they flagged, name their top areas, light category context, and invite them to "go to the
  next level."
- **Tier 2 — follow-up (post-opt-in, in portal, only if they choose to go deeper at that time):**
  drill the five-fold of (a) **any dimension scored 8+** (motivated to act) AND (b) **always
  Terrain + Tissue + Regulation** (the pivotal clinical dimensions), regardless of score. Each
  drilled dimension: felt-sense sliders on its 5 categories + optional free-text. Scoring per
  depth-scored dimension: **strongest need** (max value) + **deepest root** (min depth-level among
  flagged); profile-only dims give a resonance profile. Plus a cross-dimension overall deepest root.
- **Tier 3 — AI deep-dive + chat (in portal):** an LLM synthesizes the full profile + free-text
  across dimensions into a personalized, root-cause-aware writeup; chat follows up on specifics.

## Input control (all tiers)

Each ratable item = a **numberless slider with two endpoint labels** (no digits; favors body
intelligence over analysis). Internal value 0–100; resting (left) end = not flagged. Optional
free-text per item (Tier 2+). Need items ≈ "Doesn't apply → Strong concern"; Regulation ≈ "Not my
experience → Exactly my experience".

## Delivery staging (sub-projects; each its own spec→plan→build)

- **SP1 — Tier-1 12-area quiz + funnel cutover (THIS spec's build scope; ships the get-off-ScoreApp
  goal):** the 12-area felt-sense lead-magnet quiz, opt-in, immediate "top areas" result + ASH
  reveal, repoint funnel + affiliate links to internal, native lead-capture parity, retire external
  ScoreApp.
- **SP2 — Tier-2 adaptive drill-down (later):** 8+ dimensions + mandatory Terrain/Tissue/Regulation,
  five-fold sliders + text, two-way scoring, delivered as the portal "next level."
- **SP3 — Tier-3 AI deep-dive writeup + chat follow-up (later):** portal-delivered personalized
  analysis.

## SP1 architecture

### A. Quiz config + engine
- `data/quizzes.json` → add `mindingbody-ash` (Tier-1 only for SP1): 12 `areas`
  (`{id, prompt, ash_name, dimension_key}`), slider endpoint labels. New `scoring.mode:
  "ash-tier1"`. (Five-fold category content lands in the SP2 spec.)
- `quiz_engine.py`: keep legacy `signals` path (eye-brain, dormant). Add Tier-1 path — pure
  functions: `flagged(ratings)`, `top_areas(ratings)` (sorted by value), `drill_candidates(ratings)`
  (value ≥ 8 → 80/100), `result_for(...)` returning flagged areas + top areas + ASH reveal +
  drill list. Ratings `{area_id:{value:0–100, }}` stored in `quiz_responses.answers_json`.

### B. Input UX (`static/begin-quiz.html` + `/begin/quiz-data`)
- 12 unlabeled relatable prompts, each a numberless 2-endpoint slider (0–100, resting = not
  flagged), all optional. No academic names. Payload `{area_id:{value}}`.

### C. Immediate result (`static/begin-quiz-result.html` + `/begin/quiz/result-data`)
- Reveal the flagged areas + ASH names, name top areas, light per-area context, and a "go to the
  next level" CTA (gated to post-opt-in / portal — wires to SP2/SP3 later). Payload-driven; legacy
  eye-brain still renders.

### D. Lead-capture parity on opt-in (`/begin/quiz/opt-in`, app.py ~1894)
Port `/webhook/scoreapp` behaviors (app.py ~15549): per-flagged-area GHL tags (`ash:terrain`,
`ash:symptoms`, …), a GHL note with the rated areas + values, referral/UTM logging,
practitioner-share offer, `_record_entry_unlock("quiz", …)`. Keep the threaded `_onboard` pattern.

### E. Repoint links → internal; retire external
- `app.py:1849` `_ACTIVE_QUIZ_ID = "mindingbody-ash"`.
- `begin_funnel.py:134` & `:388` quiz land `base_url` → internal `/begin/quiz` (`internal:True`).
- All `app.py` affiliate `healing.scoreapp.com?utm_…` URLs (QUIZ_URL ~8904 + builders
  ~8031/8872/9150/9445) → internal quiz path w/ existing `?ref`/`rm_ref`.
- Leave `/webhook/scoreapp` dormant (~10 historical leads already in GHL).

## Affected files (SP1)
`data/quizzes.json`, `quiz_engine.py`, `static/begin-quiz.html`, `static/begin-quiz-result.html`,
`app.py` (`_ACTIVE_QUIZ_ID`, opt-in parity, link repoints), `begin_funnel.py`, +
`tests/test_quiz_ash_tier1.py` and touch-ups to `tests/test_begin_funnel.py`,
`tests/test_begin_journey_map.py`.

## Verification (SP1)
- Unit: ratings → flagged/top areas, drill candidates (≥8), result-data shape; opt-in emits
  expected GHL tags + note.
- Local run (deploy-chat doppler + DATA_DIR override): `/begin/quiz-data` → POST ratings → opt-in
  → `/begin/quiz/result-data`.
- Render-verify: headless `/begin/quiz` (12 sliders, no academic names) + `/begin/quiz/result`
  (areas revealed, top named) — DOM + zero console errors.
- Confirm `/begin` quiz land + a sample affiliate link resolve to internal `/begin/quiz`.

## Open items for Glen
1. Correct the **12 Tier-1 relatable prompts** (draft below) — the felt-sense, no-jargon wording.
2. Confirm the **profile-only** dimensions (no deepest-root): proposed Spirit (3), Treatment (10),
   Regulation (11). Any others? (e.g. Mind/Body if the root direction is unclear.)
3. Confirm **8+** drill threshold and the **mandatory follow-up trio** = Terrain, Tissue, Regulation.
4. Confirm SP1 / SP2 / SP3 staging (SP1 ships the ScoreApp cutover).

### Draft Tier-1 prompts (PROPOSE → Glen corrects; no academic names shown)
1. Body — "How your physical body feels — solid, depleted, heavy, or off"
2. Mind — "Mental focus, emotional patterns, and how you connect/communicate"
3. Spirit — "Sense of meaning, purpose, and emotional balance"
4. Inheritance — "Health patterns that run in your family/lineage"
5. Personal History — "Your own health history and how deep issues have gone"
6. Epigenetics — "Your body's energy/terrain and how it's regulating"
7. Symptoms — "Active symptoms — pain, heat, swelling, redness, lost function"
8. Terrain — "Your body's vitality and capacity to heal"
9. Diagnosis — "Diagnosed conditions or tissue changes"
10. Treatment — "The treatments you're using and how they're working"
11. Regulation — "How your body responds when you try to heal"
12. Prognosis — "How serious or progressive your main concern feels"

## Out of scope (SP1)
SP2 (drill-down) and SP3 (AI writeup + chat). Migrating ~10 historical ScoreApp leads (already in
GHL). Removing `/webhook/scoreapp` (dormant). Pixel-identical ScoreApp theme.
