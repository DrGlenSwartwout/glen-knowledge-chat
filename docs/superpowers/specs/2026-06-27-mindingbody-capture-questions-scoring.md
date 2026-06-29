# ScoreApp "MindingBody™" quiz — captured config (Accelerate Self Healing, LIVE)

Source: manage.scoreapp.com builder, scorecard "Accelerate Self Healing"
Public URL: https://healing.scoreapp.com  | Footer: Remedy Match, © 2025
Theme: dark forest-green bg, cream/gold accents, biofield-humanoid hero image.

## Intro / hook (landing)
Title: MindingBody™
Body: "4 things show / How you can heal / Up to a year of harm / In the next 30 days: /
Where you need help… / What's wrong… / Why… / And how fast you could heal."
Start button.

## Lead capture: FIRST (before questions) — First name, Last name*, Email* — "How shall we reach you?"
(NOTE: opposite of native funnel which captures email at opt-in AFTER answering.)

## 4 questions, each = a scoring CATEGORY (dimension), each answer scores that category 1–5

### Q1 — Section "System" — "Which system is most in need?" (single, required)
Muscles 1 | Urogenital 1 | Bones & Connective Tissue 2 | Immune 2 | Cardiovascular 2 |
Digestive 3 | Respiratory 3 | Endocrine 4 | Nerve 4 | Skin 5

### Q2 — Section "Phase" — "What's the main challenge?" (single, required)
Cancer 1 | Degeneration 1 | Viral 1 | Low Energy 1 | Aging 2 | Bacterial or Parasite 2 |
Fungal 3 | Regeneration 3 | Toxicity 4 | Allergy 4 | Stress 5 | Hormones 5

### Q3 — Section "Concern" — "What's your top concern?" (single, required)
Family Pattern 5 | Past Health 3 | Function 4 | Symptoms 4 | Risk 5 | Illness 2 |
Treatment 3 | Response 2 | Prognosis 1 | Degeneration 1

### Q4 — Section "Regulation" — "How well you heal when you try:" (single, required)
No Change 1 | Worse First 2 | Mixed Results 3 | Better Slowly 4 | Better Fast 5

## End logic: All users -> "Main Result Page"

## Category descriptions (Categories tab)
- System = "Location of issue"
- Phase = "The type of healing challenge"
- Concern = "Your top issue"
- Regulation = "Response to well selected interventions"
Each category LOGIC = "Add category score to total" (sum into one total). Total range 4–20.

## Result page = ONE "Main Result Page" (/main-result-page, DEFAULT). All users -> it.
Scoring: each dimension shown as % (score/max per dimension) + Overall Score % (donut gauge).
Dynamic content by band: low / medium / high (per-block variant toggle).

Sections: Header | Lead Form Popup | Change Details Form Popup | Result Expiry Popup |
Donut Chart (overall + 4 dimension gauge) | Category Scores (System/Phase/Concern/Regulation cards) |
Call to Action 16 (3 Items) | Footer

### Overall narrative — LOW (e.g. 20%)
"Your overall score reflects a terrain under significant load — where the body's healing
intelligence is working against resistance. In the Five Levels of Regulation, this range
corresponds to Blocked or Negative terrain states: conditions where the body's response is
either stalled, or where current approaches haven't yet found the right sequence to work with.
This is not a verdict. It is a starting point — and the starting point with the most to gain.
The body's terrain is always responsive to the right input, in the right order. Your four
dimension scores below identify the specific coordinates where that leverage lives."

### Overall narrative — MEDIUM (e.g. 60%)
"Your overall score reflects a terrain in active transition — where healing is happening, but
the pattern is incomplete. In the Five Levels of Regulation, this range corresponds to Mixed
terrain: the body is responding, something is working, but something else is still getting in
the way.
This is the most common position on the healing journey — and the most important one to
navigate precisely. Sequence matters enormously here. Your four dimension scores below identify
what to address first, what to sequence next, and what's most likely keeping results inconsistent."

### Overall narrative — HIGH (e.g. 90%)
"Your overall score reflects a terrain with active, coherent healing intelligence — where the
body responds well when given the right support. In the Five Levels of Regulation, this range
corresponds to Positive or Optimum terrain states: regulatory capacity is intact and the system
is capable of genuine recovery.
High overall capacity doesn't mean nothing left to do — it means the work you do will actually
land. Your four dimension scores below identify the specific areas where that capacity can be
directed most productively right now."

## Dimension cards (Category Scores section) — each has low/med/high Dynamic content variants
- Each card: "Your <Dim> Dimension — <subtitle>" + body paragraph(s) + numbered 5-item list
  + "Your score N/10" + band tag (low/med/high). Score shown as N/10 and donut shows % per dim.
- System card subtitle "The Five Layers"; body "The System dimension maps your body's priority
  through our embryological tissue hierarchy — five layers from foundation to integration:"
  1. Flow — Cardiovascular circulation, immune defense, urogenital filtration
  2. Support — Bones, connective tissue, and muscles — the architecture and movement of the body
  (3-5 + Phase/Concern/Regulation cards = REMAINING verbatim extraction)

## CTA section "Call to Action 16" = 3 image-based Items (image + link). Links TBD (confirm
   desired destinations w/ Glen — full cutover should drive into illtowell.com /begin funnel).

## Band tiers: overall 20%=low(red) 60%=medium(orange) 90%=high(green); per-dim same by %.
   Exact low/med/high % cutoffs = ScoreApp tier config (Settings) — extract during build
   (appears ~3 equal-ish bands by percent).

## Existing data: "First 10 leads complete" — ~10 real ScoreApp leads already exist; they have
   already flowed to GHL via /webhook/scoreapp. No lead migration needed; just stop new external
   leads after cutover.

## RESULT PAGE FULL COPY (scorecard 59912, result page 870107)

### Dimension card SUBTITLES + constant lists (same across bands)
**System — "The Five Layers"**: "The System dimension maps your body's priority through our
embryological tissue hierarchy — five layers from foundation to integration:"
1. Flow — Cardiovascular circulation, immune defense, urogenital filtration
2. Support — Bones, connective tissue, and muscles — the architecture and movement of the body
3. Process — Digestive and respiratory systems — transformation, absorption, and release
4. Communicate — Endocrine and nervous systems — the body's information networks
5. Integrity — Skin — the body's living interface, boundaries, and energetic coherence

**Phase — (Five Phases framework)**: refs "Phase 1 (Energize)", "Phase 2 (Rejuvenate)" ...
**Concern — "The Five Levels"**: "In our Clinical Theory, the Concern dimension maps five levels
of where you are starting from on your healing journey — from the most acute and critical all
the way to the preventive and family-pattern level:"
1. Prognosis & Degeneration — Facing a serious, rapidly progressing, or life-threatening challenge
2. Illness & Response — Managing active disease and monitoring how your body is responding
3. Past Health & Treatment — Wo... (capture full)
**Regulation — (Five Levels of Regulation)**: refs Blocked/Negative/Mixed/Positive/Optimum.

### LOW band (default) card interpretations
- System (Layer 1-2): "Based on your response, your body's highest system priority is at Layer
  1 or 2... Your body's priority right now is in the foundational layers — the systems that move,
  circulate, and structure everything else. In the embryological tissue hierarchy, these are the
  Flow and Support layers: the cardiovascular channels that deliver nutrition and remove waste,
  the bones and connective tissue that gi..." (capture full tail). Score tag: low.
- Phase (low): "...challenges. In the Five Phases framework, Phase 1 (Energize) and Phase 2
  (Rejuvenate) represent the body's most fundamental healing work: rebuilding cellular vitality,
  clearing viral and bacterial terrain, and reversing the processes that drive degeneration. This
  is where Biofield Analysis™ delivers its most essential guidance: identifying the exact pattern
  of stressors at this terrain level and matching your body with the tools that open the path
  forward. We will follow up with more specific guidance for your phase." Score 3/10 low.
- Concern (low): (Prognosis & Degeneration level) — capture full.
- Regulation (low): "...urable change when you try, or a pattern where things get harder before
  they improve. In the Five Levels of Regulation, these correspond to Blocked and Negative:
  terrain states where the body's response is either stalled or moving in a direction that creates
  temporary intensification before progress can begin. Neither state means you cannot heal — it
  means the sequence and match of support matter more than ever. Biofield Analysis™ is specifically
  designed for exactly this terrain: reading what the body-field is prioritizing and matching it
  with the tools that can open the door to healing. We will follow up with more specific guidance
  for your level." Score 3/10 low.

### CTA section (constant) — single primary offer (3 benefit "Items" = icons)
"Your Next Step: Find Your Top Stress Sources with a free 10-second Bioenergetic Wellness Scan
We Also Give You Free Online Courses and Community Support
[Free Lifetime Access] [Unlimited Scans] [Instant Online Report]
Button: Claim Your Free Account Now"
Footer: © Copyright 2026 Remedy Match LLC.  (= E4L voice scan / free account = primary CTA.)

### MEDIUM + HIGH band card interpretations — TO TOGGLE-CAPTURE NEXT

## STILL TO CAPTURE (implementation-phase, mechanical)
- Remaining dimension-card verbatim copy (3 of 4 cards × 3 bands) + System card layers 3-5
- CTA item links + exact tier % cutoffs + PDF report (if used in funnel)
- Dimension cards (System/Phase/Concern/Regulation) full copy + any band variants
  (System card header: "Your System Dimension — The Five Layers"; "The System dimension maps
   your body's priority through our embryological tissue hierarchy — five layers from
   foundation to integration:" ... Your score N/10 + band tag)
- CTA 3 items copy + links
- Landing Pages full copy (have intro hook already)
- PDF Reports (if used)
- Integrate (webhook to illtowell /webhook/scoreapp — confirm payload) — code already known
