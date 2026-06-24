# Lead-Magnet Quiz Funnel (Design Spec)

**Date:** 2026-06-24
**Status:** Design — pending Glen's review
**Repo:** deploy-chat (illtowell.com funnel)
**Related:** `project_neuro_magnesium_launch` (Founding Protocol = the offer this feeds), `project_jon_benson_video_process`, the acquisition-engine research (AMD psychology/compliance, Jay Abraham host-beneficiary)

---

## 1. Goal

Build the **lead-magnet quiz funnel** — acquisition-engine sub-project #1, the *hub* that fills the Founding 2,500. A quiz-first, AMD-first opt-in that captures an email, personalizes a recommendation, and routes the lead into the just-merged Founding Protocol reserve offer (PR #252). Every other acquisition channel (host-beneficiary intros, webinar, community, email) will drive traffic *to* this hub.

## 2. Scope

**In scope:** the quiz engine (questions + scoring + result), the three pages (`/begin/quiz`, result, lead-magnet landing), email-gating, GHL tagging, the free-guide delivery wiring, and routing to the Founding offer. Built parameterized so future products re-use it.

**Out of scope (dependencies, not code):**
- **The free guide PDF.** Glen has an existing book (the "bigger picture / what comes after the remedy" context). A **new product/ingredient-focused front section** is needed (a parallel BNSN task) — likely prepended as Part 1 of the same book. The funnel only needs the **finished combined PDF on R2**; the spec leaves a clean slot (`LEAD_MAGNET_PDF_KEY`) and degrades gracefully if unset.
- The other acquisition channels (host-beneficiary, webinar, community) — separate specs.
- The GHL nurture sequence content (configured in GHL UI; the funnel just tags + onboards).

## 3. The quiz — "Your Foundational Eye-and-Brain Self-Assessment" (~60 sec)

### Promise / hook
Lead with a **BT-tested headline** in the compliant "speak to the feeling, not the disease" frame. Default (A/B-able; final drawn from the Boulder-Test winners):
> *"Have you been told your vision changes are just aging — as if there's nothing you can do? There's now foundational support that actually reaches your eyes and brain. Take 60 seconds to see if it's for you."*
Alternates to test: *"Are your eyes and brain missing something essential — even though you supplement?"*

**Compliance:** the hook speaks to the reader's lived experience ("been told it's just aging"); the PRODUCT never claims to treat/prevent/slow/reverse any disease. No disease nouns (AMD/macular/glaucoma) as the thing the product acts on. Structure-function only; DSHEA disclaimer on the result; founder story biography-only.

### Question set (9 questions + email gate — all structure-function / lifestyle, never diagnostic)
1. **Segment (what brought you here):** doctor told me to "monitor / watch and wait" · I'm noticing focus/clarity changes · vision runs in my family, I want to be proactive · I supplement but wonder if I'm missing something · general foundational health
2. **Stress & sleep:** restful · occasional restlessness · frequent tension/poor sleep
3. **Muscle tension/cramps/twitches:** rarely · sometimes · often
4. **Mental clarity / focus:** sharp · occasional fog · frequent fog
5. **Screen hours/day:** under 2 · 2–6 · 6+
6. **Night-driving / low-light comfort:** comfortable · some difficulty · avoid it
7. **Magnesium-rich foods daily (greens/nuts/seeds):** yes · sometimes · rarely
8. **Currently supplementing:** magnesium · an eye formula (AREDS-type) · both · none
9. **Proactive about long-term eye+brain health:** yes · somewhat
→ **Email gate:** name + email reveals the result **and** unlocks the free guide.

### Result logic
Score the **magnesium-depletion + foundational-gap** signals (Q2–Q8) into a simple "foundational profile," and personalize the *reasoning* from the top 1–2 answers — always landing on the Neuro Magnesium Founding offer (one founding product today; engine branches when product #2 exists):
- "watch-and-wait" / family / takes an AREDS formula but no barrier-crossing magnesium (Q1/Q8) → emphasize *"reaches the blood-brain **and** blood-eye barrier where ordinary magnesium can't."*
- stress/sleep/tension (Q2/Q3) → *"calm, without the fog."*
- fog/focus (Q4) → *"a clear, steady mind."*
- screens/night-driving (Q5/Q6) → *"foundational support for eyes that work hard."*

Result copy speaks to foundational support + their lifestyle, never "you have AMD" / "this treats." It surfaces the **Founding offer card** (only when `founding.is_open`) with a personalized one-liner + the founding scarcity ("X of 2,500").

## 4. Funnel flow + architecture (mostly reuse — confirmed by recon)

```
/begin/quiz (new page)
  → answer (POST /begin/quiz/answer → quiz_responses)
  → EMAIL GATE: reuse existing POST /begin/unlock (trigger="email") → journey_state + journey_events
       + automatic background GHL onboarding (ghl_onboard_contact)
  → /begin/quiz/result (new page; GET /begin/quiz/result-data computes profile)
       + GHL tag by segment: ghl_update_tags(email, add=["lead-magnet","quiz-completed", f"awareness:{segment}"])
       + free-guide delivery: existing magic-link + R2 signed-URL pattern (LEAD_MAGNET_PDF_KEY)
       + surface Founding offer card (existing begin_funnel.surface() — add a "founding-offer" card, gated on founding.is_open)
  → /begin/product/<neuro-mag-slug> (existing) → POST /begin/founding/reserve (existing, PR #252)
```

**Reuse (per recon, file refs in the recon report):** email capture `/begin/unlock`; `journey_state`/`journey_events` + rungs (`arrival→free_tier→assess→ascend`; "quiz" is already an accepted trigger); GHL client (`ghl_upsert_contact`/`ghl_update_tags`/`ghl_onboard_contact`); magic-link email + R2 file serve; `/begin/*` static-page + JSON-data-endpoint pattern; `surface()` CTA system; the Founding reserve flow.

**Net-new (the only greenfield):** the quiz engine.

## 5. Data model (net-new)
- `quiz_questions` — `id, quiz_id, ordinal, prompt, type, options_json` (or ship questions as a versioned JSON config in `data/quizzes.json`, mirroring `data/founding_launches.json` — **preferred**, since the quiz is content that changes without a migration).
- `quiz_responses` — `id, session_id, email, quiz_id, answers_json, segment, created_at`.
Counter/scoring derived from `answers_json` at result time (no cached score). Migrations idempotent `migrate_add_*` per the repo pattern.

## 6. Endpoints (net-new; follow the `/begin/*` page + JSON-data pattern)
- `GET /begin/quiz` → `static/begin-quiz.html` (multi-step quiz UI).
- `GET /begin/quiz-data` → quiz config JSON (from `data/quizzes.json`).
- `POST /begin/quiz/answer` → store answers to `quiz_responses` (session-scoped).
- `POST /begin/quiz/opt-in` → thin wrapper over existing `/begin/unlock` (trigger="email") + tag + mint guide link.
- `GET /begin/quiz/result` → `static/begin-quiz-result.html`; `GET /begin/quiz/result-data` computes the profile + the personalized founding card.
- `GET /begin/quiz/guide` → gated guide download (magic-link token → R2 signed URL; graceful "guide coming" if `LEAD_MAGNET_PDF_KEY` unset).
- Add a `founding-offer` card to `begin_funnel.surface()` (gated on `founding.is_open`).

## 7. Compliance (hard requirements)
- Structure-function only on every screen; never imply the product treats/prevents/slows/reverses AMD/macular degeneration/glaucoma; no disease nouns as what the product acts on. The hook may name the reader's *experience* ("told it's just aging"), not a product disease claim.
- DSHEA disclaimer on the result page; founder story biography-only; no "you will reverse too."
- Reuse the funnel's existing compliance denylist/guardrails. Real apostrophes; no em-dashes / ALL-CAPS shouting in body copy.
- Email/opt-in honors existing ToS handling; nurture auto-renewal copy (if any) follows ROSCA like the Founding offer.

## 8. Repeatability
Quiz config + the result→product mapping are data (`data/quizzes.json`), and the founding card is gated by `founding.is_open(slug)` — so a future product's launch re-uses the same engine with a new quiz config + result mapping. Pairs with the per-product Founding Protocol machine and the HyperFrames promo pipeline.

## 9. Success criteria
- A visitor can take the quiz, opt in (email captured to `journey_state` + GHL onboarded + tagged by segment), see a personalized result, download the guide (when the PDF is set), and click through into `/begin/founding/reserve`.
- Result copy is structure-function compliant (zero disease claims); founding card only shows when the launch is open.
- Quiz content lives in `data/quizzes.json` (editable without a migration); the engine is product-agnostic.
- Graceful when `LEAD_MAGNET_PDF_KEY` is unset (no broken download).

## 10. Deferred to the implementation plan
- Quiz UI step/progress pattern (single-page vs step routes); exact scoring weights/thresholds for the profile bands.
- Whether opt-in sets ToS/free-tier (full member) or email-only (lighter) — and the corresponding GHL pipeline/workflow.
- Guide delivery: magic-link-gated download vs direct R2 signed URL in the email.
- Slicing (config+schema → quiz pages/answer → opt-in+result+routing → guide delivery).
