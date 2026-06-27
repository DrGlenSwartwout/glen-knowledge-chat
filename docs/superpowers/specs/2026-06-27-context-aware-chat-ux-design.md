# Context-Aware Chat UX — Design Spec

**Date:** 2026-06-27
**Status:** Approved design (brainstormed + live-validated), pending implementation plan
**Depends on:** PR #360 (chat abuse / IP-protection rate limiting) — the depth gate. Sequence this **after** #360 merges; the Engaged→email rung reuses `/full-report` + the gate.
**Author:** Glen + Claude

## Problem & motivation

Prod `query_log` analysis (199 rows, entire history): **full-answer usage is 0.5%** — one full request ever, ~1,066 words. The brief answer today is an "executive summary" that **resolves** the question and **never teases the full**, so there is no reason to go deeper. Meanwhile a full answer costs ~**1¢** to produce (Haiku 4.5; ~$1–1.50 per 100K words), and it is simultaneously the best value-demonstration and a lead-capture moment (the depth gate turns it into an email). The economics say *manufacture more full answers*, not ration them.

~**99% of traffic is the funnel** (multi-turn Socratic journey), not one-shot Q&A. Users type tiny replies ("Yes", "None", "Agree", "ED1"). The interface shows the union of every control everywhere, so no single next action stands out.

**Goal:** turn the brief into a deliberate **lead** that earns the next step, match that step to the reader's readiness (so we never over-charge cold traffic with an email ask), and simplify the interface per surface. Drive full-answer / page / scan engagement; capture leads only where personalization justifies it.

## Framing: the brief IS the lead (Jon Benson terms)

The ~200-word brief is not "a short answer that teases" — it **is the lead** of a longer arc, and a lead's only job is to **hook the reader into the next step (the CTA)**. The arc:

> **Brief (lead) → CTA (micro-commitment) → Full report / page / action (body & close)**

The brief's success metric is therefore **"did it earn the click,"** not "did it answer." The full-answer long-form (existing Break & Rebuild arc, `app.py:1239`) is the body/close that pays off the lead.

## Component 1 — The brief as a 5-beat "include-and-transcend" lead

Replace the brief synth instruction (`app.py:3075`; system DEFAULT FORMAT block `app.py:1223`) with a 5-beat structure. Plain, warm, everyday language; **imply** the deeper level, never name clinical dimensions/jargon in the brief. ~200 words. Beat labels are NOT printed (read as natural prose — same rule as the existing no-"Hook"-label rule).

| Beat | Content | Benson lead function |
|---|---|---|
| 1. **Consensus** | State the mainstream answer plainly & generously (no strawman). Credit by name what the user/doctor is already doing. | Pattern interrupt (unexpected *agreement*) + Yes-Reflex |
| 2. **Why it works** | Affirm what the consensus gets right, explained one notch deeper than the consensus states it. | Agreement-stacking + authority (explain it better than they can) |
| 3. **Limitation** | Where it plateaus/stops short — use the user's own words if they reported being stuck. | Validation turn ("not your fault — the framework has a ceiling") |
| 4. **The assumption** (THE HOOK) | Name, in plain language, the single hidden assumption beneath the consensus that causes the limitation — and STOP. Do not reveal what's true instead. | "Tease one thing in particular" + the open loop |
| 5. **Tease + readiness-matched next step** | Point past the assumption, then offer EXACTLY ONE next step matched to readiness (Component 2). | First micro-commitment / "Let's Make a Deal," benefit-framed |

**Coupling (mandatory):** beat 4 names an assumption; the **full report must break exactly that assumption.** Brief and full are written as a pair — beat 4 is the full's thesis. The existing full Break & Rebuild arc's "name & steelman the false belief" step IS beat 4; nudge the full instruction to break the central assumption a brief would credit-then-question. Never pre-empt the full's reveal in the brief.

**Hard constraints (carry forward all existing rules):**
- ~200 words; beats 1–3 must deliver a genuine, act-on-able quick win on their own (a reader who never clicks still got real help).
- **SAFETY OVERRIDE:** never withhold safety-critical / time-sensitive info behind the loop (drug–nutrient interaction, red-flag symptom, urgent contraindication) — state it plainly in the brief. Tease optimization & depth, never safety.
- No "Hook" label; formulation-first ordering; product links only from the injection table; Speckhart boundary (credential = authority, never a disease-cure claim); active discount-code rule; Sources line.
- **NOT EVERY TURN:** if the turn is a clarifying question, greeting, name/consent capture, or logistics, do NOT force the 5 beats — respond naturally. (Validated: a context-free "let's get started" correctly clarified instead of forcing structure.)
- **Tuning note (from live test):** beat 5 must *deliver* the matched offer in prose, not merely classify it; hold ~200 words (a live run drifted to 232).

## Component 2 — Readiness triage on beat 5 (inline rubric, 4 rungs)

Beat 5's CTA is chosen by the model inline (a rubric in the synth instruction — no separate classifier in v1), using signals it already sees: the question's intent, the retrieved RAG sources, `journey_state` + turn count, the identity tier (`_resolve_chat_tier` from #360), and explicit cue words.

**The decisive rule — generic vs personal depth:** if a retrieved source page already answers the deeper question *for anyone*, the depth is **generic → link the page**. If the real answer requires synthesizing *this person's* specifics (labs, history, scan, several named factors at once), it is **personal → email the report**.

| Rung | Signals | Matched next step | Goal |
|---|---|---|---|
| **Curious** (cold) | informational/"what is/should", early turn, anonymous, no personal stakes | **LINK** the page that goes deeper (RAG source URL / product page). **No email ask.** | familiarity, ecosystem entry |
| **Engaged** (warm) | named their own situation w/ specifics, plateaued, gave name/consent | **EMAIL** the personalized report (reuses #360 depth gate) | lead capture, justified by personalization |
| **Ready** (hot) | explicit intent ("let's start"), asked for the recommendation, scan done | **DIRECT ACTION** — open recommended product / start E4L scan | conversion |
| **Committed** | logged-in member | full answer **INLINE**, no gate; offer "save to my portal" | depth + retention |

**Rules:** prefer the page link over the email ask whenever generic depth exists for a cold reader (never ask for an email for something a page already answers); when unsure between two rungs choose the **lower-friction** one (link < email < action); never skip a rung; a page link is not a dead end (the page carries the next rung).

**IP alignment:** generic content is already public/indexable → link it freely; the personal synthesis is the moat → keep it gated. The triage *strengthens* the IP posture from #360 and stops wasting the email ask on public content.

**Validated live (Haiku):** cold "what foods for AMD?" → page link, no email ask; warm AMD plateau → classified Engaged→email; context-free "let's get started" → clarified (the not-every-turn guard). Rungs fire differently.

## Component 3 — Context-aware interface (per surface)

One **UI profile** computed once = **surface × identity tier**, same pattern as the console's `/api/me` role+nav (`project_console_role_based_nav`). Surface from `widget.js` `data-context` attribute (or route/referer: `/begin/*`=funnel, `/embed`=widget, portal=member). Identity from the existing tier resolver.

| Surface | Strip to | Full-answer lever | Identity tweak |
|---|---|---|---|
| **Funnel / journey** (`/begin/*`) | conversation + **quick-reply chips** + ONE next-step CTA | "email my protocol report" at the *recommendation* turn only | drop capture once email known |
| **Standalone widget** (`widget.js`) | answer + open-loop brief + 1 CTA + inline email | open-loop brief → page or email per triage | anon: gated; member: inline |
| **Member portal** (logged-in) | answer + their report history | full **inline, no gate**; "save to my reports" | no email capture (known) |
| **Concierge** (`/begin/concierge`) | answer + the one relevant offer | the offer is the CTA | buyer-aware |

**Quick-reply chips (funnel):** the bot's clarifying questions render as tap-chips (e.g. "Overactive/hyper" · "Underactive/hypo" · "Not sure" · *or type…*) mapped to the branches the bot already uses. Users currently type one-word replies — chips remove that friction. No full-CTA mid-clarification (wrong moment).

## Instrumentation

Log, per brief: the **rung chosen** and the **CTA type shown** (page / email / action / inline), plus whether it was **clicked/converted**. This is the lead's hook-rate — the real success metric — and the data to tune the rung thresholds (and the placeholder 10k/100k caps from #360) on evidence, not guesses.

## Non-goals / sequencing

- **After #360.** The Engaged rung depends on the merged depth gate + `/full-report`.
- v1 triage is the **inline rubric** only; a structured classifier is a later hardening step if the inline version proves inconsistent.
- The full-answer Break & Rebuild arc already exists; this spec only adds the beat-4/full coupling nudge, not a rewrite.
- Surfacing full more prominently in the UI generally (it's hidden behind buttons nobody clicks) is noted but out of scope here.

## Open items for the plan

- Exact placement: brief synth instruction string vs the system DEFAULT FORMAT block (keep them consistent).
- How beat-5's chosen rung is signaled to the frontend so the right CTA control renders (a structured marker in the SSE stream vs the model emitting the link/text inline).
- Chip schema: how the bot's clarifying branches expose their options to the frontend.
- `widget.js` `data-context` attribute + the UI-profile resolver shape.
