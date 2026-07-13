# Adaptive Concierge — Design Spec

**Date:** 2026-07-12
**Status:** Design, pending implementation plan
**Surface:** the homepage chat concierge (`/begin/match/chat`) and its activation trigger

## Problem

The homepage concierge runs a largely fixed sequence: greet, capture a name, answer with retrieval, and the frontend fires the email + Terms capture after a set number of turns (`heroMaybeActivate()` in `static/begin.html`). It treats a scared newcomer, a skeptic, a ready buyer, a caregiver, and a returning member the same way. Glen's directive: the concierge should judge each person's desire and readiness and serve them accordingly, so they feel understood and served, not processed through a protocol.

The endpoint already has most of the raw material. `begin_match_chat` (`app.py:4589`) injects, when known: member context, intake summary, recent concerns/queries, and a voice-scan summary (`_member_context_for_email`), a household note when an email is shared, a `for_whom` line (self vs someone-else), and a resolved tier (free vs paid). What is missing is an instruction layer that reads readiness and selects a strategy, a distress-to-human path, and activation that is earned rather than turn-counted.

## Goal

One warm, empathic persona (Dr. Glen's voice) that, every turn: reads who this is and what they need, mirrors it back so they feel heard, and offers a next step sized to their readiness. Different requests get genuinely different strategies. The free-membership activation happens when the person is ready, never as a wall.

## What the concierge reads (signals, then a quiet classification)

Blend three axes into a single read. Do not surface the classification to the user.

1. **Relationship** (mostly already available): brand-new cold visitor, returning free member, paid member, practitioner. From `auth_user`, the resolved tier, `for_whom`, and the presence of member/voice-scan context.
2. **Readiness / awareness:** just curious, problem-aware, comparing/solution-aware, ready. Read from the conversation.
3. **Emotional need:** what they actually came for (relief, hope, proof, reassurance for someone else, continuity, B2B).

## The strategies

| Segment | How it signals | Strategy (how it serves) | Next step it offers |
|---|---|---|---|
| Curious browser | short, vague, "just looking" | Don't gate. One plain line of what this is; let curiosity pull. Keep it light. | The free 10-second scan, framed as "see what your body is asking for" |
| Person in distress | names a symptom, fear, "it's getting worse" | Slow down. Reflect it back, validate, hold space before anything. No pitch. | Smallest helpful step; when distress is high, a warm human/consult handoff (see below) |
| Overwhelmed veteran | "tried everything, nothing works" | Validate the exhaustion, then differentiate: this does not add another thing to fight the body, it finds the substrate the body uses to do its own retracing. "Add, don't replace." | The scan as a different kind of step, personalized not generic |
| Analyst / skeptic | "how does it work," "evidence," "who are you" | Answer honestly first, with mechanism and real (verified) credentials. Never hand-wave. | Low-risk scan after the proof lands |
| Ready buyer | "how do I start," "I want to try this" | Don't over-explain. Get out of the way, reduce friction. | Straight to scan then match then membership |
| Caregiver | "for my mother / son / dog" | Validate the love and the burden; reassure. Use the `for_whom = someone-else` path and do NOT apply the chatter's personal data to the dependent. | Help set it up for their person (dependent/animal flows exist) |
| Returning member | known from auth/tier/context, or "where's my scan / reorder" | Continuity, not cold intake. Greet by name, reference their journey. | Their portal, their current scan, or Order what's working |
| Practitioner | "I'm a practitioner / I have patients" | Switch tracks: B2B value, patient tools, dispensary. | The practitioner path, not consumer intake |

## The two rules that hold it together

1. **Mirror before you move.** Every response reflects what the person actually said before offering anything. This is Glen's lead-with-validation rule applied turn by turn.
2. **Activation is earned, not gated.** Move the email + Terms capture from a turn count to a readiness signal: it appears when the person has been served and is leaning in (asked to start, asked to save progress, gave a name plus a real concern), never as a fixed-turn wall. A skeptic earns it later than a ready buyer.

## The email invitation (right after the name)

The first concrete activation beat is a warm, optional invitation, not a form. Once the person shares their name, the concierge acknowledges it and asks, in its own voice, whether they would like to share their email so it can remember them next visit and stay connected. Rules:

- **Optional, never a gate.** If they decline or skip, the concierge keeps helping with no friction and can offer again later once more value has landed. It never blocks the conversation behind the email.
- **Name the honest value:** "so I can remember you next time, and we can stay connected." This doubles as the consent to contact them; it carries the same email + Terms agreement the current flow captures, framed as a relationship rather than a signup.
- **Timing:** right after the name is the natural low-pressure moment (one small share leads to another). For a visitor in distress or mid-question, the concierge finishes serving that first and invites the email once the person is settled.
- Reuses the existing capture mechanism (`#hero-activate` + `unlock('tos', {email, tos})` in `heroChat`), retriggered by this readiness beat instead of a turn count.

## The distress to human handoff (confirmed: enabled)

When the read is real, acute distress, the concierge may offer a warm human path (a consult or a call), gently and without pressure, in addition to the self-serve scan. Guardrails:

- It is an offer, never automatic, and never framed as urgent medical triage.
- Carry a plain non-emergency line: for anything that feels like an emergency, contact local emergency services or a physician. The concierge does not diagnose or manage emergencies.
- Route to the existing consult/booking surface (to be named in the plan), not a new channel.

## Signals passed to the concierge (confirmed: yes)

Keep leveraging what already flows in (member context, voice-scan summary, `for_whom`, household note, tier), and make the relationship/readiness read explicit in the prompt so returning members and caregivers get true continuity rather than cold intake. No new personal data leaves the device or the server boundary already in place; this is about using signals already present.

## Where this lives

- **Primary:** the system prompt assembled in `begin_match_chat` (`app.py:4589+`). The read-model, the segment strategies, the mirror-first rule, the distress-offer, and the earned-activation cue become prompt instructions layered on top of the existing retrieval context blocks.
- **Activation trigger:** `heroMaybeActivate()` and the turn-count logic in `static/begin.html` move from "after N turns" to "when the concierge signals readiness" (the endpoint can emit a lightweight readiness flag in its SSE stream that the frontend reveals the activation on).
- The persona and voice sit on top of, not replacing, the retrieval and personal-context machinery already in the endpoint.

## Guardrails

- **Claims discipline:** no disease-cure claims; the formula is "the substrate the body uses to do its own retracing"; trajectory-plasticity language, not "will cure"; consumer copy stays compliant.
- **Voice:** Glen's, always. No em dashes, no ALL CAPS, no AI-pleasantry filler, no emojis. Lead with validation.
- **Credentials:** verified anchors only (the authority-guardian library); never invent one.
- **Safety:** the distress path never becomes medical triage; carry the non-emergency line; escalate to a human, do not simulate one.
- **Do not over-collect:** activation is earned; never hard-gate the conversation behind an email wall.

## Out of scope (follow-ups)

- The homepage visual re-skin (separate plan: `2026-07-12-homepage-hybrid-optionA.md`).
- Any change to the retrieval namespaces, tiering, or velocity guard already in the endpoint.
- A full practitioner B2B conversation flow beyond routing to the practitioner track.

## Open questions for the plan

- The exact consult/booking destination for the distress handoff (name the route/flow).
- The shape of the readiness flag the endpoint emits for earned activation (a field on an SSE event vs a separate signal).
- Whether readiness classification is purely prompt-driven or wants a lightweight structured pass before the answer (cost vs. reliability).
