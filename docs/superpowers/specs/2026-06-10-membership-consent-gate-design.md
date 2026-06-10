# Membership & Consent Gate — Design Spec

**Date:** 2026-06-10
**Status:** Approved design, ready for implementation planning
**Scope:** Tiers 0–1 in v1; Tier 2 (Wholesale) defined here, deferred to Phase 2

---

## Problem

The funnel's chat and ordering are wide open. Anyone can get individualized health advice and place orders without ever agreeing to Terms of Service. We need a Terms-of-Service agreement to stand between visitors and the things that carry liability — **individualized health advice** and **condition-specific recommendations** — and between visitors and **ordering**. The agreement is the legal shield.

The constraint that shapes everything: the gate must be **as soft as possible**. Chat stays open as the on-ramp. The agreement is a single checkbox plus first name, last name, and email — no email-verification round-trip required to get in. We want people to start chatting, get value from general education, and then be invited to opt in exactly when their question crosses into territory the agreement protects.

## Goals

- Let anyone chat immediately (general education), no gate.
- Require a ToS agreement before the chat gives **individualized advice** or **any recommendation tied to a health condition**.
- Require membership (ToS agreed + identified) before **retail ordering**.
- Require explicit approval before **wholesale ordering** (application itself requires ToS).
- Make returning members recognized with zero friction — same device automatically, new device via an optional magic-link login.
- Reuse existing primitives wherever possible; avoid building a new auth system.

## Non-Goals

- No hard email verification as a precondition to becoming a Member. The soft opt-in (name + email + checkbox) is sufficient; payment and the optional magic-link provide stronger identity where it matters.
- No paywall on general educational chat. Education is always open.
- Wholesale pricing, application form, and approval queue are **out of scope for v1** (Phase 2). This spec defines the model so nothing is lost, but the first implementation plan covers Tiers 0–1 only.

---

## The Membership Model

Three tiers, distinguished by a single consent fact plus (for wholesale) an approval flag.

### Tier 0 — Visitor (no ToS agreement)
- **General education chat only**: how conditions/ingredients/remedies work, the published science, what a product factually contains and does. Descriptive, not directive.
- **Blocked**: individualized advice, condition recommendations, retail ordering, wholesale ordering.

### Tier 1 — Member (soft opt-in: first name + last name + email + ToS checkbox)
- **Unlocks** individualized advice and condition recommendations in the general chat.
- **Unlocks** the retail ordering chat and retail checkout — must be identified as a Member to engage.
- **Recognized on return** with zero friction: `amg_session` cookie + localStorage on the same device; optional magic-link to log in from a new device.

### Tier 2 — Wholesale (application → approval; **Phase 2**)
- **Unlocks** wholesale ordering.
- Gated by an explicit `wholesale_approved` flag, not just consent.
- The wholesale application itself requires ToS agreement (so wholesale applicants are already at least Tier 1).
- Separate apply-and-approve workflow + wholesale catalog/pricing. Defined here, built later.

---

## The Live Rule: Education vs. Recommendation

This is the heart of the chat behavior and the legal shield. The chat classifies every individual-facing turn into one of two buckets, and its allowed depth depends on the asker's tier.

**Open to everyone (education):**
- How a condition works ("how do floaters form?").
- What an ingredient or remedy does ("what does astaxanthin do?").
- The published science ("what does the research say about lutein and macular health?").
- What a product factually contains or is ("what's in Retina Renew?").

**Gated behind the agreement (recommendation OR individualization):**
- Anything about the user's own body, symptoms, or situation ("I have floaters, what should *I* take?").
- **Any recommendation tied to a health condition**, even phrased generally ("for floaters, use/take X," "this remedy helps with macular degeneration").

**Behavior when a Visitor (Tier 0) asks a gated question:**
1. Give the **general educational frame** that *is* allowed ("Here's how floaters generally form and the category of nutritional support associated with eye health…").
2. Stop before naming a specific remedy-for-their-condition or any individualized direction.
3. Append a warm, one-line **invitation to opt in**: e.g., *"To get specific guidance for your situation, just add your name and check the box — takes about 10 seconds."*

**Behavior when a Member (Tier 1+) asks the same question:**
- Full individualized depth + condition recommendations (the existing clinical-qa layer), no disclaimer wall.

The classification is something the model already does implicitly when choosing answer depth. The change is to make it **explicit and consent-aware**: feed the answer policy a `member` boolean (derived from `tos_agreed_at`) and the education-vs-recommendation rule, and have it choose between "educate + invite" and "full individualized answer."

---

## Architecture

### Source of truth for consent
`journey_state.tos_agreed_at` (already in the schema) is the single source of truth for Member status, stamped with the current `tos_version` at agreement time. A visitor is a **Member** iff a `journey_state` row reachable from their identity (session cookie, or email when known/authenticated) has a non-null `tos_agreed_at`. Membership is unioned across rows by email exactly as gates are unioned today, giving cross-session and cross-device continuity once email is known.

### Identity resolution (unchanged, reused)
Order of precedence for "who is this":
1. Authenticated user via `amg_auth` cookie → `users` table (magic-link session).
2. Email known from `journey_state` keyed by `amg_session`.
3. Anonymous `amg_session` cookie only.

Member status is computed from whichever identity resolves.

### Components

**1. Consent-aware answer policy (new logic, in the chat path).**
- Inputs: the user's query, the resolved identity, and `member: bool`.
- For an individual-facing query from a non-Member, instruct the model to produce the educational frame + opt-in invitation and to withhold condition recommendations / individualized direction.
- For a Member, allow full depth (existing clinical-qa behavior).
- Pure-ish policy: it changes the system instruction / answer-mode selection based on the `member` flag and the education-vs-recommendation rule. No model retraining; a prompt + flag.

**2. Soft opt-in surface (extend existing chat opt-in bar).**
- The chat already has an opt-in bar with name/email fields. Add: first-name / last-name split and a **ToS-agreement checkbox** that links to the Terms.
- Submit → POST to the existing `/begin/unlock` path with `trigger="tos"`, writing `tos_agreed_at` + `tos_version` + names + email to `journey_state` via `record_unlock`.
- On success, the individualized tier unlocks **in the same conversation** — no page reload, no email round-trip. The invitation appears contextually (right when a gated question is asked), not as an upfront wall.

**3. Gate enforcement at endpoints.**
- **General chat** (`/chat`, `/begin/match/chat`): never blocked; the *answer policy* adjusts depth by Member status. This is a soft gate (content), not an access gate.
- **Retail ordering chat + checkout** (`/begin/checkout/<slug>` and the ordering/matcher entry to buy): hard gate — require Member status before engaging. A non-Member hitting these is routed to the soft opt-in first.
- **Wholesale ordering** (Phase 2): require `wholesale_approved`.

**4. Frictionless return.**
- **Same device:** the 365-day `amg_session` cookie + localStorage name/email already recognize the visitor; Member status persists silently, no re-prompt.
- **New device:** offer the existing magic-link ("log in again and we'll know you") as the **optional** bridge — `/auth/magic-link/request` → `/auth/magic-link/verify` → `amg_auth` session. Never a wall; it's the convenience path to carry membership across devices.

### Data flow

```
Visitor lands → amg_session minted (existing)
   │
   ├─ asks general/education question ──────────────► answered fully (Tier 0 OK)
   │
   ├─ asks about THEM or a condition rec ───────────► educational frame + opt-in invite
   │        │
   │        └─ submits name + email + ToS checkbox ─► record_unlock(trigger="tos")
   │                                                   tos_agreed_at + tos_version stamped
   │                                                   → now Member (Tier 1)
   │                                                   → same conversation re-answers w/ full depth
   │
   ├─ engages retail ordering chat / checkout ──────► requires Member; else route to opt-in
   │
   └─ returns later
         ├─ same device → cookie/localStorage → recognized, still Member
         └─ new device  → optional magic-link → amg_auth → membership carried across
```

---

## External Dependency (not code)

The ToS checkbox must link to **actual Terms** whose text contains the **health-advice disclaimer** doing the legal work. The code already supports a `tos_version` stamp and references an external terms URL, but the **content** of those Terms (disclaimer language covering individualized advice and condition recommendations) is a Glen/legal deliverable. The build is ready for it; the words are not yet written. The implementation should treat the Terms URL + `tos_version` as configuration so the legal copy can land independently.

---

## Reused vs. New

**Reused (already in the stack):**
- `users` table, magic-link auth (`amg_auth`, 30-day sessions).
- `journey_state` table with `tos_agreed_at` / `tos_version`, `record_unlock`, and email-unioned gate continuity.
- The chat opt-in bar (name/email fields).
- Chat answer-depth layers (brief-summary default vs. clinical-qa override).
- GHL onboarding fired on the existing `trigger="tos"` unlock path.

**New (to build, Tiers 0–1):**
- The consent-aware answer policy: the explicit education-vs-recommendation rule keyed on the `member` flag.
- First/last-name split + ToS checkbox on the opt-in bar, linking to Terms.
- Member-status enforcement on the retail ordering chat + checkout endpoints.
- A clean `is_member(identity)` helper deriving Member status from `tos_agreed_at` across the resolution order.

**Deferred (Phase 2, Tier 2 Wholesale):**
- Wholesale application form (ToS-gated) + approval queue + `wholesale_approved` flag.
- Wholesale catalog/pricing and wholesale ordering gate.

---

## Success Criteria

1. A brand-new visitor can ask "how do floaters form?" and get a full educational answer with no gate.
2. The same visitor asking "I have floaters, what should I take?" gets the educational frame plus a one-line opt-in invitation — and no specific condition recommendation or individualized direction.
3. Submitting first name + last name + email + ToS checkbox stamps `tos_agreed_at`/`tos_version`, and the **same conversation** immediately answers the gated question with full individualized depth.
4. A non-Member cannot engage the retail ordering chat or reach checkout without first opting in.
5. A returning Member on the same device is recognized automatically (no re-prompt); on a new device, the magic-link carries their membership over.
6. The ToS version is configurable, so updated Terms re-stamp on next agreement without code changes.

## Open Questions / Risks

- **Classification accuracy:** the education-vs-recommendation boundary is enforced by the model's judgment. Edge cases ("is naming an ingredient that helps a condition a recommendation?") need a clear, testable rule in the policy prompt and a handful of canonical examples. Worth a small eval set against `chat_log.db`-style queries.
- **Unverified soft opt-in:** anyone can type any name/email to become a Member. Acceptable for retail (payment verifies; magic-link strengthens) but means Member status alone is weak identity. Wholesale's explicit approval covers the case where stronger identity matters.
- **Streaming gate:** the chat streams answers; the consent decision must be made **before** the stream starts so a gated answer never partially leaks individualized content.
