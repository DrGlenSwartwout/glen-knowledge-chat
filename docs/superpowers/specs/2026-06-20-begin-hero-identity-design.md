# Begin Page #1 — Hero + conversational identity capture

**Date:** 2026-06-20
**Status:** Approved (design); ready for implementation plan
**Repo:** deploy-chat (Flask, Render `glen-knowledge-chat`)
**Parent:** Begin-page redesign (5 sub-projects). This is **#1, the spine** — the new front door + the one-record identity capture that #2 (sidebar cards), #3 (secondary entry points), #4 (Match/ordering), and #5 (Ascend) all hang off.

---

## Problem

Today's `/begin` page is a video + a large block of cards + an email box — a choice-heavy front door. We want a clean, welcoming front door where **chat is the hero**: a headline, a short sub, the intro video, and a simplified self-healing chat beside it (desktop, above the fold) or below it (mobile). The visitor can talk immediately; the AI asks their **name** conversationally in its first reply, and asks for their **email** at the moment they're ready to activate free membership. Everything writes to **one identity record**. The "Explore everything" link moves to the bottom, gated on membership.

## Scope (#1)

Rework the **top of `static/begin.html`** into the hero (headline + sub + video + simplified chat, responsive) wired to the existing `/begin/match/chat`; a **scripted first AI message that asks the visitor's name**, captured conversationally to the identity record; an **email-to-activate** moment that sets Tier-1 membership; the **Explore link moved to the bottom**, membership-gated. All identity writes route through the existing `/begin/unlock` → `begin_funnel.record_unlock` → `journey_state` backbone (unioned by session + email = one record). 

**Out of scope (later sub-projects):** the 6 sidebar cards / status coloring (#2 — #1 leaves a clean reveal hook for them); ScoreApp/E4L secondary entry points (#3); Match/ordering integration (#4); Ascend (#5). #1 keeps (or stubs) the existing cards block below the hero unchanged so nothing regresses while #2 is built.

---

## Confirmed decisions (Glen, 2026-06-20)

- **Chat opens immediately; the AI asks the name in its first reply** (lowest top-of-funnel friction). Name captured conversationally, no form.
- **Email at the activation moment** (after value) to activate free membership.
- **One human, one record** — a hard requirement; all capture routes through `/begin/unlock`/`journey_state`.
- **Simplified chat** = self-healing level only, NOT labelled, **no Rate feature, no Leave/View-feedback** controls.
- **Layout:** headline (+ short sub) above the video; chat **beside** the video on desktop (above the fold), **below** it on mobile.
- **Explore-everything link → bottom of the page, gated on membership** (members only).

---

## Architecture

### Hero layout (`static/begin.html`, top of page)
A new hero block: **headline** ("Welcome[, {first name}] — let's talk about your health goals", personalized once the name is known via the existing `personalize()`), an optional one-line **sub**, the existing intro **video**, and a **chat panel**. Responsive: a two-column (video | chat) grid above the fold on desktop; stacked (video, then chat) on mobile. Visually clean, minimal chrome. NO emoji; no em dashes.

### Chat (reuse `/begin/match/chat`)
The hero chat posts to the existing **`/begin/match/chat`** (the self-healing remedy conversation), rendered as a **stripped-down** transcript: message bubbles + an input, and **none** of the Rate / Leave-feedback / View-feedback controls (those are removed from this surface only). The chat **opens with a scripted first AI message** that welcomes them and asks their name ("Aloha! I'm here to help with your health goals. So I can tailor this and remember you — what should I call you?"). This first message is a fixed client-side greeting (not a model call), so it's instant and reliably asks the name.

### Conversational name capture (one record)
Because the AI's first message asks the name, the visitor's **first reply is treated as the name**: the client lightly cleans it (strip leading "I'm ", "my name is ", "it's ", punctuation; take the first 1–2 tokens) and POSTs it to **`/begin/unlock`** with a `trigger="name"` (a new low-risk trigger in `begin_funnel.VALID_TRIGGERS`) and `first_name`/`name`. `record_unlock` writes it to `journey_state` (keyed by the `amg_session` cookie, unioned with email later). The visitor's reply is ALSO sent to the match chat as their real first message, so the conversation flows. `personalize()` then greets them by name. (Heuristic capture is deliberate: we control the prompt, so the turn right after the name-ask is the name; a wrong capture is correctable and never blocks the chat.)

### Email → activate free membership (one record)
At the **activation moment** — when the AI offers to "save your progress / unlock your free membership" (surfaced by the chat at a natural value point, or a gentle inline prompt after N exchanges) — the UI shows an **inline email field + a one-line ToS opt-in** (reusing the existing soft consent copy). On submit it POSTs `/begin/unlock` with `email`, `tos=true`, `first_name`, `trigger="activate"`. That sets `journey_state.tos_agreed_at` → **`is_member` true** (Tier-1), merging name+email into the **one record** (and carrying any `rm_ref` referral cookie, already handled by `record_unlock`). After activation, membership-gated UI (the Explore link, and later the sidebar cards) unlocks.

### Explore link → bottom, membership-gated
Move the **"Explore everything"** link/button to the **bottom of the page**, below the welcome copy. It is shown/enabled only for members (`is_member` via the existing `/begin/state`), with a soft nudge to activate for non-members. (The full sidebar-card reveal on activation is #2; #1 only relocates + gates the Explore entry point and leaves a hook for #2.)

### Reuse (no new backbone)
- Identity: `/begin/unlock` + `begin_funnel.record_unlock`/`get_state` + `journey_state` (one record, session+email union) — **the backbone already exists**; #1 adds two triggers (`name`, `activate`) to `VALID_TRIGGERS`.
- Membership: `is_member` (ToS/Tier-1) + `/begin/state` for the gate.
- Chat: `/begin/match/chat` (self-healing), rendered stripped-down.
- Personalization: the existing `personalize(first_name)` + `STATE`/reveal system.

---

## Data flow
1. Visitor lands → clean hero (headline + video + chat). Chat's first AI line asks their name.
2. Their first reply → cleaned → `/begin/unlock(trigger="name", first_name=…)` (record created/updated by session) → `personalize()` greets them; the same text drives the match chat.
3. They converse (self-healing chat). At a value moment, the activation prompt appears.
4. They enter email + opt-in → `/begin/unlock(trigger="activate", email, tos=true)` → `tos_agreed_at` set → **member**; name+email now one record.
5. The bottom **Explore** entry unlocks (and #2's sidebar cards will reveal here later).

## Error handling
- Name capture is best-effort: a failed/garbled capture never blocks the chat (the message still goes to the match chat); the name can be re-asked or corrected.
- `/begin/unlock` failures are logged and degrade gracefully (chat keeps working; activation can be retried).
- Non-members: the Explore link shows a soft "activate to explore" nudge instead of an error.
- The existing cards block below the hero is left intact in #1 so nothing regresses before #2.
- NO Rate/feedback controls render on this surface (removed for the hero chat only; other chat surfaces unchanged).

## Testing
- **Identity:** `/begin/unlock` with `trigger="name"` records `first_name` to `journey_state` (by session); `trigger="activate"` with `email`+`tos` sets `tos_agreed_at` → `is_member(session)` / `is_member(email)` true; name+email resolve to **one** state (union by session+email). New triggers accepted by `record_unlock`.
- **Membership gate:** `/begin/state` reports member after activation; the Explore link is gated (member vs non-member).
- **Hero serve:** `/begin` returns 200 with the new hero (headline, video, chat panel present; no Rate/feedback controls in the hero chat; Explore link at the bottom).
- Front-end (chat flow, name-ask, responsive layout, activation prompt) = **manual visual pass** (state it). Server-side identity/membership has unit + Flask-test-client coverage.
- Follow deploy-chat test isolation (tmp `$DATA_DIR`; mock Supabase; importorskip playwright; `importlib.reload`). NO emoji; no em dashes.

## Notes
- This is a **live page** (no flag) — `/begin` is the funnel front door. Ship the hero rework carefully; keep the existing below-hero content intact until #2 replaces it. A visual pass before deploy is required.
- The name-capture heuristic (first reply after the scripted name-ask) is a deliberate, reversible choice; the alternative (a model-emitted structured signal) is a later refinement if mis-captures show up.
- #1 establishes the **one-record** contract every later entry point (#3) must use: always write through `/begin/unlock`/`journey_state`, never a parallel store.
