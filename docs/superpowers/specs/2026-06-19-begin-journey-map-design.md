# Begin Page #2 — The 4-card Journey Map (unfolding path)

**Date:** 2026-06-19
**Status:** Approved (design); ready for implementation plan
**Repo:** deploy-chat (Flask, Render `glen-knowledge-chat`, illtowell.com)
**Parent:** Begin-page redesign (5 sub-projects). This is **#2**, building on the clean hero shipped in #1 (`docs/superpowers/specs/2026-06-20-begin-hero-identity-design.md`, PR #191, live). #1 left a deliberate hook below the hero for this strip.

---

## Problem

#1 gave `/begin` a clean hero (video | chat) and one-record identity capture. The page now needs its **navigation / map**: a small, fixed set of journey steps the visitor can see and move through. Glen's "Create Your Own Healing Adventure" north-star wants this to read as a **path that draws itself** as the visitor engages, not a static menu dropped on the page. So the 4 steps **unfold left to right** the moment the visitor presses play or the conversation gets going, arrive **in-conversation with a reason** (context, meaning, benefit), and then **color by progress** so the visitor feels themselves advancing.

## Scope (#2)

A fixed **4-card journey strip** below the hero in `static/begin.html`, rendered from a new `begin_funnel.journey_map(state, ref)` injected into `/begin/state`. The strip is hidden until an **unfold** is triggered by whichever comes first — the first AI answer completing in the hero chat, or the intro video being played — then animates in left to right. After unfolding, each card shows **progress status** (done / next / available) derived from `journey_state` gates; clicking a card navigates to its destination AND advances the map by firing that step's existing gate. Existing contextual surfacing (`surface()`/`surfaced_cards`/`/begin/explore`) is untouched.

**Out of scope (later sub-projects):** secondary entry points writing to one record (#3); Match/ordering + Biofield interpretation (#4); Ascend high-ticket (#5). Also out: a real-video timecode cue (a left-in hook only; today's video is a placeholder); a BNSN-driven copy pass (all card/caption copy here is provisional and will be refined site-wide later).

---

## Confirmed decisions (Glen, 2026-06-19)

- **Four cards, in journey order:** Scan (Your Biofield), Find (Your Remedy Match), Heal (the root causes), Earn (Ambassador). (Consolidated from the original 13-card list; Quiz / Tone / Work / Link / Find-a-Practitioner / Form benched to the Explore page.)
- **Placement:** a horizontal strip directly below the hero; the hero (video | chat) stays clean above the fold. Responsive: 4-across desktop, 2x2 tablet, stacked mobile.
- **Status = progress map:** done / your-next-step (gold) / available. Felt progress, no hard locks. All 4 always visible (once unfolded); never gated out of view.
- **Unfold = the path draws itself left to right**, one card at a time with a stagger. Unfolds once.
- **Trigger = whichever fires first:** (a) the first AI answer completing in the hero chat, or (b) the intro video being played.
- **Chat-cue unfold weaves the journey into the conversation:** when the chat triggers it, the page injects a short assistant-voiced framing line giving the journey context/meaning/benefit, in sync with the unfold. It must arrive as part of the conversation, not as a bare menu.
- **Heal parenthetical = "the root causes"** (provisional; whole-site copy pass via BNSN later).
- **Live page, no feature flag** (same as #1); a manual visual pass is required before it is considered launched.
- **No new gates / no schema change** — reuse existing `VALID_TRIGGERS`. No emoji, no em dashes.

---

## Architecture

### The journey definition (`begin_funnel.py`)

A new ordered constant defines the 4 cards once:

```
JOURNEY_STEPS = [
  {"key": "scan",  "label": "Scan", "paren": "Your Biofield",
   "base_url": "/begin/voice", "internal": True,  "done_gate": "scan",        "click_trigger": "scan"},
  {"key": "find",  "label": "Find", "paren": "Your Remedy Match",
   "base_url": "/begin/match", "internal": True,  "done_gate": "question",    "click_trigger": "question"},
  {"key": "heal",  "label": "Heal", "paren": "the root causes",
   "base_url": "/begin/ascend","internal": True,  "done_gate": "paid_fork",   "click_trigger": "paid_fork"},
  {"key": "earn",  "label": "Earn", "paren": "Ambassador",
   "base_url": "/begin/path",  "internal": True,  "done_gate": "share_video", "click_trigger": "share_video"},
]
```

All `done_gate`/`click_trigger` values are already in `VALID_TRIGGERS` (`scan`, `question`, `paid_fork`, `share_video`). Internal hrefs are same-origin; if any future step is external, `card_href`-style utm threading applies (the existing `card_href` helper already distinguishes internal vs external by the leading `/`).

### `journey_map(state, ref)` (`begin_funnel.py`)

Pure function. Given a journey `state` (the dict from `get_state`, which carries `unlocked_gates`) and a `ref` slug, returns the 4 cards in order, each:

```
{"key", "label", "paren", "href", "status"}   # status in {"done", "next", "available"}
```

Status rules:
- A step is **done** when its `done_gate` is present in `state["unlocked_gates"]`.
- The **next** step is the FIRST step in order that is not done. Exactly one card is "next" (unless all are done, in which case none is "next").
- All other not-done steps are **available**.
- `href` is built from the step's `base_url` using the SAME internal/external rule `card_href` uses: an internal (`/...`) base is returned as-is (same-origin, no utm); an external (`http...`) base is threaded with the ref-based utm. The 4 journey steps are NOT `CARD_CATALOG` keys, so do not call `card_href(key)` on them; factor the internal/external threading into a small shared helper (e.g. `_thread_href(base_url, ref)`) that both `card_href` and `journey_map` call, so behavior is identical and stays in sync. All four steps are internal today, so each href is its `base_url` unchanged.

`journey_map` does not mutate state and never raises on a normal state dict.

### `/begin/state` payload (`app.py`)

`begin_state` already returns the full state plus `surfaced_cards`. Add one key:

```
payload["journey_map"] = begin_funnel.journey_map(state, ref_slug)
```

`/begin/state` stays a pure GET; no new route. The front end reads `STATE.journey_map`.

### The strip (`static/begin.html`)

A new `<section id="journey-strip">` directly below the hero `<section class="hero">`. It contains a short caption ("Your healing journey") and a `<div id="journey-cards">` the JS fills from `STATE.journey_map`. The section starts hidden (`display:none` or an `unfolded` class absent) until the unfold fires.

Rendering (`renderJourney()`):
- Build one card per `STATE.journey_map` entry, in order. Each card: label + paren + a status treatment (done = muted + check glyph drawn in SVG/text, NOT emoji; next = gold highlight + a small "your next step" tag; available = neutral). Card is an `<a>` to `href`.
- On click: fire `unlock(step.click_trigger)` fire-and-forget (it refreshes STATE and re-renders) and let the link navigate. Navigation must not be blocked by the unlock POST.
- `renderJourney()` is called on every state update — the same three sites #1 wired (`unlock()` `.then`, `arrival()` state-load `.then`, the `postMessage` handler) already call `applyExploreGate()`; add `renderJourney()` beside each.

Unfold (`unfoldJourney()`):
- Idempotent (a module-level `journeyUnfolded` flag; returns if already unfolded).
- Reveals `#journey-strip`, then adds an `in` class to each card with an increasing transition-delay (e.g. 0ms, 120ms, 240ms, 360ms) so they animate in left to right. CSS handles the transition (opacity + translateX).
- If invoked because of the **chat cue**, first append a one-line assistant framing bubble to the hero chat (via the existing `heroAppend('assistant', ...)` from #1) giving the journey meaning: provisional copy "Here's the path we'll walk together - so I can get to know you and help you find the best solutions for your unique needs. Start anywhere." If invoked because of the **video cue**, do NOT append a chat bubble (the strip's own caption carries the meaning).

Triggers:
- **Chat cue:** in the hero chat `send()` flow from #1, after the first AI answer finishes streaming (the `streamMatch(...)` promise resolves for the FIRST time), call `unfoldJourney('chat')`.
- **Video cue:** the `.video` element already has a click handler firing `unlock('video')`. Add `unfoldJourney('video')` to that handler (and to the hero video specifically).
- Whichever calls first wins; the flag makes the second a no-op.

Reload-when-already-underway: on initial state load (`arrival()`), if the journey has progressed (any of the 4 `done_gate`s present, or `current_rung` is past `arrival`/`listening`), call `unfoldJourney()` immediately (no animation needed, or a quick non-staggered reveal) so a returning visitor sees the strip already open instead of waiting for a fresh trigger.

### Reuse (no new backbone)
- State + gates: `get_state` / `journey_state` / `/begin/state` (existing).
- Triggers: `scan`, `question`, `paid_fork`, `share_video` (all existing in `VALID_TRIGGERS`).
- Href threading: the `card_href` internal/external rule (existing).
- Chat bubble injection + personalization: `heroAppend` / `STATE` / the state-refresh sites (all from #1, already in `begin.html`).
- Untouched: `surface()` / `surfaced_cards` / `renderCards` / `/begin/explore` (the contextual layer-5 cards and the full Explore map are a different system).

---

## Data flow

1. Visitor lands -> hero (video | chat) above the fold; `#journey-strip` present but hidden. `arrival()` loads `/begin/state`; if already underway, the strip shows already-unfolded.
2. Visitor either presses play (video cue) or sends a message and the first AI answer finishes (chat cue) -> `unfoldJourney()` fires once. Chat cue also injects the framing bubble.
3. The 4 cards animate in left to right, colored by current progress (Scan/Find/Heal/Earn; the first not-done is the gold "next step").
4. Visitor clicks a card -> `unlock(click_trigger)` fires (advancing the map) and the browser navigates to the destination room.
5. On any later state refresh (`unlock`, return visit, postMessage), `renderJourney()` re-colors the strip from the updated gates.

## Error handling

- If `/begin/state` fails or returns no `journey_map`, render the 4 cards from the static `JOURNEY_STEPS` order as plain "available" (no status), so the strip still works and links still navigate.
- Card clicks always navigate even if the `unlock` POST fails (fire-and-forget; the gate can be re-fired on a later visit).
- `unfoldJourney()` is idempotent; double triggers (video + chat) never double-render or double-inject the framing bubble.
- The framing bubble is appended at most once (guarded by the same unfold flag).
- NO change to the shared `/begin/match/chat` prompt or behavior; the "weaving" is a client-injected framing line, not a model instruction (reliable, controllable, BNSN-refinable later).

## Testing

- **`journey_map` unit tests:** no gates -> Scan is "next", others "available", none "done"; `scan` gate set -> Scan "done", Find "next"; `scan`+`question` set -> Find "done", Heal "next"; all four gates set -> all "done", none "next"; order is always Scan, Find, Heal, Earn; internal hrefs returned as-is (same-origin), and a ref threads exactly as `card_href` threads (assert against `card_href`'s own output for an external case if one exists, else assert internal pass-through).
- **Serve assertion:** `/begin` returns 200 and contains `id="journey-strip"` and the four labels (Scan / Find / Heal / Earn); the strip markup is present even though hidden by default.
- **State payload:** `/begin/state` includes a `journey_map` array of length 4 with the expected keys and a valid `status` for each.
- **Front-end (unfold animation, left-to-right stagger, chat-cue framing bubble, video-cue path, progress coloring, responsive layout):** manual visual pass (state it). Server-side `journey_map` + payload + serve have unit / Flask-test-client coverage.
- Follow deploy-chat test isolation (tmp `$DATA_DIR` / `LOG_DB`; `init_journey_tables`; mock GHL onboarding on any free-tier transition; `importlib` app load). Run via `doppler run -p remedy-match -c prd -- env DATA_DIR=$(mktemp -d) ~/.venvs/deploy-chat311/bin/python -m pytest ...`. NO emoji; no em dashes.

## Notes

- **Live page, no flag.** Ship carefully; visual pass before deploy. `main` autoDeploys to prod, so the merge itself goes live.
- **All copy is provisional** (card labels, parentheticals, caption, framing bubble). A whole-site copy pass via BNSN is planned separately; keep the strings in one place (`JOURNEY_STEPS` + a couple of JS constants) so that pass is an easy find-and-edit.
- The video cue is **play-start** today (placeholder video). Leave a clearly-named hook (`unfoldJourney('video')`) so a future real-video timecode cue can call the same path.
- This is a fixed 4-step journey map, deliberately distinct from the existing data-driven contextual surfacing (`surface()`), which stays as-is for the deeper/Explore surfaces.
