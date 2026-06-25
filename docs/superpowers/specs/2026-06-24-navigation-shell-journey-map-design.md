# Navigation Shell + Journey Map — Design Spec

**Date:** 2026-06-24
**Status:** Design (pre-build). Sub-project **1 of 3** in the illtowell.com wayfinding + recognition arc.
**Approach:** **Option A — re-skin, don't rebuild.** Reuse the existing `begin_funnel` journey
engine; the only new engine is the persistent shell.
**Guiding principle:** *There's no rule that says healing can't be fun.*

> **Sibling sub-projects (separate specs, NOT this one):**
> - **SP2 — Household-Aware Chat Recognition:** who the visitor is, per-person cross-visit
>   history, "is this you / or a family member?", caregiver consent, PHI-scoped profile switching.
>   *(Partially exists already — `begin_funnel.get_state` unions gates across devices by email,
>   and TOS is tracked.)*
> - **SP3 — Points / Loyalty Economy:** points with **real order value**.
>   *(Largely exists already — `dashboard/points.py`, `rewards.py`, `orders.settle_order_points`
>   (earn 5%), `pricing.apply_points`, `CLIENT_POINTS_ENABLED`.)* The **family membership tier**
>   ($199/mo = two parents + minor children) is an open business exploration that lives here.

## Goal

Give illtowell.com a single, consistent navigation frame so visitors and family members **never
feel lost**, and re-skin the existing funnel journey as an engaging **"adventure-park" map** rather
than a flat list. **No new journey data model** — the map renders what `begin_funnel` already
produces.

## Current state (what exists — reuse it)

- **No shared layout.** Every public page is a standalone file in `static/` (e.g. `begin.html`,
  `begin-explore.html`, `client-portal.html`), each served by its own `send_from_directory(STATIC, …)`
  route in `app.py`. **Zero** templates extend a base; nav markup exists ad-hoc on only 4 pages.
  **This is the root cause of "everyone gets lost," and the real new work of 1a.**
- **A canonical journey engine already exists** in `begin_funnel.py` (tested, prod-wired):
  - `JOURNEY_STEPS` = **4 stages: Scan → Find → Heal → Give**, each with ordered sub-steps,
    gate-driven, with fractional `fill`, `status` (`done`/`next`/`available`), and smart `href`s.
  - `journey_map(state, ref, signals)` → the per-stage cards (pure, tested in
    `tests/test_begin_journey_map.py`).
  - `_EXPLORE_LAYOUT` → a sectioned directory served at `/begin/explore`.
  - `get_state(cx, session_id, email)` → unions gates **across devices by email** + awareness + TOS.
- **`GET /begin/state`** already returns JSON: `journey_map`, `surfaced_cards`, gates, awareness —
  keyed by the **`amg_session`** cookie (uuid, 1-yr, set on begin pages) and `rm_ref`. Member
  identity via `get_authenticated_user(request)`.
- **Consequence (the test-day feedback):** off-site cards don't open a new tab and strand the
  visitor; no consistent back/home; no way to return to a page you saw; no map of where you are.
- **Static assets** serve from `/static/<path:filename>`. JSON-into-HTML injection precedent:
  `app.py` membership-pause does `html.replace("</head>", "<script>…</script>\n</head>")`.

## Design

### The model: 4 lands, pavilions inside (matches the existing engine)

The adventure map renders the existing 4 stages as **lands**, each containing **pavilions** (the
stage's sub-steps + relevant `_EXPLORE_LAYOUT` items). This *is* "sections, and inside there's lots
more to discover" — and it reuses tested, wired code rather than replacing it. See Appendix A.

### 1. The Shell — one injected client module across standalone pages

No template engine to inherit from, so the shell is a **single self-contained client bundle**:

- **`static/shell.js` + `static/shell.css`** — render the top **ribbon**, the expand-to-full-map
  overlay, the **"My Path" trail**, persistent **Back / Home**, external-link tagging. Self-contained;
  no framework. On load it fetches **`GET /begin/state`** and renders the `journey_map` it returns.
- **Injection via Flask `after_request`** — inserts the `<link>`+`<script defer>` include into served
  **public** HTML (text/html, 200) by replacing `</head>` (same precedent as membership-pause).
  **Excluded:** `/console/*`, `/admin/*`, `/api/*`, `/begin/state` + other JSON, non-200, non-HTML,
  and email templates. One hook, zero page-file edits, **auto-covers new pages**. **No-op when the
  flag is off.**

### 2. Two modes — funnel vs. member (the shell decides)

- **Funnel / explore mode** (`/begin/*`, `/learn/*`, public, unauthenticated): the **ribbon IS the
  nav** — minimal, **no escape hatches.** Shows where you are on the path, the glowing next stop,
  Back/Home. Click → full map.
- **Member mode** (`get_authenticated_user` present, or member routes `client-portal`, `coaching`,
  `affiliate-hub`, `cert-portal`): an **efficient nav bar** (Journal · Reveals · Coaching · Account)
  **plus** an unobtrusive **upper-left toggle** that flips **game-map ⇄ efficient-nav** (serves
  younger/family members and power users both).
- Mode + current-land/pavilion derived from a small checked-in **route map** in `shell.js`.

### 3. The Journey Map — ribbon that expands to the park

- **Ribbon (always on, top):** a slim stylized path of the **4 lands** with "you are here," the
  **gold gem** marking the current stage (**`journey_map` already tags it `status:"next"`** — the gem
  just renders that), and fogged upcoming lands. Pretty when closed.
- **Click → full map overlay:** the walking-tour park. Each **land** opens to its **pavilions**
  (sub-steps + Explore items), drawn from a **categorical style library** (§5), never bespoke-per-page.
- **Fog-of-war:** lands/pavilions not yet reached (`status:"available"` beyond the current `next`)
  sit in soft mist and **reveal as gates complete.** Fogged next-stops carry **open-loop intrigue
  copy.** Reframes "unseen" as *discovery*, never homework.
- **Channels kept separate (no color collision):** *where I've been* = a **visible drawn walking
  path** connecting completed lands (driven by `fill`/`done`), extending as you go (replaces a
  checkmark). *What's best next* = the **gold gem** on the `status:"next"` land — ranked by position +
  emphasis, **never red** (reads as alarm), **never hue-only** (gem has a label/shape — colorblind-safe).

### 4. "My Path" trail + the bug fixes (the core wayfinding win)

- **My Path (this visit):** an always-available ordered trail of pages this session — pure client
  `localStorage`, **anonymous, no PHI.** Fixes "I clicked away and can't get back."
- **Back / Home** persistent on every in-site page (not the browser button).
- **External links** (anything leaving illtowell.com) get `target="_blank" rel="noopener"` + a tiny
  **"opens elsewhere ↗"** marker, applied automatically by the shell. (You can't give a back button
  on a site you don't control — so don't navigate them away in the same tab.)

### 5. Categorical art style library

Illustration is tied to **page/pavilion category** (e.g. scan · find · heal · give · learn ·
remedy · tool), **not** to individual pages — so new pages **inherit** a look. 1a ships an
**icon/style set keyed by category**; bespoke illustration of marquee stops is art-direction later,
no code change.

### 6. Gamification — deferred out of 1a

**1a carries NO points.** A cosmetic exploration-points counter would collide with the existing
real `dashboard/points.py` economy. Defer all earning/celebration/virtual-remedy mechanics to **1b**,
where they're reconciled against the real points system (and SP3). The **gem and the visible trail**
in 1a are pure rendering of existing `journey_map` state — not a reward economy.

**1b (later, still cosmetic-vs-real reconciliation):** virtual remedies as a guided catalog tour
(featured-formulations set, real names + compliant "healing power" + open loop), avatar that
improves as a **brightening, more coherent biofield — never a changing body**, in-game powers stay
**metaphor + compliant language** (run through the compliance/authority guardian).

## Data

1a stays **server-light — no new table, no new endpoint, no new json spine:**

- **Reads `GET /begin/state`** for `journey_map` + state (keyed by `amg_session`).
- **`static/shell-map.json`** (checked in) — purely *presentational* config: per-land flavor
  name, per-category icon/style, intrigue copy. Maps onto the existing 4 stages; does **not** define
  the journey (the engine does).
- *This-visit* trail lives in `localStorage`. Cross-visit identity-bound history (the 🟢/🔵/⚪️
  "library") is **SP2**.

## Out of scope (explicit)

- Knowing **who** the visitor is across visits, profile switching, caregiver consent → **SP2**
  (partly exists via `get_state` email-union).
- Points value/ledger/abuse controls → **SP3** (largely exists via `points.py`).
- **Family membership tier** ($199/mo) → with SP3.
- Restructuring the `begin_funnel` journey engine (that was rejected Option B).
- All §6 reward mechanics (virtual remedies, avatar, celebrations, cosmetic points) → **1b**.

## Testing

Python (the testable surface), following the `tests/` monkeypatch-`LOG_DB` client-fixture pattern:
- `after_request` injection: include inserted on a public HTML 200 (e.g. `begin.html`); **NOT** on
  `/console/*`, `/admin/*`, `/api/*`, `/begin/state` (JSON), non-200, non-HTML; **no-op when flag off**.
- Pure helper(s): route→mode (funnel vs member) resolution; presentational `shell-map.json` validity
  (every land maps to an existing `JOURNEY_STEPS` key; every category has a style).

Client (pure JS helpers, unit-tested in isolation + manual/visual QA of the rendered shell):
- trail push de-dups consecutive repeats; external-link detection tags off-site hrefs only;
  ribbon renders `journey_map` (gem on `status:"next"`, trail on completed lands).

## Rollout

- Behind flag **`JOURNEY_SHELL_ENABLED`** (Doppler `remedy-match/prd`), **dark by default**. Flag
  off → `after_request` injects nothing; site is byte-identical to today.
- Additive: no schema change, no destructive ops, console/admin/funnel routes untouched, the
  `begin_funnel` engine unchanged.
- Ships assets + presentational config together; flip flag to pilot, validate on `/begin/*` first,
  then member surfaces.
- **Increment split:**
  - **1a — shell + map render + visible trail + bug-fixes, NO points** (the fast wayfinding win).
    **BUILT 2026-06-25** (`shell_nav.py`, `static/shell.{js,css}`, `static/shell-map.json`,
    `app.py` after_request; 20 tests). **1a go-live:** set `JOURNEY_SHELL_ENABLED=1` in Doppler
    `remedy-match/prd` after pilot review.
  - **1b — reward layer** (§6) — virtual remedies, avatar, celebrations, reconciled with real points.

**Local test command** (the suite needs prod secrets at import but a local data dir):
`doppler run -p remedy-match -c prd -- env DATA_DIR=/tmp/jshell-test python3 -m pytest …`

## Inputs needed from Glen (during planning, not blocking)

1. **Land/pavilion flavor** — Appendix A draft (names, intrigue copy); the 4 lands themselves are
   fixed by the engine (Scan/Find/Heal/Give).
2. **Category → art style** direction (placeholder icon set is fine to start).
3. **(1b later)** featured-formulations set for the virtual-remedy tour.

## Appendix A — Map presentation over the existing engine (for Glen's edits)

The **4 lands are fixed** (`begin_funnel.JOURNEY_STEPS`); flavor names + the pavilions inside are
presentational and editable. `/begin/explore` = the **full-map overview itself**, not a land.

| Land (engine key) | Flavor name | Beat | Pavilions inside (existing sub-steps) |
|---|---|---|---|
| **scan** | The Listening Pool | **Scan** your biofield | Voice scan · Wellness Whispering MasterClass |
| **find** | The Hall of Mirrors | **Find** your remedy match | Match via chat · Biofield interpretation |
| **heal** | The Sanctuary | **Heal** the root causes | Intake form · Accelerated Self Healing™ MasterClass |
| **give** | The Beacon | **Give** — lift others | Be an Ambassador · Bring a friend |

Entrance/Welcome (`/begin`), Ingredient Deep-Dive, Tools & Partners, and Order/Product remain
reachable surfaces (and `_EXPLORE_LAYOUT` sections); in the map they attach as **pavilions/side-trails
under the nearest land** rather than as separate top-level lands — keeping the spine the canonical 4.
