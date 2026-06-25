# Navigation Shell + Journey Map — Design Spec

**Date:** 2026-06-24
**Status:** Design (pre-build). Sub-project **1 of 3** in the illtowell.com wayfinding + recognition arc.
**Guiding principle:** *There's no rule that says healing can't be fun.*

> **Sibling sub-projects (separate specs, NOT this one):**
> - **SP2 — Household-Aware Chat Recognition:** who the visitor is, per-person cross-visit
>   history, "is this you / or a family member?", caregiver consent, PHI-scoped profile switching.
> - **SP3 — Points / Loyalty Economy:** points with **real order value** (a financial ledger,
>   abuse controls, ties into orders/QBO). Depends on SP2. The **family membership tier**
>   ($199/mo = two parents + minor children) is an open business exploration that lives here.
>
> SP1 builds the **visible shell + journey map**. Points, celebrations, and the gem appear but
> are **cosmetic/stubbed** — no real value, no ledger, no identity. SP2/SP3 light them up.

## Goal

Give illtowell.com a single, consistent navigation frame so visitors and family members **never
feel lost**, and turn that frame into an engaging **"adventure-park" journey map** rather than a
list of pages. The map is a **curated spine of ~8–15 marquee "pavilions"** (the self-healing
journey), shared and identical for everyone — so a returning family member or an Ambassador
introducing a referral lands in the same familiar world.

## Current state (the problem, structurally)

- **No shared layout.** Every public page is a standalone file in `static/` (e.g. `begin.html`,
  `begin-explore.html`, `client-portal.html`), each served by its own `send_from_directory(STATIC, …)`
  route in `app.py`. **Zero** templates extend a base; nav markup exists ad-hoc on only 4 pages
  (`atlas.html`, `funnel.html`, `shaira-workspace.html`, `othon-tqm.html`).
- **Consequence (the test-day feedback):** cards that open off-site don't open a new tab and
  strand the visitor with no way back; there's no consistent back/home; no way to return to a
  page you saw before; no map of where you are or where you've been.
- **The funnel pavilions already exist as routes:** `/begin`, `/begin/explore`, `/begin/learn`,
  `/begin/voice`, `/begin/match`, `/begin/path`, `/begin/biofield`, `/begin/ascend`,
  `/begin/tools`, `/begin/product`, `/begin/buy`, plus the 56 `/learn` SEO pages and the member
  surfaces (`client-portal`, `coaching`, `affiliate-hub`, `cert-portal`). SP1 unifies them under
  one shell; it does **not** rebuild them.

## Design

### 1. The Shell — one injected client module across standalone pages

Because there is no template engine to inherit from, the shell is a **single self-contained
client bundle**, not 50 edited files:

- **`static/shell.js` + `static/shell.css`** — render the top **ribbon**, the expand-to-full-map
  overlay, the **"My Path" trail**, the persistent **Back / Home** affordance, external-link
  tagging, and the cosmetic gem/points/celebration layer. Self-contained; no framework.
- **Injection via Flask `after_request`** — a filter inserts the `<link>`+`<script defer>` include
  into served **public** HTML responses (text/html, 200). **Excluded:** `/console/*`, `/admin/*`,
  `/api/*`, JSON, and the email templates. This is one hook, touches zero page files, and
  **auto-covers new pages** (including auto-created topic pages). No-op when the flag is off.

### 2. Two modes — funnel vs. member (the shell decides)

The shell classifies the current route and renders one of two faces:

- **Funnel / explore mode** (`/begin/*`, `/learn/*`, public): the **ribbon IS the nav** — minimal,
  **no escape hatches** that leak people out of the flow. Shows: where you are on the path, the
  glowing next stop, Back/Home. Click the ribbon → full map overlay.
- **Member mode** (`client-portal`, `coaching`, `affiliate-hub`, `cert-portal`, …): an **efficient
  nav bar** (Journal · Reveals · Coaching · Account) **plus** an unobtrusive **upper-left toggle
  icon** that switches **game-map ⇄ efficient-nav**. The toggle keeps the playful map available
  (great for younger/family members) without forcing it on power users.

Mode + current-pavilion are derived from a small checked-in **route→pavilion / route→mode map**.

### 3. The Journey Map — ribbon that expands to the park

- **Ribbon (always on, top):** a slim stylized path of the pavilions with "you are here," the
  **gold gem** marking the next mission, and fogged upcoming stops. Pretty when closed.
- **Click → full map overlay:** the walking-tour "adventure park." Each **pavilion** is a marquee
  area (a *section*, not a single page — "inside there's lots more to discover"), drawn from a
  **categorical style library** (see §5), never bespoke-per-page.
- **Fog-of-war:** unexplored pavilions sit in soft mist and **reveal as you visit**. Fogged
  next-stops carry **open-loop intrigue copy** ("Something about your terrain is waiting in here…").
  Reframes "unseen" as *discovery*, never homework.
- **Channels kept separate (no color collision):** *where I've been* is the **trail** channel — a
  **visible drawn walking path that connects visited pavilions** and extends as you go (the path
  *is* your healing journey made visible — replaces a checkmark). *What's best next* is the **loud**
  channel — the **gold gem / "Start here"** quest marker, ranked by **position + emphasis**, never
  red (red reads as alarm) and never hue-only (colorblind-safe: gem has a label/shape, not just a
  color).

### 4. "My Path" trail + the bug fixes

- **My Path (this visit):** an always-available ordered trail of pages this session (the walked
  path). **Anonymous-safe** — pure client `localStorage`, no identity, no PHI. This alone fixes
  "I clicked away and can't get back."
- **Back / Home** persistent on every in-site page (not the browser button).
- **External links** (anything leaving illtowell.com) get `target="_blank" rel="noopener"` + a tiny
  **"opens elsewhere ↗"** marker — you can't give a back button on a site you don't control, so
  don't navigate them away in the same tab. The shell tags these automatically.

### 5. Categorical art style library

Illustration is tied to **page category** (e.g. symptom · function · remedy · course · tool ·
biofield), **not** to individual pages — so a brand-new page (including auto-created topic pages)
**inherits** a look automatically. SP1 ships an **icon/style set keyed by category**; full bespoke
illustration of the marquee spine stops can be art-directed later without code changes.

### 6. Gamification — present but cosmetic in SP1

The gem, a points counter, celebration animations, and the reward layer are **wired and visible**,
but **client-only and value-free** in SP1. Split into two increments (see Rollout):

**6a — Earning + celebration (ships with the shell):**
- **First-time-only earning.** A reward event fires the **first** time a visitor reaches a pavilion
  /completes an activity (tracked in `localStorage` `seen` set) — **idempotent**, never on
  refresh/re-click. The *value to the business is familiarity* — first time doing "X" (learn the
  philosophy/systems/methods), so repeats earn nothing.
- **Variable amount = scaled to the milestone's importance** (from the spine config), not random.
- **Celebration keyed to the pavilion/activity** — confetti is one option; each category gets the
  animation that fits it.

**6b — Virtual remedies + avatar (the reward layer — a guided tour of the real catalog):**
- Each pavilion can grant a **virtual remedy or healing tool** drawn from a curated
  **featured-formulations set** (Glen picks the marquee ~6–10 of his 150+). Each grant carries the
  real **formulation name**, a one-line **"healing power"** (compliant mechanism/benefit language),
  and an **open loop** to the real product. By **The Dispensary** (pavilion 10) the visitor has
  "used" virtual versions of the remedies they can now actually receive — the game *is* the catalog
  tour and purchase closes the loop. Pre-framing disguised as play.
- **The avatar improves as a brightening, more *coherent biofield* — never a changing body.** As
  remedies are collected the field glows / grows more coherent (on-brand with the coherent-field
  model). **No before/after body imagery** — avoids implied health claims and body-image issues.
- **Guardrails (hard requirements):** in-game "powers" stay **metaphor** and use the **same
  compliant language as the real catalog** — a virtual remedy's ability must not become an implied
  disease-cure claim about the real SKU. Featured copy runs through the compliance/authority
  guardian before ship.

**Hard stub boundary (all of §6).** No ledger, no server balance, no real order value, no identity.
Collecting virtual remedies is a *learning/preframe* device, **not a currency**. SP3 swaps the
cosmetic local counter for a real, identity-bound, audited balance.

## Data

SP1 stays **server-light — no new table**:

- **`static/journey_spine.json`** (checked in) — the ordered pavilions: `slug`, `title`, `routes[]`,
  `category` (→ art style), `intrigue` (teaser copy), `points` (importance-scaled), `mode`.
- **`GET /api/journey`** — returns the spine (+ route→pavilion/mode map). The visitor's *this-visit*
  trail and *this-device* seen-set live in `localStorage`. Cross-visit, identity-bound history
  (the 🟢/🔵/⚪️ "library") is **deferred to SP2** — `/api/journey/state` is named but not built here.

## Out of scope (explicit)

- Knowing **who** the visitor is across visits, profile switching, caregiver consent → **SP2**.
- **Real** points value, ledger, abuse/fraud controls, orders/QBO integration → **SP3**.
- **Family membership tier** ($199/mo) → business exploration, recorded with SP3.
- Authoring the actual pavilion content/illustration (a content task; SP1 ships the machinery +
  a categorical placeholder set).

## Testing

Python (the testable surface):
- `after_request` injection: include inserted on a public HTML page; **NOT** on `/console/*`,
  `/admin/*`, `/api/*`, non-HTML, or non-200; no-op when flag off.
- Pure helpers: route→pavilion resolution; route→mode (funnel vs member) resolution; spine config
  validity (ordered, unique slugs, valid categories, every `routes[]` resolvable).
- `GET /api/journey` returns the spine; shape stable.

Client (pure JS helpers, unit-tested in isolation + manual/visual QA for the rendered shell):
- trail push de-dups consecutive repeats; seen-set add is idempotent; **first-visit points award
  exactly once** per pavilion; external-link detection tags off-site hrefs only.

## Rollout

- Behind flag **`JOURNEY_SHELL_ENABLED`** (Doppler `remedy-match/prd`), **dark by default**. Flag
  off → `after_request` injects nothing; site is byte-identical to today.
- Additive: no schema change, no destructive ops, console/admin untouched, funnel routes unchanged.
- Ships assets + spine config + endpoint together; flip flag to pilot, validate on `/begin/*`
  first, then member surfaces.
- **Increment split:**
  - **1a — shell + map + visible trail + bug-fixes** (the fast wayfinding win): ribbon→map,
    pavilions, fog, gem, "My Path" trail, Back/Home, external-link tagging, two modes. Ships first.
  - **1b — virtual-remedy + avatar reward layer** (§6b): featured-formulations set, avatar
    biofield-coherence progression, open-loop teasers. Layered on after 1a, still cosmetic.

## Inputs needed from Glen (during planning, not blocking)

1. **Pavilion spine** — drafted below (Appendix A) from the real `/begin/*` sections; Glen edits
   names/order/intrigue copy.
2. **Category → art style** direction (placeholder icon set is fine to start).
3. **Featured-formulations set for 1b** — the marquee ~6–10 formulations the avatar can collect as
   virtual remedies, each with a one-line "healing power."

## Appendix A — Drafted pavilion spine (for Glen's edits)

Adventure-park flavor name + plain function, in healing-journey order. `/begin/explore`
("Explore Everything") = the **full-map overview itself**, not a stop.

| # | Pavilion (flavor) | Route | Journey beat | Virtual reward (draft) |
|---|---|---|---|---|
| 1 | **The Gateway** — Welcome | `/begin` | Arrive | map unlocked, starter field |
| 2 | **The Listening Pool** — Voice & Frequencies | `/begin/voice` | Listen — your body speaks | a frequency attunement |
| 3 | **The Hall of Mirrors** — Biofield Analysis | `/begin/biofield` | See — terrain revealed | first virtual remedy (foundational) |
| 4 | **The Matching** — Find Your Remedy | `/begin/match` | Match — the one remedy | your matched virtual remedy |
| 5 | **The Crossroads** — Two Ways to Begin (intake) | `/begin/path` | Choose your path | path-chosen mark |
| 6 | **The Library** — The Research / Learn | `/begin/learn` | Understand | an insight power |
| 7 | **The Apothecary** — Ingredient Deep-Dive | `/begin/ingredient` | Look inside — mechanisms | a mechanism tool |
| 8 | **The Workshop** — Tools & Partners | `/begin/tools` | Equip — healing tools | a healing tool *(external partners)* |
| 9 | **The Ascent** — Ascend / membership | `/begin/ascend` | Deepen — go further | a depth/tier unlock |
| 10 | **The Dispensary** — Product / Order | `/begin/product` + `/begin/buy` | Receive — your real remedy | **virtual → real: loop closes** |
