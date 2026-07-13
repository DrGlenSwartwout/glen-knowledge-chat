# Homepage & Header Redesign — illtowell.com + myhealingoasis.com

**Date:** 2026-07-12
**Status:** Design approved, pending implementation plan
**Register:** brand (homepage) + product (portal)

## Problem

The two sites share one injected "journey ribbon" (`static/shell.js`, injected by
`shell_nav.py` via the after-request hook in `app.py`), plus their own page-level
headers. The result is a chrome layer where gamification features designed for
invested clients (biofield orb, offers Wallet, "lands" progress bar, Journey Quest,
"My Path") leak onto the public illtowell.com homepage, where a first-time visitor
has no idea what a "biofield orb" or a "wallet of offers" is. The icons are raw
emoji with no system, so they render inconsistently and read as toy-like. Dark mode
is implemented three separate times on two different storage keys, so the homepage
and portal toggles do not agree.

The homepage's job is to make a stranger want to start. The portal's job is to help
a client make progress. Game mechanics belong in the second room, not the first.

## Decisions (locked with Glen)

1. **Strip the game layer from illtowell.com.** The orb, Wallet, lands progress bar,
   Journey Quest, and "My Path" do not appear on the public homepage. A first-time
   visitor sees exactly two choices: learn how it works, or start.
2. **Homepage direction: hybrid.** Direction A's committed deep-green hero with a
   live voice-signature waveform, followed by Direction B's calm horizontal
   Scan -> Find -> Heal flow. (Mockup: artifact `two-directions-v2`.)
3. **Onboarding process is "sold" on the homepage as a static 3-step explainer**
   (Scan -> Find -> Heal), answering "what do I do and why" before commitment. The
   live click-to-continue progress bar belongs in the portal, not the homepage.
4. **CTA naming is stateful.** No portal yet -> "Start My Healing Oasis". Returning /
   known -> "Enter My Healing Oasis". One brand phrase ("Healing Oasis"), the word
   "portal" dropped as jargon. Replaces the current `#oasis-btn` "My Healing Oasis".
5. **Nav:** How it works · Products · About.
6. **Portal header:** keep the three-tab bar (Current Analysis / Scan History /
   Orders & Invoices); keep the live progress bar, relabeled **"Your healing
   journey"**; **retire "My Path"**; **drop the biofield orb from the chrome**;
   **keep the Wallet, renamed "My Offers"**, still fed by `/api/journey/wallet`.
7. **Theme system: one three-state toggle** (Light / Dark / **Auto**), one storage
   key, always visible on both sites. Auto follows local sunrise/sunset computed in
   the browser. Default new visitors to Auto.
8. **Icons:** one inline-SVG set replaces every emoji glyph.

## Scope

### 1. Shared header (the injected ribbon) — `shell.js` / `shell_nav.py` / `shell.css`

The ribbon stays the single injected header, but `buildRibbon()` becomes
mode-aware in a stricter way than today's `funnel` vs `member` split.

- **Funnel mode (public, illtowell.com):** wordmark (left), nav links
  (How it works · Products · About), the stateful Healing Oasis CTA, the theme
  toggle. Remove from funnel mode: `🏠` home, `←` back, `🗺️` map, `.js-path` lands
  bar, `.js-mypath-btn`, `.js-orb`, `Wallet`. These are already partly flag-gated
  (`REWARDS_1B_ENABLED`); the change makes their absence unconditional in funnel
  mode rather than flag-dependent.
- **Member mode (portal, myhealingoasis.com):** keep the map toggle and the member
  nav; add the relabeled progress bar ("Your healing journey"); keep "My Offers".
  Remove the orb and "My Path".
- All emoji glyphs replaced with the SVG icon set (see §5).

### 2. illtowell.com homepage (hybrid) — `static/begin.html`

- **Hero:** committed deep-green ground, warm serif display headline, the live
  voice-signature waveform (Canvas, animated, honors `prefers-reduced-motion`),
  stateful CTA, a "takes about 2 minutes" reassurance line. Replaces the current
  centered `.brandbar` + hero treatment.
- **How it works:** calm horizontal numbered Scan -> Find -> Heal flow with SVG
  icons, one line of *what* + one line of *why* per step, a welcome-call reassurance
  line, and a closing CTA. This is a static explainer, not the interactive ribbon
  progress bar.
- **Imagery:** hero and steps use real botanical/lifestyle photography (Glen already
  generates these via the formulation-image-studio skill). No colored-block
  placeholders. Photography asset sourcing is a plan task, not this spec.
- The `#oasis-btn` request-a-link modal flow (`POST /api/healing-oasis/request`,
  gated by `GET /api/healing-oasis/status`) is preserved; only its label and
  placement change (it becomes the header CTA + hero CTA).

### 3. myhealingoasis.com portal header — `static/client-portal.html` / `client-login.html`

- Keep `.tabbar#portalTabs` (Current Analysis / Scan History / Orders & Invoices).
- Keep the live progress bar (`renderLands()` data via `/begin/state`), relabeled
  "Your healing journey". Click-to-continue behavior unchanged.
- Retire "My Path" (it only listed pages-visited-this-session and collided with the
  real journey).
- Remove the biofield orb from the chrome. If biofield status is surfaced later, it
  belongs inside the Current Analysis tab with a real label, not as a mystery dot.
- Rename "Wallet" -> "My Offers"; panel and `/api/journey/wallet` data unchanged.
- Replace the portal's own `#themeToggle` (its own SVG + `rm_portal_theme` key) with
  the unified toggle (§4).

### 4. Theme system: three-state, sun-following — new `static/theme-mode.js`

Consolidate the three current implementations (ribbon toggle in `shell.js`, floating
`static/theme-toggle.js`, portal/login `#themeToggle`) into one module.

- **States:** Light / Dark / Auto. Persist the *mode* (not just the resolved theme)
  under a single key (standardize on `rm-theme`; migrate any `rm_portal_theme` value
  on first load). Default new visitors to Auto.
- **Auto:** resolve theme from local sunrise/sunset. Algorithm: NOAA "Almanac for
  Computers" sunrise/sunset, computed client-side from latitude + longitude + date
  (no server, no external call). Verified accurate to within ~2 min against Honolulu,
  London, and Anchorage. Light between sunrise and sunset, Dark otherwise. Flip at
  the next threshold via a timer; recompute daily.
- **Location:** longitude falls back to the device time-zone offset; latitude is the
  seasonal driver. On selecting Auto, request Geolocation **once**, cache lat/long in
  `localStorage`, and never re-ask. If permission is denied or unavailable, fall back
  to a sensible day/night window and continue. Coordinates never leave the device.
- Applies `data-theme` on `:root`; both sites' palettes must be driven by CSS custom
  properties so light and dark both stay legible (portal's committed worlds excepted).

### 5. Icon system — `shell.js`, `shell-map.json`, both HTML files

- One inline-SVG line set (uniform ~2px stroke), replacing every emoji: home, back,
  map, theme sun/moon/auto, the land-category icons (scan waveform, find rings, heal
  sprout), the arrow, close. No icon font, no emoji, no external sprite.
- SVG source lives in one place (a small JS map or a shared partial) so both the
  ribbon and the page bodies pull from the same set.

## Out of scope

- Rebuilding the Journey Quest overlay (`journey-quest.js`); it is simply not shown
  in funnel mode. Its own future is a separate decision.
- The `/api/journey/wallet` offers engine internals (only the label changes).
- New photography generation (a plan task; assets come from the image-studio skill).
- A self-hosted webfont decision: baseline is a system serif/sans stack
  (Iowan/Palatino display, Avenir Next/system-ui body); adopting a self-hosted
  display face is a follow-up, not a blocker.

## Files affected

- `static/shell.js` — mode-aware `buildRibbon`, remove funnel-mode game elements,
  SVG icons, relabels.
- `static/shell.css` — icon/toggle styling, remove orb/lands styling from funnel.
- `shell_nav.py` — mode detection unchanged; confirm funnel vs member gating.
- `static/shell-map.json` — land icons become SVG references.
- `static/begin.html` — new hero + how-it-works, stateful CTA, remove brandbar.
- `static/client-portal.html` — relabels, retire My Path, drop orb, unified toggle.
- `static/client-login.html` — unified toggle.
- `static/theme-toggle.js` — folded into new `static/theme-mode.js` (or retired).
- New: `static/theme-mode.js` (three-state + sun engine), `static/icons.js` (or
  inline SVG map).
- Flags to review: `REWARDS_1B_ENABLED`, `JOURNEY_QUEST_ENABLED`,
  `HEALING_OASIS_ENABLED`.

## Testing / verification

- deploy-chat has no CI (merge == deploy), so verification is render-based.
- Render both sites locally (headless Chrome) and confirm: funnel header has no
  orb/wallet/lands/My Path; portal header keeps tabs + "Your healing journey" + "My
  Offers" and has no orb/My Path.
- Unit-test the sunrise/sunset function against fixed date + lat/long fixtures
  (Honolulu, London, Anchorage, plus a near-polar case returning null).
- Verify theme mode persists across a reload and that homepage and portal now agree
  on the resolved theme.
- Confirm the Healing Oasis request-link modal still posts to
  `/api/healing-oasis/request` after the CTA relabel/move.

## Risks

- The ribbon is injected on every page via an after-request hook; changing
  `buildRibbon` affects all surfaces at once. Stage behind the existing mode split
  and verify member surfaces (coaching, practitioner) still render.
- Geolocation prompts can feel intrusive; defaulting to Auto must degrade silently
  when permission is denied, never block the page.
- Stateful CTA depends on knowing whether the visitor has a portal; reuse the
  existing `/api/healing-oasis/status` signal rather than inventing new state.
