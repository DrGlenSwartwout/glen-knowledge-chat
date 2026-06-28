# Journey Ribbon — Trademark Scene Redesign

**Date:** 2026-06-26
**Surface:** illtowell.com journey shell (dark behind `JOURNEY_SHELL_ENABLED`)
**Status:** Design approved — ready for implementation plan

---

## Goal

Redesign the journey-shell wayfinding so it (a) carries Dr. Glen Swartwout's **trademarks** as the land names to build brand recognition, and (b) upgrades the imagery from CSS+emoji to a **photo-quality nature scene** — a path leaving a hobbit "home", past four wooden signposts, to a glowing "Healing Oasis". The scene lives in the expanded map view and is tied into the thin ribbon via small thumbnails sliced from the same image.

This is interface-only. No change to the funnel engine, gates, points, or routing.

## Decisions (locked)

- **Names:** drop the poetic names (Listening Pool / Hall of Mirrors / Sanctuary / Beacon). Trademarks only.
- **Master art:** `v2-3-teal-dawn` (source: `~/Downloads/journey-ribbon-samples/v2-3-teal-dawn.png`, Flux 1.1 Pro). Four evenly-spaced blank rectangular signboards, glowing hobbit door at left, calm water oasis at right, purple-lupine + gold palette.
- **Sign text:** overlaid as crisp HTML/CSS text positioned over the signs — never baked into the image (keeps the ™ clean and the names editable).
- **Fog-of-war:** locked-land ribbon thumbnails render **grayscale + slight blur**, blooming to full color as each land unlocks.
- **Gating:** ships together behind the existing `JOURNEY_SHELL_ENABLED` flag. No new flag.

## Name mapping

Engine keys are unchanged; only display names change.

| key (unchanged) | function | new display name |
|-----------------|----------|------------------|
| `scan` | listen to the body | **Wellness Whispering** |
| `find` | match the remedy | **Remedy Match** |
| `heal` | heal the root causes | **Accelerated Self Healing™** |
| `give` | lift others / share | **Healing Oasis** |

Scene reading order, left → right: **🏠 Home → Wellness Whispering → Remedy Match → Accelerated Self Healing™ → Healing Oasis (the water).**

## Architecture

Three layers, matching the current implementation:

### 1. Config (`static/shell-map.json`) — extended schema

```jsonc
{
  "lands": {
    "scan": { "name": "Wellness Whispering", "category": "scan",
              "intrigue": "Your body is already speaking. Step in and listen.",
              "thumb": "/static/journey/thumb-scan.webp",
              "featured": { ... unchanged ... } },
    "find": { "name": "Remedy Match", ... "thumb": "/static/journey/thumb-find.webp" },
    "heal": { "name": "Accelerated Self Healing™", ... "thumb": "/static/journey/thumb-heal.webp" },
    "give": { "name": "Healing Oasis", ... "thumb": "/static/journey/thumb-give.webp" }
  },
  "categories": { ...unchanged (icon/hue kept as fallback)... },
  "scene": {
    "image": "/static/journey/scene.webp",
    "home":  { "x": 8,  "y": 55, "href": "home" },
    "signs": {
      "scan": { "x": 40, "y": 40, "w": 9,  "rot": 0 },
      "find": { "x": 50, "y": 39, "w": 11, "rot": 0 },
      "heal": { "x": 73, "y": 28, "w": 16, "rot": 0 },
      "give": { "x": 90, "y": 50, "href": "give" }
    }
  }
}
```

- `thumb` — path to the pre-cut ribbon thumbnail for that land.
- `scene.image` — web-optimized master scene.
- `scene.signs.<key>` — placement of the overlaid trademark label as **percentages of the rendered scene box** (x = left%, y = top%, w = label width%, rot = degrees). Exact values measured from the final optimized image during implementation (the numbers above are placeholders).
- `scene.home` / `scene.signs.give` — clickable anchors for the hobbit home and the oasis.
- `category.icon`/`hue` are retained as graceful fallback if an image fails to load.

### 2. Asset pipeline (build step, scripted)

A small repeatable script (`scripts/build_journey_assets.py` or shell using `sips`/Pillow):

1. Read the source `v2-3-teal-dawn.png`.
2. Emit `static/journey/scene.webp` — resized to ~1600px wide, quality-tuned to < 300 KB.
3. Emit four `thumb-<key>.webp` crops (square-ish, ~96px) framed on each sign, and `thumb-home.webp` on the hobbit door.
4. Print the measured sign bounding boxes so the `scene.signs` coords can be filled into `shell-map.json`.

The script is committed so the assets are reproducible if the master art changes.

### 3. Renderers (`static/shell.js` + `static/shell.css`)

- **Ribbon (`renderLands`)** — replace the emoji `<span class="js-icon">` with `<img class="js-thumb" src="{lands[key].thumb}">`. Apply CSS filter for fog state: `.js-land.fog .js-thumb { filter: grayscale(1) blur(0.6px); opacity:.5 }`; done/next lands show full color. Gold 💎 on `.next` unchanged. Emoji remains as `onerror`/missing-thumb fallback.
- **Expanded map (`buildOverlay`)** — replace the cream pavilion grid with a single scene stage:
  - `<div class="js-scene">` with `background-image: scene.image`, fixed 16:9 aspect ratio, responsive width.
  - For each land, an absolutely-positioned `<button class="js-sign-label" style="left:x%; top:y%; width:w%; transform:rotate(rot)">` containing the trademark text (engraved-look: dark brown text, subtle light text-shadow, letter-spacing). Click → existing `card.href` (external → new tab, same logic as today). Fog/done/next states tint the label.
  - Home anchor button over the hobbit door (→ Home), oasis anchor over the water (→ Give href).
  - The featured-product "Claim 15% off" affordance (when `REWARDS`) is preserved, shown in a panel below or on the active sign's popover.
- **Theme:** the scene panel carries its own surface treatment independent of the site light/dark theme; label contrast is tuned against the wooden boards, which read the same in both themes.

## Data flow (unchanged)

`/begin/state` → `journey_map()` still returns the same `[{key,label,paren,href,status,fill,steps}]`. `shell.js` still merges that with `shell-map.json`. The only additions are reading `lands[].thumb` and the new `scene` block. No backend route or engine change.

## Responsive / a11y

- Scene scales to container width via the 16:9 box; sign labels stay anchored because they are positioned in %. Label font uses `clamp()` so text stays legible from mobile to desktop.
- Each sign label is a real `<button>` with an `aria-label` of the land name + status (locked/available/done). Keyboard focusable; the overlay traps focus and closes on Esc (preserve existing overlay open/close behavior).
- `<img>` thumbnails carry `alt` = land name.

## Testing

- **`test_shell_map_config.py`** (extend): every land has a non-empty `name` and a `thumb`; `scene.image` present; each of the four `scene.signs` keys has numeric `x/y`; referenced static files exist on disk; names include the four trademarks (and the ™ on Accelerated Self Healing).
- **Asset test:** `scene.webp` exists and is < 350 KB; the five thumbnails exist.
- **`test_begin_journey_map.py`** — unchanged and must still pass (keys untouched), proving the rename is display-only.
- **Render-verify (mandatory, per prior incident):** headless browser loads a shell-injected page with `JOURNEY_SHELL_ENABLED=1`, opens the map overlay, asserts the scene image and all four sign labels render, the trademarks appear as text (incl. ™), and the console has **zero errors**. Verify ribbon thumbnails render and locked lands carry the grayscale/blur class. Do **not** rely on "script injected" as proof.

## Out of scope

- No change to funnel gates, points, or `begin_funnel.JOURNEY_STEPS` labels/hrefs.
- No new backend endpoint.
- Member-mode vs funnel-mode behavior unchanged (same scene/names for both).
- Generating alternate scenes for other surfaces (landing pages, emails) — separate effort.

## Implementation phases (for the plan)

1. **Assets** — build script + commit `scene.webp` + 5 thumbnails; produce a standalone static overlay **preview** (names on signs) for Glen's approval *before* touching app code.
2. **Config** — extend `shell-map.json` (names, thumbs, scene block with measured coords) + extend config tests.
3. **Ribbon** — thumbnails + fog grayscale/blur in `shell.js`/`shell.css`.
4. **Map overlay** — scene stage + positioned trademark labels + home/oasis anchors.
5. **Render-verify + go-live** — headless DOM/console check behind the existing flag.
