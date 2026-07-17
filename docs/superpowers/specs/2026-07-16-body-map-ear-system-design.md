# Body Map — Ear (Auricular) System — Design Spec

**Date:** 2026-07-16
**Status:** Approved design, pre-implementation
**App:** deploy-chat (`glen-knowledge-chat`)
**Builds on:** Body Map v1 (iris/sclera, PR #945 merged) and the Atlas deep-link slice (PR #948; the ear generalizes its focus params, so the ear branches off that work).

---

## Goal

Add the **ear (auricular)** as the second Body Map system, and in doing so generalize the engine from iris-only (radial sectors) to **typed geometry**, so the ear — and later foot/hand/face/meridians — drop in as data rather than forked renderers. The generalization is **additive**: the live iris renders identically; the engine learns point geometry, a drawn ear surface, and region grouping alongside the untouched iris path.

## Locked decisions (from brainstorm)

1. **Typed geometry (additive).** Every zone can declare `geometry.type`. A zone with `radial`+`sector` (the iris) is treated as `sector` with no change. Ear zones use `geometry: {type:"point", x, y}`. The renderer dispatches on type. Future: `polygon`, `path`.
2. **Drawn SVG ear outline** is the reference surface and the warp template — no shipped raster image. Matches v1's "draw from geometry" principle; themes light/dark.
3. **Grouping axis = ear anatomical regions** (helix, antihelix, concha, tragus, antitragus, lobe, triangular fossa), used as the toggle filter the way germ layers were for the iris, and carrying the inverted-fetus logic (lobe = head, antihelix = spine, concha = organs).
4. **Laterality = `side` (left/right)**, generalizing v1's `eye`. The ear ships the **left** side first; mirror later. The iris keeps `eye`; the engine reads `side || eye`.
5. **Overlay anchors are system-defined.** The ear uses three landmarks — top of helix, bottom of lobe, tragus — for the same similarity warp. Anchors move from hardcoded (iris) to a per-system list the page reads.
6. **Source = a standard auricular chart as scaffold** (like the iris used a standard topography), which Glen corrects via the existing admin overlay editor.

---

## Architecture (additive generalization)

The three v1 surfaces each gain a dispatch, defaulting to the existing iris behavior when the new fields are absent.

### 1. Data model — generalized zone + system fields

A system file may now declare its **reference frame**, a **grouping list**, and an **anchor set**; zones may declare **`side`**, **`group`**, and **`geometry`**.

`data/bodymap-ear.json` (new):
```json
{
  "system": "ear",
  "reference_frame": "ear_outline",
  "outline": "M .... Z",
  "groups": [
    { "id": "helix", "label": "Helix (outer rim)" },
    { "id": "antihelix", "label": "Antihelix (spine)" },
    { "id": "concha", "label": "Concha (internal organs)" },
    { "id": "triangular-fossa", "label": "Triangular fossa" },
    { "id": "tragus", "label": "Tragus" },
    { "id": "antitragus", "label": "Antitragus" },
    { "id": "lobe", "label": "Lobe (head and face)" }
  ],
  "anchors": [
    { "key": "helix-top", "template": {"x": 0.46, "y": 0.06}, "hint": "Tap the top of your ear, at the top of the rim." },
    { "key": "lobe-bottom", "template": {"x": 0.52, "y": 0.96}, "hint": "Tap the bottom of your earlobe." },
    { "key": "tragus", "template": {"x": 0.30, "y": 0.60}, "hint": "Tap the tragus, the small flap in front of the ear canal." }
  ],
  "zones": [
    {
      "id": "ear-L-shenmen",
      "side": "left",
      "group": "triangular-fossa",
      "geometry": { "type": "point", "x": 0.44, "y": 0.28 },
      "anatomy": "Shen Men",
      "meaning_standard": "Master calming point; relaxation and stress regulation.",
      "meaning_glen": "",
      "layers": { "embryological_depth": null, "stress_affirmation": null, "touch_for_health": null }
    }
  ]
}
```
- **Normalized frame:** the ear `outline` path and every point `x,y` live in a normalized `[0,1] × [0,1]` box (mapped to the 600×600 canvas at render). This is distinct from the iris `unit_circle` frame.
- **`groups`** is the generic grouping list; the engine reads `payload.groups || payload.germ_layers`.
- **`side`** generalizes `eye`; the engine reads `zone.side || zone.eye`.
- **`geometry.type`**: `"point"` for the ear; iris zones (radial/sector, no `geometry`) are treated as `"sector"`.
- **`anchors`** with `template` coords generalize the overlay (see §4).

### 2. Store — generalized validation (`bodymap_store`)

`validate_zone` dispatches on geometry, additively:
- If the zone has `radial` and `sector` (and no `geometry` / `geometry.type == "sector"`): validate as today (the iris path — unchanged rules).
- If `geometry.type == "point"`: require `geometry.x`, `geometry.y` numbers in `[0,1]`; require `anatomy`, `meaning_standard`, and a laterality (`side` or `eye`) and a grouping (`group` or `germ_layer`).
- `build_payload` continues to drop invalid zones and add `meaning_display`; it passes through `outline`, `groups`, `anchors`, `reference_frame` when present.
- The `ear` system is registered in `SYSTEMS` (`data/bodymap-ear.json`, persistent-disk reseed like the others). Iris/sclera validation and payloads are unaffected.

### 3. Client — generalized rendering (`static/body-map.js`)

`renderChart` dispatches on the loaded system's reference surface, defaulting to iris:
- **Reference surface:** `unit_circle` → draw germ-layer rings (existing). `ear_outline` → draw the `outline` SVG path (with light region guides) instead of rings.
- **Coordinate map:** `refToScreen` becomes frame-aware — `unit_circle` uses the existing `chartR` mapping; `ear_outline` maps normalized `[0,1]` to the canvas box.
- **Zone rendering:** a `sector` zone → existing `arcSectorPoints` path; a `point` zone → a labeled dot (small circle marker) at its mapped coordinate, clickable exactly like a sector (selects + opens the panel).
- **Grouping toggles:** built from `payload.groups || payload.germ_layers`; filtering a zone reads `zone.group || zone.germ_layer`.
- **Laterality:** the eye/side `<select>` relabels per system (Eye: Right/Left for iris; Side: Left/Right for ear) and filters on `zone.side || zone.eye`.
- **`selectZone` / `currentZones` / the panel** are geometry-agnostic already (they key on `id`, laterality, grouping) and need only the `side||eye`, `group||germ_layer` generalization.

### 4. Client — generalized overlay warp (`static/body-map.js`)

The overlay becomes anchor-driven:
- If the loaded payload defines `anchors` (the ear): show that system's anchor prompts; each tapped anchor has a known `template` coordinate. Fit a **similarity transform** (translation + rotation + uniform scale) from the anchor correspondences (two anchors fully determine it; a third refines/validates). Apply it to map each zone's normalized template coordinate onto the photo. This is the general form.
- If the payload defines no `anchors` (the iris): keep the existing bespoke pupil/limbus/twelve construction unchanged (additive — zero change to the live iris overlay).
- The disclaimer ("This is a visual approximation, not a diagnosis.") is unchanged and applies to all systems.

### 5. Client — system dropdown + focus params

- `static/body-map.html` gains an **Ear** option in `#bm-system`.
- `applyFocusFromURL` generalizes: `?side=` (accepting the current `?eye=` too), `?group=` (accepting `?layer=`), `?zone=` — all reusing the generalized select/highlight paths, still never throwing on unknown params. (The Atlas cluster map can later target ear zones; not required for this slice.)

---

## Scope

### In scope
- Store: generalized `validate_zone` (sector + point), `ear` system registration, payload pass-through of `outline`/`groups`/`anchors`/`reference_frame`. Unit tests for point-zone validation and that the ear seed validates.
- Seed `data/bodymap-ear.json`: a drawn ear outline path + a starter set of well-known auricular points (Shen Men, Point Zero, Sympathetic, Kidney, Liver, Lung, Heart, Stomach, plus a lobe/head point and a helix point), grouped by region, left side.
- Client: reference-surface dispatch (ear outline vs iris rings), point-zone rendering (dots), grouping generalization, `side`/`eye` laterality relabel, anchor-driven overlay for the ear, Ear in the dropdown, generalized focus params.
- The **iris renders and behaves identically** (hard requirement) — verified by re-rendering the iris after the change.

### Explicitly deferred
- Right-ear side (ships left first); full auricular point set (starter subset now, Glen expands via admin).
- `polygon`/`path` geometry types (schema leaves room; not built).
- Atlas cluster-map targets pointing at ear zones; reverse links.
- Retro-migrating the iris to the explicit typed schema (additive keeps it as-is).
- ML auto-anchoring; dim/zoom focus polish.

## Success criteria
- Selecting **Ear** in the system dropdown draws the ear outline with the auricular points as labeled dots; clicking a point opens its meanings panel; the region toggles filter points by anatomical region; the side selector is labeled for the ear.
- Uploading an ear photo and tapping the three ear landmarks warps the points onto the user's ear (photo stays in-browser, same no-PHI guarantee).
- `/body-map?system=ear&zone=ear-L-shenmen` opens the ear centered on Shen Men; an unknown param falls back to the default view without error.
- The **iris** system renders and behaves exactly as before (reference chart, overlay, focus params, admin) — no visual or behavioral change.
- `pytest tests/test_bodymap_store.py` passes, including new point-zone tests and an ear-seed-validity test.
