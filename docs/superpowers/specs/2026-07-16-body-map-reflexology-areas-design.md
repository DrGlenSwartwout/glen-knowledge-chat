# Body Map — Reflexology Areas: polygon geometry + bilateral mirror + admin drawing tool

**Date:** 2026-07-16
**Status:** Approved design, pre-implementation
**App:** deploy-chat (`glen-knowledge-chat`)
**Builds on:** the shipped Body Map (iris/sclera/ear/foot) and an in-branch rendering fix (smooth-curve outlines via `transformPathD`, in-place labels, region colors).

---

## Why

The shipped foot map is wrong at the root: reflexology is **areas**, not points; the two soles must be **bilateral mirrors** with the visceral **asymmetries** as the only exceptions; and its coordinates were guessed from memory. This spec fixes the process, not just the output: model reflexology as mirrored polygon areas, and author the geometry by **tracing a real reference chart** in a drawing tool rather than eyeballing numbers.

## Locked decisions (from the design conversation)

1. **Areas, not points.** Reflexology zones become a new `polygon` geometry type (a closed boundary of normalized points), rendered as filled, labeled regions. The typed engine already dispatches on geometry: `sector` (iris), `point` (ear/acupoints), now `polygon` (reflexology).
2. **Bilateral mirror is the model.** A zone is authored **once** on a canonical side; the contralateral side is its mirror (`x → 1 − x`). The named visceral-asymmetry set is the only departure, placed per true side: **liver, gallbladder, ileocecal, appendix, ascending colon on the right; heart, spleen, stomach, pancreas, splenic-flexure/descending/sigmoid colon on the left; transverse colon crossing.** Correctness is visible: the mirror holds or it doesn't.
3. **Authoring by tracing, in an admin tool (option B).** An admin drawing surface loads a **reference chart as a faint underlay** (client-side only, never shipped or stored — the licensed stock chart is reference, not product), the author clicks a zone's boundary and tags it, and the resulting **original polygon** is saved. Whoever draws is tracing the real chart, so geometry is accurate by construction, and Glen (the expert) places the ones that matter.
4. **Reference set chosen:** foot = Vecteezy #77159447 (area-based, correctly lateralized, labeled); hand and ear references collected for later. The app ships only our own polygons — no license entanglement.

---

## Architecture

### 1. Geometry — the `polygon` type (`bodymap_store` + `body-map.js`)

- **Zone shape:** `geometry: {"type": "polygon", "points": [[x,y], [x,y], ...]}` in normalized `[0,1]`, plus a `label_at: {x,y}` (where the label sits) or derived centroid. Plus `bilateral: true|false` and, for bilateral zones, a canonical `side` the author drew on.
- **Store validation** (`validate_zone`): additive branch for `polygon` — require `points` (≥3 pairs, each in `[0,1]`), `anatomy`, `meaning_standard`, a grouping, and either `bilateral: true` or an explicit `side`. Iris/ear/existing zones unchanged.
- **Renderer** (`renderChart`): a `polygon` zone renders as an SVG `path` (closed, filled with the region color at low opacity, stroked), clickable, with the in-place label at its centroid/`label_at`. Reuses the label + region-color work already in the branch.

### 2. Bilateral mirror rendering

- The payload/renderer treats a `bilateral` zone as present on **both** sides: when viewing the side it was drawn on, render at its points; when viewing the other side, render the **mirror** (`x → 1 − x`) of every point (and the label). A non-bilateral zone (asymmetric organ) renders only on its `side`.
- This lives in the renderer (author once, mirror at display), so the seed stays half the size and the symmetry is enforced by construction rather than duplicated data.

### 3. Admin drawing tool (`/admin/body-map/draw`)

- A new console-key-gated admin page. Controls: pick **system** (foot) and **side** (canonical), **load a reference image** (file input → faint client-side underlay, exactly like the photo overlay — stays in the browser, never uploaded).
- **Draw:** click to drop boundary points on the underlay; the in-progress polygon shows; double-click / close to finish. Then a small form tags it: **anatomy**, **group** (from the system's groups), **bilateral?** (checkbox), **meaning_standard**.
- **Save:** POST the tagged polygon to a new admin endpoint that appends/updates a zone (with `geometry.polygon`) in the system's seed JSON on the persistent disk — an extension of the existing overlay-edit mechanism (`set_zone_overlay` → a broader `upsert_zone`). List / edit / delete existing zones in the same tool.
- The underlay image is only a tracing aid; nothing about it is saved. Only the polygon coordinates + tags persist.

### 4. Re-author the foot

- Using the tool over the #77159447 reference, replace the foot's point zones with **polygon areas**, authored once per canonical side + the asymmetric organs, verified against the reference's lateralization. The point-based foot seed is retired for the foot (the ear stays points — acupoints are genuinely points).

---

## Scope

### In scope
- `polygon` geometry: store validation + payload passthrough + renderer (filled labeled regions).
- Bilateral mirror rendering (author once, mirror the contralateral side; asymmetry exceptions).
- Admin drawing tool: reference underlay, click-to-draw, tag, save/list/edit/delete zones with geometry; a store `upsert_zone` (+ delete) with console-key routes.
- Re-author the foot as mirrored polygon areas from the reference.

### Explicitly deferred
- Hand and face systems (the tool + polygon type make them straightforward once the foot proves it).
- Re-authoring the ear as points is unnecessary (acupoints are points); iris/sclera stay sectors.
- Auto-suggesting zones from an image (no ML tracing); per-vertex editing polish; overlapping-zone z-order controls.

## Success criteria
- Selecting **Foot** shows the reflex map as **filled, labeled areas** on a recognizable sole, colored by region; the left and right soles are **mirror images** except for the correctly-placed asymmetric organs (liver/gallbladder right; heart/spleen left).
- In `/admin/body-map/draw`, an author can load the reference chart as an underlay, trace a zone, tag it (anatomy/group/bilateral/meaning), save it, and see it render on the map — with the reference image never leaving the browser or being stored.
- A bilateral zone drawn once appears mirrored on the other sole automatically; an asymmetric organ appears only on its side.
- The iris, sclera, and ear are unchanged; store tests pass including polygon validation and mirror behavior.
