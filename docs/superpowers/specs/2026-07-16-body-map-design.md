# Body Map — Design Spec (v1)

**Date:** 2026-07-16
**Status:** Approved design, pre-implementation
**App:** deploy-chat (`glen-knowledge-chat`)
**Route:** `/body-map` (public page, alongside the practitioner-finder)

---

## The big idea

An interactive atlas of the body's mapped systems — every tradition that reads the
body through a surface (acupuncture meridians/vessels, EAV points, reflex zones of
the ear/iris/sclera/face/hands/feet, dermatomes/neurotomes, lymph pathways). Each
system is a data-backed layer of points/zones and their meanings. The signature
feature: the user uploads a photo of a body region and sees the relevant points and
zones projected onto their own anatomy.

This spec covers **v1 only**, which proves the whole engine end-to-end on the system
Glen knows best — the **eye (iridology + sclerology)** — and is built so every future
system is "just more data" poured into the same machine.

## Locked decisions (from brainstorm)

1. **Phasing:** Build the public atlas engine first; personalize into the portal
   later (option C). The hard part — a structured zones-and-meanings dataset plus a
   photo-overlay renderer — is shared by both, so building the public atlas first
   reuses everything.
2. **Overlay mechanic:** Manual landmark anchoring with a warp-to-fit. No per-part ML
   in v1. It is the only approach that works reliably across every future system
   without a separate trained model, it is buildable today, and it is honest about
   being an approximation (which matters clinically).
3. **v1 system:** Iris + sclerology (the eye). Glen's deepest authority, the cleanest
   overlay to build the anchor-and-warp mechanic on (a close-up single region with
   unambiguous anchors), and the natural wedge into the eye-health ecosystem.
4. **Content source:** Standard published iris topography as the scaffold, annotated
   and corrected with Glen's clinical overlay over time. The **embryological germ
   layers** are the radial organizing axis of the map (endoderm/mesoderm/ectoderm as
   the rings from pupil outward), which pre-wires the future "peel back the layers"
   feature.

## Naming note

The token **"atlas" is already taken** in this codebase — there is a "Knowledge
Atlas" concept-graph feature (`atlas_*.py`). This feature is therefore named **Body
Map** everywhere (route `/body-map`, module prefix `bodymap_`, data file
`bodymap-iris.json`) to avoid collision. It borrows the Knowledge Atlas *data
pattern* (see Data model) but shares no code or namespace with it.

---

## Architecture

A single public page in the existing deploy-chat Flask app — same app, same nav,
same deploy as the practitioner-finder. Three parts:

- **Data layer** — `bodymap-<system>.json` files (v1: `bodymap-iris.json`) holding the
  structured zones-and-meanings dataset. Following the Knowledge Atlas pattern: the
  git-committed seed ships in `data/`, and a mutable curated copy lives on the
  persistent disk (`DATA_DIR=/data` on Render) so Glen's annotations survive
  redeploys. An admin surface lets Glen edit/approve zone content without a
  code deploy. No Pinecone or Flask dependency in the store module itself.
- **Server** — a small `bodymap_*` module registering the `/body-map` route and a
  read API that serves the approved zone dataset as JSON to the page. No photo ever
  reaches the server in v1 (see Privacy).
- **Client** — the page. Renders the standard reference chart from the zone geometry,
  handles photo upload + anchor placement + warp + overlay, and shows the meanings
  panel. All overlay math and rendering happen in-browser (SVG over the image).

### Data model (`bodymap-iris.json`)

The value of the whole feature lives here. Shape:

```
{
  "system": "iridology",              // v1 ships "iridology" and "sclerology"
  "reference_frame": "unit_circle",   // normalized: pupil at origin, iris edge at r=1
  "germ_layers": [                    // the radial organizing spine
    { "id": "endoderm", "r_inner": 0.0, "r_outer": 0.33, "label": "..." },
    { "id": "mesoderm", "r_inner": 0.33, "r_outer": 0.66, "label": "..." },
    { "id": "ectoderm", "r_inner": 0.66, "r_outer": 1.0,  "label": "..." }
  ],
  "zones": [
    {
      "id": "iris-R-03-liver",
      "eye": "right",                 // right | left
      "germ_layer": "endoderm",
      "radial": { "r_inner": 0.10, "r_outer": 0.30 },
      "sector": { "start_deg": 75, "end_deg": 105 },   // clock geometry
      "anatomy": "liver",
      "meaning_standard": "…scaffold text from published topography…",
      "meaning_glen": "",             // Glen's clinical overlay slot (fills over time)
      "layers": {                     // placeholders for future data layers
        "embryological_depth": null,
        "stress_affirmation": null,   // Louise-Hay-style zone→pattern→affirmation
        "touch_for_health": null      // muscle↔meridian↔organ association
      }
    }
    // …
  ]
}
```

Key properties:
- **Geometry is normalized**, defined once against the standard chart's unit circle.
  The warp maps this frame onto any uploaded photo, so zone definitions are
  photo-independent and reusable.
- **`meaning_standard` is the scaffold; `meaning_glen` is the asset.** Both render;
  Glen's overlay takes visual precedence once filled.
- **Germ layer is a first-class field**, so the rings are both an organizing structure
  and a toggleable display layer from day one.
- **`layers` placeholders exist but are unused in v1** — they reserve the shape so
  future systems/layers add data without a schema migration.

### The overlay mechanic (3-anchor similarity warp)

1. User uploads an eye close-up (stays in the browser).
2. User drops **3 anchors**: (a) pupil center, (b) a point on the iris/limbus edge,
   (c) the 12-o'clock reference on the iris edge.
3. From the anchors, compute a **similarity transform** — translation (pupil center),
   scale (pupil-center → limbus distance = r=1), rotation (12-o'clock direction).
4. Apply the transform to the normalized zone geometry → render the zones as SVG
   paths overlaid on the photo.
5. Tapping a rendered zone opens the meanings panel for that zone.
6. A persistent "This is a visual approximation, not a diagnosis" note stays on screen.

A similarity transform (not a full homography) is the right v1 fidelity: it needs
only 3 anchors, is numerically trivial, and matches how a roughly frontal eye photo
relates to the standard chart. Perspective/elliptical correction is a later refinement,
not a v1 need.

### The page UI

Two synced views over the same zone data:
- **Reference chart view** — the standard iris map rendered from the geometry, no
  photo required. This is the browsable SEO/authority surface: explore zones, read
  meanings, toggle germ-layer rings.
- **Your-photo overlay view** — the uploaded photo with the warped overlay. The hook.

Shared controls: **left/right eye** switch, **germ-layer ring** toggles (the "peel
back the layers" seed — v1 shows/hides rings; true depth-peeling is deferred), and the
**meanings side panel** driven by the currently selected zone.

### Privacy (v1)

**The uploaded photo never leaves the browser.** No upload to the server, nothing
stored, overlay rendered entirely client-side. This keeps v1 out of PHI-storage
territory and is the honest default. Server-side photo storage arrives only later,
deliberately, inside the logged-in portal under explicit consent.

---

## Scope

### In scope (v1)
- `/body-map` public page in deploy-chat.
- Iris + sclerology systems only.
- `bodymap-iris.json` (+ sclerology) seed dataset: geometry + `meaning_standard`
  scaffold + empty `meaning_glen` slots, organized by germ layer.
- Admin surface for Glen to edit/approve zone content (Knowledge Atlas pattern).
- 3-anchor similarity warp, client-side overlay rendering.
- Reference-chart view + your-photo overlay view, eye switch, germ-layer toggles,
  meanings panel.
- Client-side-only photo handling (no server upload, no storage).

### Explicitly deferred
- All other systems: ear/auricular, foot/hand reflexology, face, meridians/vessels,
  EAV points, lymph pathways, dermatomes/neurotomes.
- ML auto-anchoring (auto-detect pupil/limbus/landmarks).
- Embryological **depth-peeling** interaction (v1 shows rings; peeling is later).
- Stress/affirmation layer (Louise-Hay-style) and Touch for Health layer.
- Any portal personalization (client photo + E4L/Biofield findings on the map).
- Perspective/homography correction of tilted photos.

## Roadmap (same engine, more data)
- **New systems** = a new `bodymap-<system>.json` with its own anchor set and geometry;
  the renderer and admin surface are unchanged.
- **Metaphysical + Touch-for-Health layers** = fill the `layers.*` fields already on
  every zone; add panel sections to display them.
- **Depth-peeling** = animate/isolate germ-layer rings using geometry already present.
- **Portal personalization** = reuse the whole renderer with the logged-in client's
  photo + findings, under consent, with server-side storage added deliberately.

## Success criteria (v1)
- A visitor can explore the iris reference chart and read zone meanings with no photo.
- A visitor can upload an eye photo, place 3 anchors, and see the zone overlay warped
  onto their own iris, then tap zones to read meanings.
- Glen can edit a zone's `meaning_glen` and have it appear live without a code deploy.
- No eye photo is ever transmitted to or stored on the server in v1.
