# Body Map ← Atlas Deep-Link — Design Spec

**Date:** 2026-07-16
**Status:** Approved design, pre-implementation
**App:** deploy-chat (`glen-knowledge-chat`)
**Builds on:** Body Map v1 (PR #945, merged) and the Knowledge Atlas (`atlas_*.py`, `atlas.js`, `data/atlas-concepts.json`)

---

## Goal

Let a relevant Knowledge Atlas topic link to a Body Map view **centered** on the body location it relates to. Clicking the link opens `/body-map`, preselects the right system and eye, highlights the target zone(s), and opens the meanings panel.

## Locked decisions (from brainstorm)

1. **Mapping = both (C):** an automatic cluster→region default with an optional per-concept override for precision. A concept shows the link only when it resolves to a real body location; systemic clusters get no link (this is what scopes it to "relevant topics").
2. **"Centered" = preselect + highlight + panel-open (behavior 1):** open the right system + eye, highlight the target zone(s), open the panel for the primary zone. No dim/zoom in this slice (deferred polish).
3. **One direction:** Atlas concept → Body Map (per Glen's ask). Reverse (Body Map → Atlas) is deferred.
4. **Cluster map lives in `bodymap_store`** as a small, testable constant; per-concept overrides carry the precision (same division of labor as the `meaning_glen` overlay).

---

## Architecture

Three parts:

1. **Body Map page — focus params.** `static/body-map.js` reads URL query params on load and applies a focus after the system data loads.
2. **Resolver — `bodymap_store.resolve_atlas_target(concept)`.** A pure, dependency-free function returning the deep-link target for an Atlas concept (or `None`). Unit-tested in isolation.
3. **Atlas link render.** Each resolvable concept gets a "View on the Body Map" link to the computed deep URL, rendered where concept links already appear.

### 1. Body Map focus params (`static/body-map.js`)

On page load, after `loadSystem()` resolves, read `location.search`:

- `?system=<iridology|sclerology>` — select that system (default iridology). Invalid → ignore, keep default.
- `?eye=<right|left>` — select that eye (default right).
- `?zone=<zone_id>` — after render, select that zone (highlight + open its panel), the same effect as clicking it. If the zone belongs to the other eye, switch the eye selector to match first.
- `?layer=<germ_layer_id>` — highlight all zones in that germ layer: check the matching germ-layer toggle (so only that layer's zones show) and open the panel on the first of them. `zone` takes precedence over `layer` if both are present.

Rules:
- Applying focus reuses the existing `selectZone` / eye-select / layer-toggle code paths — no new rendering logic, just programmatic invocation.
- If a param names something not in the payload (unknown zone/layer/system), ignore it silently and leave the default view. Never throw.
- Focus applies only on initial load; it does not fight later user interaction.

### 2. Resolver (`bodymap_store.resolve_atlas_target(concept)`)

```
resolve_atlas_target(concept) -> dict | None
```

Resolution order:
1. **Per-concept override:** if `concept` has a truthy `body_map` field of shape `{"system": ..., "zone": ...}` or `{"system": ..., "layer": ...}`, return it (validated).
2. **Cluster map:** else look up `concept["cluster"]` in `ATLAS_CLUSTER_MAP`; if present, return its target.
3. **Else `None`** (no link).

Target shape (one of):
- `{"system": "iridology", "zone": "iris-R-liver"}`
- `{"system": "iridology", "layer": "mesoderm"}`
- `{"system": "iridology"}` (system only)

A companion `atlas_target_url(target) -> str` builds the querystring, e.g. `/body-map?system=iridology&zone=iris-R-liver`.

**`ATLAS_CLUSTER_MAP` (starter — Glen refines):**

| cluster | target |
|---|---|
| `gut-digestive` | `{system: iridology, zone: iris-R-intestines}` |
| `brain-nervous` | `{system: iridology, zone: iris-R-brain}` |
| `circulation-cardio` | `{system: iridology, zone: iris-L-heart}` |
| `structural-musculoskeletal` | `{system: iridology, layer: mesoderm}` |
| `detox-drainage` | `{system: iridology, zone: iris-R-liver}` |
| `metabolic-bloodsugar` | `{system: iridology, zone: iris-R-liver}` |
| `immune` | `{system: iridology, zone: iris-R-intestines}` |
| `eye-health` | `{system: iridology}` |

Clusters not in the map (`hormones-endocrine`, `antioxidants`, `minerals`, `foundations-terrain`, `adaptogens`, `energetic-medicine`, `other`, …) resolve to `None` → no link. Zone ids reference the shipped seed (`data/bodymap-iridology.json`); the map is validated by a test that every referenced zone/layer exists in the seed.

### 3. Atlas link render

The Atlas concept payload (or the concept detail rendering in `atlas.js`) gains an optional `body_map_url`. Server-side is preferred so the resolver stays in Python and testable: where the Atlas graph/concepts are served, attach `body_map_url = atlas_target_url(resolve_atlas_target(c))` when the resolver returns a target. `atlas.js` renders a "View on the Body Map" link (styled like the existing concept links) when `body_map_url` is present.

If wiring the server payload proves invasive, the fallback is to ship `ATLAS_CLUSTER_MAP` to the client as a small JSON endpoint and resolve in `atlas.js` — but the Python resolver remains the source of truth and is tested regardless.

---

## Scope

### In scope
- `resolve_atlas_target` + `atlas_target_url` + `ATLAS_CLUSTER_MAP` in `bodymap_store`, with unit tests (override precedence, cluster hit, no-match `None`, url building, seed-reference validity).
- Body Map focus params (`system`, `eye`, `zone`, `layer`) applied on load, reusing existing select/highlight paths.
- Atlas concept link ("View on the Body Map") on resolvable concepts.

### Explicitly deferred
- Dim/zoom "focus" polish (brainstorm option 2).
- Reverse direction (Body Map zone → Atlas cluster/concepts).
- New Body Map zones or systems (uses the 9 shipped seed zones).
- Bulk authoring of per-concept overrides beyond a small starter set (e.g., the ED# drivers) — overrides accrete over time.

## Success criteria
- `/body-map?system=iridology&zone=iris-R-liver` opens with the liver zone highlighted and its panel open; `?layer=mesoderm` highlights the mesoderm zones; an unknown param leaves the default view without error.
- `resolve_atlas_target` returns the override when present, else the cluster target, else `None`; every zone/layer it can return exists in the seed.
- A concept in a mapped cluster (e.g., a `brain-nervous` formulation) renders a working "View on the Body Map" link; a concept in an unmapped cluster (e.g., `antioxidants`) renders none.
