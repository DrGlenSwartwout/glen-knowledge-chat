# Body Map ← Atlas Deep-Link Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Let a relevant Knowledge Atlas concept link to a `/body-map` view centered (preselect + highlight + panel) on the body location it relates to.

**Architecture:** A pure resolver in `bodymap_store.py` maps an Atlas concept → a Body Map deep-link target (per-concept override → cluster map → none). The `/atlas/data` route enriches each concept with a `body_map_url`. `atlas.js` renders a "View on the Body Map" link. `body-map.js` reads `?system/eye/zone/layer` on load and applies the focus by reusing existing select/highlight paths.

**Tech Stack:** Python 3 + Flask, vanilla JS + inline SVG, pytest for the store, headless Chrome for client render-verification.

## Global Constraints

- **Work inside the worktree** `/tmp/wt-deploy-chat-f33b5ea0` on branch `sess/f33b5ea0-atlas` (already created off the merged `origin/main`, so Body Map v1 is present). All edits and commits happen there.
- **Never run the bare full test suite** (a bare `pytest` sends real email). Run ONLY `pytest tests/test_bodymap_store.py -v`.
- **App-import / route verification harness:** `doppler run -p remedy-match -c prd -- env -u DATA_DIR python3 -c "..."`. `-p remedy-match -c prd` selects the config; `env -u DATA_DIR` strips the injected `/data` path so the app import does not start the cron scheduler and stores fall back to the repo/worktree `data/`. A bare `doppler run` fails; importing with DATA_DIR set starts real cron jobs. Use `python3`, not `python`.
- **`bodymap_store.py` keeps zero Flask/Pinecone imports** (pure Python; `urllib.parse` is fine).
- **Body Map focus must never throw** on an unknown/invalid param — ignore it and keep the default view.
- **Reuse existing paths** in `body-map.js` (`selectZone`, the eye `<select>`, the germ-layer checkbox, `renderChart`, `currentZones`, `state`) — add no new rendering logic.
- **Naming:** `body-map`/`bodymap` only. **Copy rules:** no em dashes, no ALL CAPS, no "Hook:" label.

---

## File Structure

**Modify:**
- `bodymap_store.py` — add `ATLAS_CLUSTER_MAP`, `resolve_atlas_target(concept)`, `atlas_target_url(target)`.
- `tests/test_bodymap_store.py` — resolver unit tests.
- `app.py` — enrich `/atlas/data` concepts with `body_map_url`.
- `static/body-map.js` — read `?system/eye/zone/layer` on load and apply focus.
- `static/atlas.js` — render the "View on the Body Map" link in the concept drawer.

---

### Task 1: Resolver + cluster map (`bodymap_store`)

**Files:**
- Modify: `bodymap_store.py`
- Test: `tests/test_bodymap_store.py`

**Interfaces:**
- Consumes: `load_map` / `SYSTEMS` (existing) for the seed-validity test.
- Produces: `ATLAS_CLUSTER_MAP` (dict), `resolve_atlas_target(concept) -> dict|None`, `atlas_target_url(target) -> str`.

- [ ] **Step 1: Write the failing tests** (append to `tests/test_bodymap_store.py`)

```python
def test_resolve_atlas_target_override_wins():
    c = {"cluster": "brain-nervous", "body_map": {"system": "iridology", "zone": "iris-R-liver"}}
    assert bodymap_store.resolve_atlas_target(c) == {"system": "iridology", "zone": "iris-R-liver"}


def test_resolve_atlas_target_cluster_hit():
    c = {"cluster": "brain-nervous"}
    assert bodymap_store.resolve_atlas_target(c) == {"system": "iridology", "zone": "iris-R-brain"}


def test_resolve_atlas_target_unmapped_is_none():
    assert bodymap_store.resolve_atlas_target({"cluster": "antioxidants"}) is None
    assert bodymap_store.resolve_atlas_target({}) is None
    assert bodymap_store.resolve_atlas_target("nope") is None


def test_resolve_atlas_target_ignores_override_without_system():
    c = {"cluster": "brain-nervous", "body_map": {"zone": "iris-R-liver"}}
    assert bodymap_store.resolve_atlas_target(c) == {"system": "iridology", "zone": "iris-R-brain"}


def test_atlas_target_url_variants():
    assert bodymap_store.atlas_target_url({"system": "iridology", "zone": "iris-R-liver"}) == "/body-map?system=iridology&zone=iris-R-liver"
    assert bodymap_store.atlas_target_url({"system": "iridology", "layer": "mesoderm"}) == "/body-map?system=iridology&layer=mesoderm"
    assert bodymap_store.atlas_target_url({"system": "iridology"}) == "/body-map?system=iridology"
    assert bodymap_store.atlas_target_url(None) == ""
    # zone takes precedence over layer when both present
    assert bodymap_store.atlas_target_url({"system": "iridology", "zone": "iris-R-liver", "layer": "mesoderm"}) == "/body-map?system=iridology&zone=iris-R-liver"


def test_cluster_map_targets_exist_in_seed():
    import pathlib
    repo = pathlib.Path(bodymap_store.__file__).resolve().parent / "data"
    seed = json.loads((repo / "bodymap-iridology.json").read_text())
    zones = {z["id"] for z in seed["zones"]}
    layers = {g["id"] for g in seed["germ_layers"]}
    for cluster, tgt in bodymap_store.ATLAS_CLUSTER_MAP.items():
        assert tgt.get("system") == "iridology", cluster
        if "zone" in tgt:
            assert tgt["zone"] in zones, f"{cluster} -> unknown zone {tgt['zone']}"
        if "layer" in tgt:
            assert tgt["layer"] in layers, f"{cluster} -> unknown layer {tgt['layer']}"
```

- [ ] **Step 2: Run to verify they fail**

Run: `pytest tests/test_bodymap_store.py -k atlas_target -v`  (plus `-k cluster_map`)
Expected: FAIL with `AttributeError: module 'bodymap_store' has no attribute 'resolve_atlas_target'`

- [ ] **Step 3: Implement** (append to `bodymap_store.py`)

```python
ATLAS_CLUSTER_MAP = {
    "gut-digestive": {"system": "iridology", "zone": "iris-R-intestines"},
    "brain-nervous": {"system": "iridology", "zone": "iris-R-brain"},
    "circulation-cardio": {"system": "iridology", "zone": "iris-L-heart"},
    "structural-musculoskeletal": {"system": "iridology", "layer": "mesoderm"},
    "detox-drainage": {"system": "iridology", "zone": "iris-R-liver"},
    "metabolic-bloodsugar": {"system": "iridology", "zone": "iris-R-liver"},
    "immune": {"system": "iridology", "zone": "iris-R-intestines"},
    "eye-health": {"system": "iridology"},
}


def resolve_atlas_target(concept):
    """Body Map deep-link target for an Atlas concept, or None.
    Order: per-concept override (concept['body_map'] with a 'system') -> cluster map -> None."""
    if not isinstance(concept, dict):
        return None
    override = concept.get("body_map")
    if isinstance(override, dict) and override.get("system"):
        return override
    return ATLAS_CLUSTER_MAP.get(concept.get("cluster"))


def atlas_target_url(target):
    """Build the /body-map deep-link URL for a target dict. Returns '' when falsy/invalid."""
    if not isinstance(target, dict) or not target.get("system"):
        return ""
    from urllib.parse import urlencode
    params = {"system": target["system"]}
    if target.get("eye"):
        params["eye"] = target["eye"]
    if target.get("zone"):
        params["zone"] = target["zone"]
    elif target.get("layer"):
        params["layer"] = target["layer"]
    return "/body-map?" + urlencode(params)
```

- [ ] **Step 4: Run to verify pass**

Run: `pytest tests/test_bodymap_store.py -v`
Expected: PASS (all prior tests + 6 new).

- [ ] **Step 5: Commit**

```bash
git add bodymap_store.py tests/test_bodymap_store.py
git commit -m "feat(bodymap): atlas concept -> body-map target resolver + cluster map"
```

---

### Task 2: Body Map focus params (`static/body-map.js`)

**Files:**
- Modify: `static/body-map.js`

**Interfaces:**
- Consumes existing (all inside the IIFE): `state`, `loadSystem`, `selectZone`, `renderChart`, `currentZones`, `__bmSelfCheck`, DOM ids `bm-system`, `bm-eye`, `bml-<layer>`.
- Produces: `applyFocusFromURL(params)`, and a modified `wire()` that honors `?system` on initial load and calls `applyFocusFromURL` after the first `loadSystem` resolves.

- [ ] **Step 1: Add `applyFocusFromURL` inside the IIFE** (place it near `loadSystem`, before `window.__bm =`)

```javascript
  function applyFocusFromURL(params) {
    if (!state.payload) return;
    const eye = params.get("eye");
    if (eye === "right" || eye === "left") {
      state.eye = eye; document.getElementById("bm-eye").value = eye; renderChart();
    }
    const zoneId = params.get("zone");
    if (zoneId) {
      const z = (state.payload.zones || []).find(x => x.id === zoneId);
      if (z) {
        if (z.eye !== state.eye) {
          state.eye = z.eye; document.getElementById("bm-eye").value = z.eye; renderChart();
        }
        selectZone(z);
        return;
      }
    }
    const layerId = params.get("layer");
    if (layerId) {
      const layer = (state.payload.germ_layers || []).find(g => g.id === layerId);
      if (layer) {
        state.activeLayers.clear(); state.activeLayers.add(layerId);
        const cb = document.getElementById("bml-" + layerId);
        if (cb) cb.checked = true;
        renderChart();
        const first = currentZones()[0];
        if (first) selectZone(first);
      }
    }
  }
```

- [ ] **Step 2: Modify `wire()`** to honor `?system` on initial load and apply focus after load

Replace the body of `wire()` so it reads params, sets the initial system on the `<select>`, and applies focus after `loadSystem` resolves. The existing `wire()` ends with `wireOverlay();` (from v1) and `loadSystem("iridology").then(__bmSelfCheck);`. Change the final two lines to:

```javascript
    wireOverlay();
    const params = new URLSearchParams(location.search);
    const sys = params.get("system");
    const initialSystem = (sys === "iridology" || sys === "sclerology") ? sys : "iridology";
    document.getElementById("bm-system").value = initialSystem;
    loadSystem(initialSystem).then(function () { applyFocusFromURL(params); __bmSelfCheck(); });
```

(Leave the earlier lines of `wire()` — the `bm-system` change listener and `bm-eye` change listener — unchanged.)

- [ ] **Step 3: Syntax check**

Run: `node --check static/body-map.js`
Expected: exit 0, no output.

- [ ] **Step 4: Commit**

```bash
git add static/body-map.js
git commit -m "feat(bodymap): honor ?system/eye/zone/layer focus params on load"
```

(The controller render-verifies `?zone=`, `?layer=`, `?eye=`, and an unknown param in a real browser after this task.)

---

### Task 3: Enrich `/atlas/data` with `body_map_url` (`app.py`)

**Files:**
- Modify: `app.py` (the `atlas_data` route)

**Interfaces:**
- Consumes: `atlas_store.build_graph`, `bodymap_store.resolve_atlas_target`, `bodymap_store.atlas_target_url` (both modules already imported in app.py).
- Produces: each `/atlas/data` concept gains `body_map_url` when it resolves to a target.

- [ ] **Step 1: Modify the `atlas_data` route**

Current:
```python
@app.route("/atlas/data")
def atlas_data():
    return jsonify(atlas_store.build_graph())
```
Replace with:
```python
@app.route("/atlas/data")
def atlas_data():
    graph = atlas_store.build_graph()
    for c in graph.get("concepts", []):
        url = bodymap_store.atlas_target_url(bodymap_store.resolve_atlas_target(c))
        if url:
            c["body_map_url"] = url
    return jsonify(graph)
```

- [ ] **Step 2: Verify enrichment under the app-import harness**

Create `/tmp/bm_atlas_check.sh`:
```bash
#!/bin/bash
cd /tmp/wt-deploy-chat-f33b5ea0 && doppler run -p remedy-match -c prd -- env -u DATA_DIR python3 -c "
import app
c = app.app.test_client()
d = c.get('/atlas/data').get_json()
by = {x['id']: x for x in d['concepts']}
# a brain-nervous concept should be enriched; an antioxidants concept should not
mapped = [x for x in d['concepts'] if x.get('cluster')=='brain-nervous']
unmapped = [x for x in d['concepts'] if x.get('cluster')=='antioxidants']
print('mapped sample has url:', bool(mapped) and 'body_map_url' in mapped[0], mapped[0].get('body_map_url') if mapped else None)
print('unmapped has no url:', bool(unmapped) and 'body_map_url' not in unmapped[0])
"
```
Run: `bash /tmp/bm_atlas_check.sh`
Expected: `mapped sample has url: True /body-map?system=iridology&zone=iris-R-brain` and `unmapped has no url: True`.

- [ ] **Step 3: Commit**

```bash
git add app.py
git commit -m "feat(atlas): enrich /atlas/data concepts with body_map_url"
```

---

### Task 4: Atlas drawer link (`static/atlas.js`)

**Files:**
- Modify: `static/atlas.js` (`renderDrawer`)

**Interfaces:**
- Consumes: `c.body_map_url` from the enriched payload (Task 3).
- Produces: a "View on the Body Map" link in the concept drawer when `body_map_url` is present.

- [ ] **Step 1: Modify `renderDrawer`**

The current `renderDrawer` builds `links` from `c.links` then sets `this.drawer.innerHTML`. After the `links` variable is built and before the `this.drawer.innerHTML = ...` line, add:
```javascript
    var bmLink = c.body_map_url
      ? '<a href="' + c.body_map_url + '">◉ View on the Body Map</a>'
      : "";
```
Then change the drawer assignment's links div to include `bmLink`. Replace:
```javascript
    this.drawer.innerHTML = "<h4>" + c.label + "</h4><div>" + (c.summary || "") + "</div>" +
      '<div style="margin-top:8px">' + (links || "<span>No links yet.</span>") + "</div>";
```
with:
```javascript
    this.drawer.innerHTML = "<h4>" + c.label + "</h4><div>" + (c.summary || "") + "</div>" +
      '<div style="margin-top:8px">' + (links || "<span>No links yet.</span>") + "</div>" +
      (bmLink ? '<div style="margin-top:8px">' + bmLink + "</div>" : "");
```
(`c.body_map_url` is a server-computed URL from our resolver; the label is static text. Do not change the existing `c.links`/`c.label`/`c.summary` rendering — that pattern is pre-existing and out of scope.)

- [ ] **Step 2: Syntax check**

Run: `node --check static/atlas.js`
Expected: exit 0, no output.

- [ ] **Step 3: Commit**

```bash
git add static/atlas.js
git commit -m "feat(atlas): render View-on-Body-Map link in concept drawer"
```

(The controller render-verifies the full path: open `/atlas`, select a mapped concept → link present with correct href → click → Body Map opens with the zone highlighted; select an unmapped concept → no link.)

---

## Self-review notes

- **Spec coverage:** resolver + cluster map + override → Task 1; focus params (system/eye/zone/layer, never throw) → Task 2; payload enrichment → Task 3; Atlas link on resolvable concepts only → Task 4. All covered.
- **Deferred (per spec):** dim/zoom focus polish; reverse direction; new zones/systems; bulk override authoring.
- **Testing discipline:** only `bodymap_store` gets pytest (dependency-free, safe). Routes verified via the Doppler import harness; client focus + drawer link verified by headless Chrome (controller).
- **XSS note:** `body_map_url` is a server-computed URL from our own resolver and the link label is static; it is injected via the drawer's pre-existing `innerHTML` pattern (not refactored here — out of scope, and the value is not user-authored).
