# Body Map — Ear System (typed-geometry generalization) Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add the ear (auricular) as the second Body Map system by generalizing the engine to typed geometry (point zones + a drawn ear outline + region grouping + system-defined anchors), additively, so the live iris renders and behaves identically.

**Architecture:** Each v1 surface gains a dispatch that defaults to the existing iris behavior when the new fields are absent. Store `validate_zone` branches on geometry type; `build_payload` passes through `outline`/`groups`/`anchors`. The client reads `side||eye`, `group||germ_layer`, dispatches the reference surface (iris rings vs ear outline) and the zone mark (sector arc vs point dot), and drives the overlay from a per-system anchor list.

**Tech Stack:** Python 3 + Flask, vanilla JS + inline SVG, pytest for the store, headless Chrome (claude-in-chrome) for client render-verification.

## Global Constraints

- **Work in the worktree** `/tmp/wt-deploy-chat-f33b5ea0` on branch `sess/f33b5ea0-ear`. All edits/commits there.
- **HARD INVARIANT — the iris is unchanged.** Iris (iridology/sclerology) reference chart, overlay, focus params, admin, and `selectZone` panel text must be pixel- and behavior-identical after every task. The generalization is additive: new code paths activate only on the new fields. Re-render the iris to confirm.
- **Never run the bare full test suite** (a bare `pytest` sends real email). Run ONLY `pytest tests/test_bodymap_store.py -v`.
- **App-import / route verification harness:** `doppler run -p remedy-match -c prd -- env -u DATA_DIR python3 -c "..."` (`-p remedy-match -c prd` selects config; `env -u DATA_DIR` strips the injected `/data` so the import starts no cron scheduler and stores read the worktree seeds). Bare `doppler run` fails; importing with DATA_DIR set starts real cron. Use `python3`.
- **`bodymap_store.py` keeps zero Flask/Pinecone imports.**
- **No PHI to server:** the uploaded photo stays in the browser (`URL.createObjectURL`); no upload/POST of the image.
- **Focus/overlay must never throw** on unknown params or missing fields — guard every lookup.
- **Copy rules:** no em dashes, no ALL CAPS, no "Hook:" label. **Naming:** `body-map`/`bodymap` only. Required disclaimer text unchanged: "This is a visual approximation, not a diagnosis."

---

## File Structure

**Modify:**
- `bodymap_store.py` — generalized `validate_zone` (sector + point), `SYSTEMS`/`_SEED_NAMES` += ear, `build_payload` passes through `outline`/`groups`/`anchors`.
- `tests/test_bodymap_store.py` — point-zone validation tests + ear-seed-validity test.
- `static/body-map.js` — laterality/grouping generalization, frame-aware `refToScreen`, `renderChart` surface+mark dispatch, `loadSystem` side-default + side-select rebuild, anchor-driven overlay + `fitSimilarity`, generalized `applyFocusFromURL`, `wire()` allow-list.
- `static/body-map.html` — add the Ear option and a `#bm-side-label` span on the laterality control.

**Create:**
- `data/bodymap-ear.json` — ear outline path + region groups + overlay anchors + starter auricular point zones (left side).

---

### Task 1: Store — generalized validation + ear registration

**Files:**
- Modify: `bodymap_store.py`
- Test: `tests/test_bodymap_store.py`

**Interfaces:**
- Produces: a `validate_zone` that accepts both sector (iris) and point (ear) zones; `SYSTEMS["ear"]`; `build_payload` returning `groups`/`outline`/`anchors`.

- [ ] **Step 1: Write the failing tests** (append to `tests/test_bodymap_store.py`)

```python
def _point_zone(**over):
    base = {
        "id": "ear-L-shenmen", "side": "left", "group": "triangular-fossa",
        "geometry": {"type": "point", "x": 0.44, "y": 0.28},
        "anatomy": "Shen Men", "meaning_standard": "Calming point.",
        "meaning_glen": "", "layers": {},
    }
    base.update(over)
    return base


def test_validate_point_zone_accepts_complete():
    ok, err = bodymap_store.validate_zone(_point_zone())
    assert ok is True and err is None


def test_validate_point_zone_rejects_out_of_range_xy():
    ok, err = bodymap_store.validate_zone(_point_zone(geometry={"type": "point", "x": 1.4, "y": 0.2}))
    assert ok is False and "point" in err


def test_validate_point_zone_requires_side_and_group():
    z = _point_zone(); del z["side"]
    ok, err = bodymap_store.validate_zone(z)
    assert ok is False and ("side" in err or "eye" in err)
    z2 = _point_zone(); del z2["group"]
    ok2, err2 = bodymap_store.validate_zone(z2)
    assert ok2 is False and "grouping" in err2


def test_validate_sector_zone_still_accepts_iris():
    iris = _zone()  # the iris fixture from earlier in this file (radial+sector+eye+germ_layer)
    ok, err = bodymap_store.validate_zone(iris)
    assert ok is True and err is None


def test_build_payload_passes_through_ear_fields(tmp_path, monkeypatch):
    p = tmp_path / "bodymap-ear.json"
    p.write_text(json.dumps({
        "system": "ear", "reference_frame": "ear_outline", "outline": "M 0 0 Z",
        "groups": [{"id": "lobe", "label": "Lobe"}],
        "anchors": [{"key": "helix-top", "template": {"x": 0.5, "y": 0.05}, "hint": "top"}],
        "zones": [_point_zone(group="lobe")],
    }))
    monkeypatch.setattr(bodymap_store, "SYSTEMS", dict(bodymap_store.SYSTEMS, ear=p))
    payload = bodymap_store.build_payload("ear")
    assert payload["reference_frame"] == "ear_outline"
    assert payload["outline"] == "M 0 0 Z"
    assert payload["groups"] == [{"id": "lobe", "label": "Lobe"}]
    assert payload["anchors"][0]["key"] == "helix-top"
    assert payload["zones"][0]["meaning_display"] == "Calming point."
```

- [ ] **Step 2: Run to verify they fail**

Run: `pytest tests/test_bodymap_store.py -k "point_zone or ear_fields" -v`
Expected: FAIL (point zones rejected by current sector-only validation; `groups`/`outline`/`anchors` absent from payload).

- [ ] **Step 3: Implement** — replace `validate_zone` and the `SYSTEMS`/`_SEED_NAMES` lines and `build_payload` in `bodymap_store.py`.

Replace the `_REQUIRED` tuple and `validate_zone` with:
```python
_REQUIRED_COMMON = ("id", "anatomy", "meaning_standard")


def validate_zone(z):
    """Return (ok, error_message_or_None). Accepts sector zones (iris) and point zones (ear)."""
    if not isinstance(z, dict):
        return False, "zone must be an object"
    for key in _REQUIRED_COMMON:
        if key not in z:
            return False, f"missing required field: {key}"
    if (z.get("side") or z.get("eye")) not in ("right", "left"):
        return False, "side/eye must be 'right' or 'left'"
    if not (z.get("group") or z.get("germ_layer")):
        return False, "missing grouping (group or germ_layer)"
    geo = z.get("geometry") or {}
    gtype = geo.get("type") or ("sector" if ("radial" in z and "sector" in z) else None)
    if gtype == "point":
        x, y = geo.get("x"), geo.get("y")
        if not all(isinstance(v, (int, float)) for v in (x, y)):
            return False, "geometry point x/y must be numbers"
        if not (0.0 <= float(x) <= 1.0 and 0.0 <= float(y) <= 1.0):
            return False, "geometry point x/y must be in [0,1]"
        return True, None
    if gtype == "sector":
        radial = z.get("radial") or {}
        ri, ro = radial.get("r_inner"), radial.get("r_outer")
        if not all(isinstance(v, (int, float)) for v in (ri, ro)):
            return False, "radial.r_inner/r_outer must be numbers"
        if not (0.0 <= float(ri) < float(ro) <= 3.0):
            return False, "radial must satisfy 0 <= r_inner < r_outer <= 3"
        sector = z.get("sector") or {}
        s, e = sector.get("start_deg"), sector.get("end_deg")
        if not all(isinstance(v, (int, float)) for v in (s, e)):
            return False, "sector.start_deg/end_deg must be numbers"
        if not (0.0 <= float(s) < float(e) <= 360.0):
            return False, "sector must satisfy 0 <= start_deg < end_deg <= 360"
        return True, None
    return False, "unknown geometry type"
```

Add `"ear"` to `SYSTEMS` and `_SEED_NAMES`:
```python
SYSTEMS = {
    "iridology": DATA_DIR / "bodymap-iridology.json",
    "sclerology": DATA_DIR / "bodymap-sclerology.json",
    "ear": DATA_DIR / "bodymap-ear.json",
}

_SEED_NAMES = ("bodymap-iridology.json", "bodymap-sclerology.json", "bodymap-ear.json")
```

In `build_payload`, replace the returned dict with (adds `groups`/`outline`/`anchors` pass-through; everything else unchanged):
```python
    return {
        "system": data.get("system", system),
        "reference_frame": data.get("reference_frame", "unit_circle"),
        "germ_layers": data.get("germ_layers", []),
        "groups": data.get("groups", []),
        "outline": data.get("outline", ""),
        "anchors": data.get("anchors", []),
        "zones": zones,
    }
```

- [ ] **Step 4: Run to verify pass**

Run: `pytest tests/test_bodymap_store.py -v`
Expected: PASS (all prior iris/sclera + atlas tests still green + the new point/ear tests). This proves iris validation is unchanged.

- [ ] **Step 5: Commit**

```bash
git add bodymap_store.py tests/test_bodymap_store.py
git commit -m "feat(bodymap): typed-geometry validation (point zones) + ear system registration"
```

---

### Task 2: Ear seed data

**Files:**
- Create: `data/bodymap-ear.json`
- Test: `tests/test_bodymap_store.py`

**Interfaces:**
- Consumes: `validate_zone`, `SYSTEMS` (Task 1).
- Produces: the shipped ear seed every later task loads.

This is a **starter** scaffold (a plausible normalized ear outline + a well-known subset of auricular points, left side), refined later via the admin editor. Coordinates are normalized `[0,1]`.

- [ ] **Step 1: Write the failing test** (append to `tests/test_bodymap_store.py`)

```python
def test_shipped_ear_seed_valid():
    import pathlib
    repo = pathlib.Path(bodymap_store.__file__).resolve().parent / "data"
    data = json.loads((repo / "bodymap-ear.json").read_text())
    assert data["reference_frame"] == "ear_outline"
    assert data.get("outline")
    assert data.get("zones")
    group_ids = {g["id"] for g in data.get("groups", [])}
    for z in data["zones"]:
        ok, err = bodymap_store.validate_zone(z)
        assert ok, f"ear zone {z.get('id')}: {err}"
        assert z["geometry"]["type"] == "point"
        assert z["group"] in group_ids, f"{z['id']} bad group {z['group']}"
    keys = {a["key"] for a in data.get("anchors", [])}
    assert {"helix-top", "lobe-bottom", "tragus"} <= keys
```

- [ ] **Step 2: Run to verify it fails**

Run: `pytest tests/test_bodymap_store.py::test_shipped_ear_seed_valid -v`
Expected: FAIL (file does not exist).

- [ ] **Step 3: Create `data/bodymap-ear.json`**

```json
{
  "system": "ear",
  "reference_frame": "ear_outline",
  "outline": "M 0.55 0.05 C 0.75 0.06 0.82 0.25 0.80 0.42 C 0.79 0.55 0.72 0.62 0.66 0.70 C 0.60 0.80 0.58 0.92 0.48 0.95 C 0.38 0.97 0.30 0.90 0.30 0.78 C 0.24 0.75 0.20 0.66 0.24 0.55 C 0.20 0.45 0.22 0.28 0.32 0.16 C 0.40 0.07 0.47 0.05 0.55 0.05 Z",
  "groups": [
    { "id": "helix", "label": "Helix (outer rim)" },
    { "id": "antihelix", "label": "Antihelix (spine)" },
    { "id": "triangular-fossa", "label": "Triangular fossa" },
    { "id": "concha", "label": "Concha (internal organs)" },
    { "id": "tragus", "label": "Tragus" },
    { "id": "antitragus", "label": "Antitragus" },
    { "id": "lobe", "label": "Lobe (head and face)" }
  ],
  "anchors": [
    { "key": "helix-top", "template": {"x": 0.50, "y": 0.08}, "hint": "Tap the top of your ear, at the top of the rim." },
    { "key": "lobe-bottom", "template": {"x": 0.48, "y": 0.93}, "hint": "Tap the bottom of your earlobe." },
    { "key": "tragus", "template": {"x": 0.31, "y": 0.60}, "hint": "Tap the tragus, the small flap in front of the ear canal." }
  ],
  "zones": [
    { "id": "ear-L-apex", "side": "left", "group": "helix", "geometry": {"type": "point", "x": 0.52, "y": 0.11}, "anatomy": "Ear Apex", "meaning_standard": "Point at the top of the helix used for calming and to settle heat.", "meaning_glen": "", "layers": {"embryological_depth": null, "stress_affirmation": null, "touch_for_health": null} },
    { "id": "ear-L-shenmen", "side": "left", "group": "triangular-fossa", "geometry": {"type": "point", "x": 0.44, "y": 0.26}, "anatomy": "Shen Men", "meaning_standard": "Master calming and regulating point in the triangular fossa.", "meaning_glen": "", "layers": {"embryological_depth": null, "stress_affirmation": null, "touch_for_health": null} },
    { "id": "ear-L-sympathetic", "side": "left", "group": "antihelix", "geometry": {"type": "point", "x": 0.30, "y": 0.40}, "anatomy": "Sympathetic", "meaning_standard": "Autonomic balancing point along the inner antihelix border.", "meaning_glen": "", "layers": {"embryological_depth": null, "stress_affirmation": null, "touch_for_health": null} },
    { "id": "ear-L-point-zero", "side": "left", "group": "concha", "geometry": {"type": "point", "x": 0.50, "y": 0.50}, "anatomy": "Point Zero", "meaning_standard": "Central reference point at the root of the helix; homeostatic anchor.", "meaning_glen": "", "layers": {"embryological_depth": null, "stress_affirmation": null, "touch_for_health": null} },
    { "id": "ear-L-kidney", "side": "left", "group": "concha", "geometry": {"type": "point", "x": 0.55, "y": 0.42}, "anatomy": "Kidney", "meaning_standard": "Concha point reflecting the kidney and fluid balance.", "meaning_glen": "", "layers": {"embryological_depth": null, "stress_affirmation": null, "touch_for_health": null} },
    { "id": "ear-L-liver", "side": "left", "group": "concha", "geometry": {"type": "point", "x": 0.61, "y": 0.53}, "anatomy": "Liver", "meaning_standard": "Concha point reflecting the liver.", "meaning_glen": "", "layers": {"embryological_depth": null, "stress_affirmation": null, "touch_for_health": null} },
    { "id": "ear-L-lung", "side": "left", "group": "concha", "geometry": {"type": "point", "x": 0.50, "y": 0.60}, "anatomy": "Lung", "meaning_standard": "Concha point reflecting the lungs; used for respiration and detox support.", "meaning_glen": "", "layers": {"embryological_depth": null, "stress_affirmation": null, "touch_for_health": null} },
    { "id": "ear-L-heart", "side": "left", "group": "concha", "geometry": {"type": "point", "x": 0.48, "y": 0.53}, "anatomy": "Heart", "meaning_standard": "Deep concha point reflecting the heart and circulation.", "meaning_glen": "", "layers": {"embryological_depth": null, "stress_affirmation": null, "touch_for_health": null} },
    { "id": "ear-L-stomach", "side": "left", "group": "concha", "geometry": {"type": "point", "x": 0.46, "y": 0.47}, "anatomy": "Stomach", "meaning_standard": "Concha point at the helix root reflecting the stomach.", "meaning_glen": "", "layers": {"embryological_depth": null, "stress_affirmation": null, "touch_for_health": null} },
    { "id": "ear-L-subcortex", "side": "left", "group": "lobe", "geometry": {"type": "point", "x": 0.50, "y": 0.84}, "anatomy": "Subcortex (head)", "meaning_standard": "Lobe point associated with the head and higher regulation.", "meaning_glen": "", "layers": {"embryological_depth": null, "stress_affirmation": null, "touch_for_health": null} }
  ]
}
```

- [ ] **Step 4: Run to verify it passes**

Run: `pytest tests/test_bodymap_store.py::test_shipped_ear_seed_valid -v`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add data/bodymap-ear.json tests/test_bodymap_store.py
git commit -m "feat(bodymap): seed ear (auricular) starter dataset"
```

---

### Task 3: Client — reference-chart generalization (iris unchanged)

**Files:**
- Modify: `static/body-map.js`, `static/body-map.html`

**Interfaces:**
- Consumes the enriched payload (`reference_frame`, `outline`, `groups`, point `geometry`, `side`).
- Produces frame/mark dispatch; iris path unchanged.

- [ ] **Step 1: `static/body-map.html` — add the Ear option and a laterality label span**

Change the system select to include Ear:
```html
      <select id="bm-system">
        <option value="iridology">Iridology (iris)</option>
        <option value="sclerology">Sclerology (sclera)</option>
        <option value="ear">Ear (auricular)</option>
      </select>
```
Change the eye control label to carry an id (so it can relabel to "Side"):
```html
    <label><span id="bm-side-label">Eye</span>
      <select id="bm-eye">
        <option value="right">Right</option>
        <option value="left">Left</option>
      </select>
    </label>
```

- [ ] **Step 2: `static/body-map.js` — add laterality/grouping helpers** (near the top, after `state`)

```javascript
  function zoneSide(z) { return z.side || z.eye; }
  function zoneGroup(z) { return z.group || z.germ_layer; }
  function groupsOf(p) { return (p && (p.groups && p.groups.length ? p.groups : p.germ_layers)) || []; }
```

- [ ] **Step 3: `static/body-map.js` — frame-aware `refToScreen`** (replace the one-liner at line ~26)

```javascript
  // reference-frame normalized point -> reference-chart screen point
  function refToScreen(p) {
    if (state.frame === "ear_outline") { return { x: p.x * VIEW, y: p.y * VIEW }; }
    const R = state.chartR || 250; return { x: CX + p.x * R, y: CY + p.y * R };
  }
```

- [ ] **Step 4: `static/body-map.js` — generalize `currentZones` and `selectZone`**

Replace `currentZones`:
```javascript
  function currentZones() {
    if (!state.payload) return [];
    return state.payload.zones.filter(z => zoneSide(z) === state.eye &&
      (state.activeLayers.size === 0 || state.activeLayers.has(zoneGroup(z))));
  }
```
Replace the meta construction in `selectZone` (keep the rest of the function identical) so iris text is unchanged and the ear reads naturally:
```javascript
  function selectZone(z) {
    document.querySelectorAll(".bm-zone").forEach(e => e.classList.toggle("bm-sel", e.dataset.id === z.id));
    const panel = document.getElementById("bm-panel");
    panel.replaceChildren();
    const h = document.createElement("h2"); h.textContent = z.anatomy;
    const groupNoun = z.germ_layer ? " layer, " : " region, ";
    const sideNoun = z.eye ? " eye" : " ear";
    const meta = document.createElement("p");
    const strong = document.createElement("strong"); strong.textContent = zoneGroup(z);
    meta.append(strong, document.createTextNode(groupNoun + zoneSide(z) + sideNoun));
    const body = document.createElement("p"); body.textContent = z.meaning_display || z.meaning_standard;
    panel.append(h, meta, body);
  }
```

- [ ] **Step 5: `static/body-map.js` — generalize `renderLayerToggles`** (source from `groupsOf`)

Replace the `(state.payload.germ_layers || []).forEach(...)` line in `renderLayerToggles` with:
```javascript
    groupsOf(state.payload).forEach(g => {
```
(everything else in the function unchanged.)

- [ ] **Step 6: `static/body-map.js` — `renderChart` surface + mark dispatch**

Replace the germ-layer ring block and the zone loop in `renderChart` so it dispatches (iris behavior byte-identical when `frame !== "ear_outline"` and zones are sectors):
```javascript
  function renderChart() {
    if (!state.payload) return;
    const svg = document.getElementById("bm-svg");
    svg.innerHTML = "";
    const mapFn = state.transform ? (p) => state.transform(p) : refToScreen;
    if (state.frame === "ear_outline") {
      if (state.payload.outline) {
        const path = document.createElementNS(svgNS, "path");
        path.setAttribute("d", pointsToPath(sampleOutline(state.payload.outline), mapFn));
        path.setAttribute("fill", "none"); path.setAttribute("stroke", "#c8b98f"); path.setAttribute("stroke-width", "1.5");
        svg.appendChild(path);
      }
    } else {
      (state.payload.germ_layers || []).forEach(g => {
        [g.r_inner, g.r_outer].forEach(rr => {
          const c = document.createElementNS(svgNS, "circle");
          const o = mapFn({ x: 0, y: 0 }), edge = mapFn({ x: rr, y: 0 });
          c.setAttribute("cx", o.x); c.setAttribute("cy", o.y);
          c.setAttribute("r", Math.hypot(edge.x - o.x, edge.y - o.y));
          c.setAttribute("fill", "none"); c.setAttribute("stroke", "#d9cfb8"); c.setAttribute("stroke-width", "1");
          svg.appendChild(c);
        });
      });
    }
    currentZones().forEach(z => {
      const geo = z.geometry || {};
      if (geo.type === "point") {
        const s = mapFn({ x: geo.x, y: geo.y });
        const dot = document.createElementNS(svgNS, "circle");
        dot.setAttribute("cx", s.x); dot.setAttribute("cy", s.y); dot.setAttribute("r", 7);
        dot.setAttribute("class", "bm-zone bm-point"); dot.dataset.id = z.id;
        dot.addEventListener("click", () => selectZone(z));
        svg.appendChild(dot);
      } else {
        const path = document.createElementNS(svgNS, "path");
        path.setAttribute("d", pointsToPath(arcSectorPoints(z.radial, z.sector), mapFn));
        path.setAttribute("class", "bm-zone"); path.dataset.id = z.id;
        path.addEventListener("click", () => selectZone(z));
        svg.appendChild(path);
      }
    });
  }
```
Add an `sampleOutline` helper (converts the SVG-ish `outline` into a polyline our `pointsToPath` + `mapFn` can transform — since the outline is normalized, sample its path). Add near `pointsToPath`:
```javascript
  // Parse a normalized outline path ("M x y C ... Z") into a list of {x,y} anchor points
  // by pulling every numeric coordinate pair. Good enough to draw a closed, warpable outline.
  function sampleOutline(d) {
    const nums = (d.match(/-?\d*\.?\d+/g) || []).map(Number);
    const pts = [];
    for (let i = 0; i + 1 < nums.length; i += 2) pts.push({ x: nums[i], y: nums[i + 1] });
    return pts;
  }
```

- [ ] **Step 7: `static/body-map.js` — `loadSystem` sets frame + default side + rebuilds the side select**

Replace `loadSystem` with:
```javascript
  async function loadSystem(system) {
    const res = await fetch("/body-map/data?system=" + encodeURIComponent(system));
    state.payload = await res.json();
    state.frame = state.payload.reference_frame || "unit_circle";
    state.chartR = computeChartR(state.payload);
    state.activeLayers.clear();
    // laterality: relabel + repopulate from the sides present, keep current if still valid
    const sides = [...new Set((state.payload.zones || []).map(zoneSide))];
    const sel = document.getElementById("bm-eye");
    sel.replaceChildren();
    sides.forEach(s => { const o = document.createElement("option"); o.value = s; o.textContent = s.charAt(0).toUpperCase() + s.slice(1); sel.appendChild(o); });
    if (!sides.includes(state.eye)) state.eye = sides[0] || "right";
    sel.value = state.eye;
    document.getElementById("bm-side-label").textContent = state.frame === "ear_outline" ? "Side" : "Eye";
    renderLayerToggles(); renderChart();
  }
```

- [ ] **Step 8: Syntax check**

Run: `node --check static/body-map.js`
Expected: exit 0, no output.

- [ ] **Step 9: Commit**

```bash
git add static/body-map.js static/body-map.html
git commit -m "feat(bodymap): frame-aware reference chart + point zones + side/group generalization"
```

(Controller render-verify: iris renders identically — rings, sector zones, panel text, eye select; Ear renders the outline + point dots, region toggles, Side select. Iris is the invariant.)

---

### Task 4: Client — generalized overlay anchors + focus params

**Files:**
- Modify: `static/body-map.js`

**Interfaces:**
- Consumes `payload.anchors` (ear) with `template` coords; falls back to the iris pupil/limbus/twelve construction when absent.

- [ ] **Step 1: Replace the fixed `ANCHOR_STEPS` usage with a per-system accessor**

Keep the const `ANCHOR_STEPS` (the iris default) but add an accessor and a similarity fitter (near `computeSimilarity`):
```javascript
  function activeAnchorSteps() {
    const a = state.payload && state.payload.anchors;
    return (a && a.length) ? a : ANCHOR_STEPS;
  }

  // Fit a similarity (translation + rotation + uniform scale) mapping template coords -> screen,
  // from the first two anchor correspondences. Exact for 2 points; a third is not required.
  function fitSimilarity(steps) {
    const a0 = steps[0].template, a1 = steps[1].template;
    const b0 = anchors[steps[0].key], b1 = anchors[steps[1].key];
    const dax = a1.x - a0.x, day = a1.y - a0.y;
    const dbx = b1.x - b0.x, dby = b1.y - b0.y;
    const denom = dax * dax + day * day || 1e-9;
    const mx = (dbx * dax + dby * day) / denom;
    const my = (dby * dax - dbx * day) / denom;
    const tx = b0.x - (mx * a0.x - my * a0.y);
    const ty = b0.y - (my * a0.x + mx * a0.y);
    return (n) => ({ x: mx * n.x - my * n.y + tx, y: my * n.x + mx * n.y + ty });
  }
```

- [ ] **Step 2: Generalize `beginAnchoring` and `onCanvasClick`** to use `activeAnchorSteps()` and dispatch the transform builder

Replace `beginAnchoring`'s last line and `onCanvasClick` so they read the active steps and, on completion, build the transform per system:
```javascript
  function beginAnchoring() {
    anchorIdx = 0; Object.keys(anchors).forEach(k => delete anchors[k]);
    state.transform = null; renderChart();
    document.getElementById("bm-anchor-hint").textContent = activeAnchorSteps()[0].hint;
  }

  function onCanvasClick(evt) {
    const steps = activeAnchorSteps();
    if (document.getElementById("bm-photo").hidden || anchorIdx >= steps.length) return;
    const svg = document.getElementById("bm-svg");
    const rect = svg.getBoundingClientRect();
    const x = (evt.clientX - rect.left) / rect.width * VIEW;
    const y = (evt.clientY - rect.top) / rect.height * VIEW;
    anchors[steps[anchorIdx].key] = { x, y };
    anchorIdx++;
    if (anchorIdx < steps.length) {
      document.getElementById("bm-anchor-hint").textContent = steps[anchorIdx].hint;
      drawAnchors();
    } else {
      document.getElementById("bm-anchor-hint").textContent = "Overlay placed. Re-upload to redo.";
      state.transform = (state.payload && state.payload.anchors && state.payload.anchors.length)
        ? fitSimilarity(steps)
        : computeSimilarity(anchors.pupil, anchors.limbus, anchors.twelve);
      renderChart(); drawAnchors();
      console.log("[bodymap] overlay placed");
    }
  }
```

- [ ] **Step 3: Generalize `applyFocusFromURL`** (accept `side`/`group`, keep `eye`/`layer`; use helpers)

Replace `applyFocusFromURL` with:
```javascript
  function applyFocusFromURL(params) {
    if (!state.payload) return;
    const side = params.get("side") || params.get("eye");
    const sides = new Set((state.payload.zones || []).map(zoneSide));
    if (side && sides.has(side)) {
      state.eye = side; document.getElementById("bm-eye").value = side; renderChart();
    }
    const zoneId = params.get("zone");
    if (zoneId) {
      const z = (state.payload.zones || []).find(x => x.id === zoneId);
      if (z) {
        if (zoneSide(z) !== state.eye) {
          state.eye = zoneSide(z); document.getElementById("bm-eye").value = state.eye; renderChart();
        }
        selectZone(z);
        return;
      }
    }
    const groupId = params.get("group") || params.get("layer");
    if (groupId) {
      const grp = groupsOf(state.payload).find(g => g.id === groupId);
      if (grp) {
        state.activeLayers.clear(); state.activeLayers.add(groupId);
        const cb = document.getElementById("bml-" + groupId);
        if (cb) cb.checked = true;
        renderChart();
        const first = currentZones()[0];
        if (first) selectZone(first);
      }
    }
  }
```

- [ ] **Step 4: `wire()` — allow `ear` as an initial `?system`** (replace the allow-list line)

```javascript
    const initialSystem = (sys === "iridology" || sys === "sclerology" || sys === "ear") ? sys : "iridology";
```

- [ ] **Step 5: Syntax check**

Run: `node --check static/body-map.js`
Expected: exit 0.

- [ ] **Step 6: Commit**

```bash
git add static/body-map.js
git commit -m "feat(bodymap): system-defined overlay anchors + similarity fit + side/group focus"
```

(Controller render-verify: the iris overlay still uses pupil/limbus/twelve and warps identically; the ear shows helix/lobe/tragus prompts and warps the points onto an ear photo; `/body-map?system=ear&zone=ear-L-shenmen` opens the ear on Shen Men; the photo never uploads to the server.)

---

## Self-review notes

- **Spec coverage:** typed geometry (point) → Task 1/3; ear seed with outline/groups/anchors/points → Task 2; drawn ear outline surface → Task 3 (renderChart + sampleOutline); region grouping → Task 3 (groupsOf/renderLayerToggles); `side` laterality + relabel → Task 3 (loadSystem); anchor-driven overlay + similarity fit → Task 4; generalized focus params → Task 4; Ear in dropdown → Task 3. All covered.
- **Iris invariant:** Task 1 keeps sector validation identical (proved by the retained iris tests); Task 3's dispatch runs the exact old code when `frame !== "ear_outline"` and zones are sectors; Task 4's transform builder falls back to `computeSimilarity` when there are no `payload.anchors`. Every task ends with an iris re-render check.
- **Deferred (per spec):** right ear, full point set, polygon/path types, atlas targets on ear zones, iris retro-migration, ML anchoring.
- **Testing discipline:** store via `pytest tests/test_bodymap_store.py` only; client via headless Chrome (controller), never a bare full pytest.
