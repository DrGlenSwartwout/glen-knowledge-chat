# Body Map — Reflexology Areas (polygon + mirror + drawing tool) Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax.

**Goal:** Model reflexology as filled, labeled **polygon areas** with **bilateral mirror** symmetry, and add an **admin drawing tool** that traces zones from a reference underlay — so the foot map is authored correctly rather than guessed.

**Architecture:** Additive `polygon` geometry type (store + renderer) alongside sector/point. The mirror lives in the renderer: a `bilateral` zone is authored once and mirrored (`x→1−x`) on the opposite side; asymmetric organs are placed per-side. A console-key-gated drawing tool loads a reference image as a client-side-only underlay and saves traced polygons via new `upsert_zone`/`delete_zone` store functions.

**Tech Stack:** Python 3 + Flask, vanilla JS + inline SVG, pytest for the store, headless Chrome for the client/tool.

## Global Constraints

- **Work in the worktree** `/tmp/wt-deploy-chat-f33b5ea0` on branch `sess/f33b5ea0-mapfix` (carries the rendering fix: `transformPathD`, labels, `groupColor`). All edits/commits there.
- **INVARIANT:** iris (`unit_circle`/sector), sclera, and ear (`point`) render and behave identically. Polygon is a new, additive branch.
- **Never run the bare full test suite** (bare `pytest` sends real email). Run ONLY `pytest tests/test_bodymap_store.py -v`.
- **App-import/route harness:** `doppler run -p remedy-match -c prd -- env -u DATA_DIR python3 -c "..."` (`-p remedy-match -c prd`; `env -u DATA_DIR` strips `/data`, so no cron scheduler and stores read the worktree seeds). Use `python3`.
- **`bodymap_store.py` free of Flask/Pinecone imports.**
- **No PHI / no reference-image leakage to server:** the drawing tool's reference underlay stays in the browser via `URL.createObjectURL`; only the traced polygon coordinates + tags are saved. No image upload/fetch.
- **Copy rules** (no em dashes, no ALL CAPS, no "Hook:"); naming `body-map`/`bodymap`.

---

## Data shape — polygon zone

```json
{
  "id": "foot-liver",
  "side": "right",
  "bilateral": false,
  "group": "digestive",
  "geometry": {"type": "polygon", "points": [[0.60,0.44],[0.66,0.46],[0.64,0.53],[0.58,0.51]]},
  "anatomy": "Liver",
  "meaning_standard": "Reflex area for the liver.",
  "meaning_glen": "",
  "layers": {"embryological_depth": null, "stress_affirmation": null, "touch_for_health": null}
}
```
- `points`: normalized `[0,1]` boundary, ≥3 pairs. `bilateral: true` → authored once on `side`, rendered mirrored on the other; `bilateral: false` → only on `side` (asymmetric organs).

---

### Task 1: Store — polygon validation + upsert/delete

**Files:** Modify `bodymap_store.py`; Test `tests/test_bodymap_store.py`

- [ ] **Step 1: Failing tests** (append)

```python
def _poly_zone(**over):
    base = {"id": "foot-liver", "side": "right", "bilateral": False, "group": "digestive",
            "geometry": {"type": "polygon", "points": [[0.6,0.44],[0.66,0.46],[0.64,0.53],[0.58,0.51]]},
            "anatomy": "Liver", "meaning_standard": "Liver reflex area.", "meaning_glen": "", "layers": {}}
    base.update(over); return base

def test_validate_polygon_accepts():
    ok, err = bodymap_store.validate_zone(_poly_zone()); assert ok is True and err is None

def test_validate_polygon_rejects_too_few_points():
    ok, err = bodymap_store.validate_zone(_poly_zone(geometry={"type":"polygon","points":[[0.1,0.1],[0.2,0.2]]}))
    assert ok is False and "polygon" in err

def test_validate_polygon_rejects_out_of_range():
    ok, err = bodymap_store.validate_zone(_poly_zone(geometry={"type":"polygon","points":[[0.1,0.1],[0.2,0.2],[1.4,0.3]]}))
    assert ok is False and "polygon" in err

def test_upsert_and_delete_zone(tmp_path, monkeypatch):
    p = tmp_path / "bodymap-foot.json"
    p.write_text(json.dumps({"system":"foot","reference_frame":"foot_outline","groups":[{"id":"digestive","label":"Digestive"}],"zones":[]}))
    monkeypatch.setattr(bodymap_store, "SYSTEMS", dict(bodymap_store.SYSTEMS, foot=p))
    bodymap_store.upsert_zone("foot", _poly_zone())
    assert len(json.loads(p.read_text())["zones"]) == 1
    bodymap_store.upsert_zone("foot", _poly_zone(anatomy="Liver v2"))   # update same id
    zs = json.loads(p.read_text())["zones"]; assert len(zs) == 1 and zs[0]["anatomy"] == "Liver v2"
    bodymap_store.delete_zone("foot", "foot-liver")
    assert json.loads(p.read_text())["zones"] == []
    try:
        bodymap_store.delete_zone("foot", "nope"); assert False
    except KeyError:
        pass

def test_upsert_zone_rejects_invalid(tmp_path, monkeypatch):
    p = tmp_path / "bodymap-foot.json"; p.write_text(json.dumps({"system":"foot","zones":[]}))
    monkeypatch.setattr(bodymap_store, "SYSTEMS", dict(bodymap_store.SYSTEMS, foot=p))
    try:
        bodymap_store.upsert_zone("foot", _poly_zone(anatomy=None)); assert False
    except ValueError:
        pass
```

- [ ] **Step 2: Run — expect FAIL** (`pytest tests/test_bodymap_store.py -k "polygon or upsert" -v`)

- [ ] **Step 3: Implement.** In `validate_zone`, add a polygon branch BEFORE the `point` branch:

```python
    if gtype == "polygon":
        pts = geo.get("points")
        if not isinstance(pts, list) or len(pts) < 3:
            return False, "geometry polygon needs >= 3 points"
        for p in pts:
            if not (isinstance(p, (list, tuple)) and len(p) == 2
                    and all(isinstance(v, (int, float)) for v in p)
                    and 0.0 <= float(p[0]) <= 1.0 and 0.0 <= float(p[1]) <= 1.0):
                return False, "polygon points must be [x,y] pairs in [0,1]"
        return True, None
```
Add at module level (near `set_zone_overlay`):
```python
def upsert_zone(system, zone):
    """Add or replace a zone (matched by id) in the system's seed. Raises ValueError if invalid."""
    ok, err = validate_zone(zone)
    if not ok:
        raise ValueError(err)
    path = SYSTEMS[system]
    data = load_map(system)
    zones = data.setdefault("zones", [])
    for i, z in enumerate(zones):
        if z.get("id") == zone.get("id"):
            zones[i] = zone
            break
    else:
        zones.append(zone)
    _write(path, data)


def delete_zone(system, zone_id):
    """Remove a zone by id. Raises KeyError if not present."""
    path = SYSTEMS[system]
    data = load_map(system)
    kept = [z for z in data.get("zones", []) if z.get("id") != zone_id]
    if len(kept) == len(data.get("zones", [])):
        raise KeyError(zone_id)
    data["zones"] = kept
    _write(path, data)
```

- [ ] **Step 4: Run — expect PASS** (`pytest tests/test_bodymap_store.py -v`, all prior + new)

- [ ] **Step 5: Commit** — `git commit -m "feat(bodymap): polygon geometry validation + upsert/delete zone"`

---

### Task 2: Renderer — polygon areas + bilateral mirror (`static/body-map.js`)

**Files:** Modify `static/body-map.js`

- [ ] **Step 1: Generalize `currentZones`** to include bilateral zones on both sides:

```javascript
  function currentZones() {
    if (!state.payload) return [];
    return state.payload.zones.filter(z =>
      (z.bilateral || zoneSide(z) === state.eye) &&
      (state.activeLayers.size === 0 || state.activeLayers.has(zoneGroup(z))));
  }
```

- [ ] **Step 2: Add a mirrored point-mapper + polygon/label helpers** inside `renderChart`, and add the polygon branch. Replace the zone loop body so it computes a per-zone mirror and handles `polygon`, `point`, and the existing sector `else`:

```javascript
    currentZones().forEach(z => {
      const geo = z.geometry || {};
      const mir = z.bilateral && zoneSide(z) !== state.eye;         // mirror the contralateral side
      const N = (x, y) => mapFn({ x: mir ? 1 - x : x, y: y });
      const col = groupColor(z);
      function addLabel(sx, sy) {
        const onRight = sx > 300;
        const t = document.createElementNS(svgNS, "text");
        t.setAttribute("x", (sx + (onRight ? -8 : 8)).toFixed(1));
        t.setAttribute("y", (sy + 3).toFixed(1));
        t.setAttribute("text-anchor", onRight ? "end" : "start");
        t.setAttribute("class", "bm-label"); t.dataset.id = z.id;
        t.textContent = z.anatomy; t.addEventListener("click", () => selectZone(z));
        svg.appendChild(t);
      }
      if (geo.type === "polygon") {
        const pts = geo.points || [];
        const d = pts.map((p, i) => { const s = N(p[0], p[1]); return (i ? "L" : "M") + s.x.toFixed(1) + " " + s.y.toFixed(1); }).join(" ") + " Z";
        const path = document.createElementNS(svgNS, "path");
        path.setAttribute("d", d); path.setAttribute("class", "bm-zone bm-area"); path.dataset.id = z.id;
        path.setAttribute("fill", col); path.setAttribute("fill-opacity", "0.35");
        path.setAttribute("stroke", col); path.setAttribute("stroke-width", "1.2");
        path.addEventListener("click", () => selectZone(z));
        svg.appendChild(path);
        const cx = pts.reduce((a, p) => a + p[0], 0) / pts.length;
        const cy = pts.reduce((a, p) => a + p[1], 0) / pts.length;
        const c = N(cx, cy); addLabel(c.x, c.y);
      } else if (geo.type === "point") {
        const s = N(geo.x, geo.y);
        const dot = document.createElementNS(svgNS, "circle");
        dot.setAttribute("cx", s.x); dot.setAttribute("cy", s.y); dot.setAttribute("r", 5);
        dot.setAttribute("class", "bm-zone bm-point"); dot.dataset.id = z.id;
        dot.setAttribute("fill", col); dot.setAttribute("stroke", "#fff"); dot.setAttribute("stroke-width", "1");
        dot.addEventListener("click", () => selectZone(z));
        svg.appendChild(dot); addLabel(s.x, s.y);
      } else {
        const path = document.createElementNS(svgNS, "path");
        path.setAttribute("d", pointsToPath(arcSectorPoints(z.radial, z.sector), mapFn));
        path.setAttribute("class", "bm-zone"); path.dataset.id = z.id;
        path.addEventListener("click", () => selectZone(z));
        svg.appendChild(path);
      }
    });
```
(Note: sector zones keep `mapFn` directly — iris is never bilateral, so `mir` is false and the point/polygon `N` mapper is unused for them. Behavior for iris/ear unchanged.)

- [ ] **Step 3: `node --check static/body-map.js`** → exit 0. Commit — `git commit -m "feat(bodymap): render polygon area zones with bilateral mirror"`

(Controller render-verify: inject a bilateral polygon into `__bm.state.payload.zones` on the foot, confirm it fills+labels and appears mirrored when the Side switches; confirm iris/ear unchanged.)

---

### Task 3: Admin routes — draw page + upsert/delete (`app.py`)

**Files:** Modify `app.py`

- [ ] **Step 1:** After the existing `/admin/body-map/zone` route, add:

```python
@app.route("/admin/body-map/draw")
def admin_body_map_draw_page():
    return send_from_directory(STATIC, "admin-body-map-draw.html")


@app.route("/admin/body-map/zone/upsert", methods=["POST"])
@require_console_key
def admin_body_map_zone_upsert():
    body = request.get_json(silent=True) or {}
    system, zone = body.get("system"), body.get("zone")
    if not system or not isinstance(zone, dict):
        return fail("system and zone required", 400)
    try:
        bodymap_store.upsert_zone(system, zone)
    except ValueError as e:
        return fail(str(e), 400)
    except KeyError:
        return fail("unknown system", 404)
    return ok({"upserted": zone.get("id")})


@app.route("/admin/body-map/zone/delete", methods=["POST"])
@require_console_key
def admin_body_map_zone_delete():
    body = request.get_json(silent=True) or {}
    system, zid = body.get("system"), body.get("id")
    if not system or not zid:
        return fail("system and id required", 400)
    try:
        bodymap_store.delete_zone(system, zid)
    except KeyError:
        return fail("unknown zone id", 404)
    return ok({"deleted": zid})
```

- [ ] **Step 2: Verify** — write `/tmp/bm_draw_check.sh`:
```bash
cd /tmp/wt-deploy-chat-f33b5ea0 && doppler run -p remedy-match -c prd -- env -u DATA_DIR python3 -c "
import app, os, json
h={'X-Console-Key':os.environ.get('CONSOLE_SECRET',''),'Content-Type':'application/json'}
c=app.app.test_client()
z={'id':'foot-TEST','side':'right','bilateral':True,'group':'digestive','geometry':{'type':'polygon','points':[[0.6,0.44],[0.66,0.46],[0.64,0.53]]},'anatomy':'Test','meaning_standard':'t','meaning_glen':'','layers':{}}
print('upsert', c.post('/admin/body-map/zone/upsert', headers=h, data=json.dumps({'system':'foot','zone':z})).status_code)
print('bad', c.post('/admin/body-map/zone/upsert', headers=h, data=json.dumps({'system':'foot','zone':{'id':'x'}})).status_code)
print('delete', c.post('/admin/body-map/zone/delete', headers=h, data=json.dumps({'system':'foot','id':'foot-TEST'})).status_code)
print('page', c.get('/admin/body-map/draw').status_code)
"
```
Run `bash /tmp/bm_draw_check.sh` → expect `upsert 200`, `bad 400`, `delete 200`, `page 200` (page 404 until Task 4 creates the file — acceptable here; re-run after Task 4). After the run, restore the seed: `git checkout -- data/bodymap-foot.json`.

- [ ] **Step 3: Commit** — `git commit -m "feat(bodymap): admin upsert/delete zone routes + draw page route"`

---

### Task 4: The drawing tool page (`static/admin-body-map-draw.html`)

**Files:** Create `static/admin-body-map-draw.html`

A single console-key-gated page. Requirements (implement in vanilla JS, no framework):
- **Controls:** console key input; system `<select>` (foot); side `<select>` (left/right, the canonical side to author on); a file input to **load a reference image** (`URL.createObjectURL`, drawn as a faint backdrop in the SVG — never uploaded).
- **Canvas:** a 600x600 `<svg viewBox="0 0 600 600">` with the reference `<image>` at low opacity as backdrop; clicking adds a boundary vertex (normalized = clickX/600, clickY/600); the in-progress polygon draws live; a "Close shape" button finishes it.
- **Tag form:** anatomy (text), group `<select>` (fetched from `/body-map/data?system=<sys>` `groups`), bilateral (checkbox), meaning_standard (textarea), id (auto from anatomy slug, editable).
- **Save:** POST `/admin/body-map/zone/upsert` with `{system, zone}` where `zone` carries `geometry:{type:"polygon",points:[[x,y],...]}`, `side`, `bilateral`, `group`, `anatomy`, `meaning_standard`, `meaning_glen:""`, `layers:{...}`. On success, clear the in-progress polygon and refresh the zone list.
- **Zone list:** GET `/admin/body-map/zones?system=<sys>` (console key), list each zone's id/anatomy with a Delete button → POST `/admin/body-map/zone/delete`.
- **Safety:** build all DOM via `createElement`/`textContent` (no innerHTML-of-data); the reference image is client-side only.

- [ ] **Step 1: Create the page** implementing the above. (The controller render-verifies the full flow in a browser.)
- [ ] **Step 2:** Re-run `bash /tmp/bm_draw_check.sh` → `page 200`.
- [ ] **Step 3: Commit** — `git commit -m "feat(bodymap): admin area-drawing tool (trace reference underlay to polygons)"`

(Controller render-verify: open `/admin/body-map/draw`, load a reference image, trace a polygon, tag + save it, confirm it appears in the zone list and renders as a filled labeled area on `/body-map?system=foot`; confirm the reference image never leaves the browser.)

---

## Post-build (content, via the tool — not a code task)
Using the tool over the #77159447 reference, author the foot's reflex areas as mirrored polygons (bilateral zones once + the asymmetric organs per side), retiring the guessed point zones. This is a tracing session, done after the tool ships; Glen verifies the anatomy.

## Self-review notes
- **Spec coverage:** polygon type → Task 1/2; bilateral mirror → Task 2; drawing tool + upsert/delete → Tasks 3/4; reference-image-stays-client-side → Task 4 constraint; foot re-author → post-build content step. Covered.
- **Invariant:** polygon is an additive geometry branch; `mir` is false for non-bilateral zones; iris/ear/sector paths unchanged. Verified by re-render.
- **Testing:** store via `pytest tests/test_bodymap_store.py`; routes via the Doppler harness; tool via headless Chrome.
