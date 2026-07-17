# Body Map — Foot Reflexology System Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add foot reflexology (both soles, full standard reflex map, grouped by body correspondence) as the third Body Map system, generalizing the outline-render dispatch so any outline system works.

**Architecture:** The point engine (point zones, drawn outline surface, `side` laterality, region grouping, per-system anchors + `fitSimilarity`, focus, admin) already exists from the ear slice. The only engine change is generalizing the outline dispatch from the hardcoded `"ear_outline"` to any non-`unit_circle` frame. The rest is registration + content.

**Tech Stack:** Python 3 + Flask, vanilla JS + inline SVG, pytest for the store, headless Chrome for render-verification.

## Global Constraints

- **Work in the worktree** `/tmp/wt-deploy-chat-f33b5ea0` on branch `sess/f33b5ea0-foot` (off the updated `origin/main`, which has iris + atlas + ear merged). All edits/commits there.
- **IRIS + EAR INVARIANT:** the iris (`unit_circle`) and ear (`ear_outline`) must render and behave identically. The frame generalization keeps `unit_circle` on the rings/unit-circle path and every `*_outline` frame (ear included) on the outline path. Re-render iris and ear to confirm.
- **Never run the bare full test suite** (a bare `pytest` sends real email). Run ONLY `pytest tests/test_bodymap_store.py -v`.
- **App-import / route verification harness:** `doppler run -p remedy-match -c prd -- env -u DATA_DIR python3 -c "..."` (`-p remedy-match -c prd`; `env -u DATA_DIR` strips the injected `/data`, so no cron scheduler starts and stores read the worktree seeds). Use `python3`.
- **`bodymap_store.py` keeps zero Flask/Pinecone imports.**
- **No PHI to server:** the photo stays in-browser (`URL.createObjectURL`); no upload/fetch of the image.
- **Copy rules:** no em dashes, no ALL CAPS, no "Hook:" label. **Naming:** `body-map`/`bodymap`. Disclaimer text unchanged.

---

## File Structure

**Modify:**
- `static/body-map.js` — generalize the outline dispatch (`renderChart`, `refToScreen`) via an `isOutlineFrame()` test; add `foot` to the `wire()` allow-list.
- `bodymap_store.py` — register the foot system in `SYSTEMS`/`_SEED_NAMES`.
- `static/body-map.html` — add the Foot dropdown option.
- `tests/test_bodymap_store.py` — foot-registration + foot-seed-validity tests.

**Create:**
- `data/bodymap-foot.json` — both-soles standard reflex chart.

---

### Task 1: Engine frame-generalization + foot registration

**Files:**
- Modify: `static/body-map.js`, `bodymap_store.py`, `static/body-map.html`
- Test: `tests/test_bodymap_store.py`

**Interfaces:**
- Produces: outline dispatch that works for any `*_outline` frame; `SYSTEMS["foot"]`; Foot dropdown option.

- [ ] **Step 1: `static/body-map.js` — add `isOutlineFrame()` and use it in both dispatch sites**

Add the helper near the other helpers (after `groupsOf`):
```javascript
  function isOutlineFrame() { return state.frame && state.frame !== "unit_circle"; }
```
In `refToScreen`, replace `if (state.frame === "ear_outline")` with `if (isOutlineFrame())`.
In `renderChart`, replace `if (state.frame === "ear_outline")` with `if (isOutlineFrame())`.
(Everything else in those functions unchanged — iris `unit_circle` still takes the rings/unit-circle path; ear `ear_outline` and foot `foot_outline` take the outline path.)

- [ ] **Step 2: `static/body-map.js` — add `foot` to the `wire()` allow-list**

Replace the allow-list line in `wire()`:
```javascript
    const initialSystem = (sys === "iridology" || sys === "sclerology" || sys === "ear" || sys === "foot") ? sys : "iridology";
```

- [ ] **Step 3: `bodymap_store.py` — register the foot system**

Add `"foot"` to `SYSTEMS` and `"bodymap-foot.json"` to `_SEED_NAMES`:
```python
SYSTEMS = {
    "iridology": DATA_DIR / "bodymap-iridology.json",
    "sclerology": DATA_DIR / "bodymap-sclerology.json",
    "ear": DATA_DIR / "bodymap-ear.json",
    "foot": DATA_DIR / "bodymap-foot.json",
}

_SEED_NAMES = ("bodymap-iridology.json", "bodymap-sclerology.json", "bodymap-ear.json", "bodymap-foot.json")
```

- [ ] **Step 4: `static/body-map.html` — add the Foot option**

```html
        <option value="ear">Ear (auricular)</option>
        <option value="foot">Foot (reflexology)</option>
```

- [ ] **Step 5: Write the registration test** (append to `tests/test_bodymap_store.py`)

```python
def test_foot_system_registered():
    assert "foot" in bodymap_store.SYSTEMS
    assert bodymap_store.SYSTEMS["foot"].name == "bodymap-foot.json"
```

- [ ] **Step 6: Verify**

Run: `pytest tests/test_bodymap_store.py -v` → all pass (prior + new registration test).
Run: `node --check static/body-map.js` → exit 0.

- [ ] **Step 7: Commit**

```bash
git add static/body-map.js bodymap_store.py static/body-map.html tests/test_bodymap_store.py
git commit -m "feat(bodymap): generalize outline dispatch to any frame + register foot system"
```

(Controller render-verify: iris + ear unchanged after the frame generalization; foot appears in the dropdown, renders empty until Task 2.)

---

### Task 2: Foot seed — both soles, full standard reflex chart

**Files:**
- Create: `data/bodymap-foot.json`
- Test: `tests/test_bodymap_store.py`

**Interfaces:**
- Consumes: `validate_zone`, `SYSTEMS` (Task 1).
- Produces: the shipped foot seed.

**Coordinate convention (normalized `[0,1]`, one shared sole template for both sides):** the sole is drawn with the **toes at the top (small y)** and the **heel at the bottom (large y)**; the **medial edge (big-toe / spine side) is toward small x (~0.28-0.36)** and the **lateral edge (little-toe side) toward large x (~0.66-0.75)**. Bands: toes `y 0.05-0.18`, ball `y 0.18-0.40`, upper arch `y 0.40-0.55`, lower arch `y 0.55-0.72`, heel `y 0.72-0.92`. Both left and right soles use the SAME template coordinates for a given reflex; laterality is carried by the `side` field, and lateralized organs exist only on their side (the mirrored outline is deferred).

- [ ] **Step 1: Write the failing test** (append to `tests/test_bodymap_store.py`)

```python
def test_shipped_foot_seed_valid():
    import pathlib
    repo = pathlib.Path(bodymap_store.__file__).resolve().parent / "data"
    data = json.loads((repo / "bodymap-foot.json").read_text())
    assert data["reference_frame"] == "foot_outline"
    assert data.get("outline")
    group_ids = {g["id"] for g in data.get("groups", [])}
    sides = set()
    for z in data["zones"]:
        ok, err = bodymap_store.validate_zone(z)
        assert ok, f"foot zone {z.get('id')}: {err}"
        assert z["geometry"]["type"] == "point"
        assert z["group"] in group_ids, f"{z['id']} bad group {z['group']}"
        sides.add(z["side"])
    assert sides == {"left", "right"}, "both soles must be populated"
    keys = {a["key"] for a in data.get("anchors", [])}
    assert {"big-toe-tip", "heel-center", "little-toe-base"} <= keys
    # lateralized organs on the correct side
    ids = {z["id"] for z in data["zones"]}
    assert "foot-R-liver" in ids and "foot-L-heart" in ids
```

- [ ] **Step 2: Run to verify it fails**

Run: `pytest tests/test_bodymap_store.py::test_shipped_foot_seed_valid -v`
Expected: FAIL (file does not exist).

- [ ] **Step 3: Author `data/bodymap-foot.json`**

Build the file with this exact top-level structure, then one point zone per row of the reflex table below.

Top-level:
```json
{
  "system": "foot",
  "reference_frame": "foot_outline",
  "outline": "M 0.34 0.10 C 0.28 0.02 0.20 0.04 0.20 0.14 C 0.20 0.24 0.24 0.34 0.24 0.44 C 0.24 0.56 0.20 0.66 0.24 0.76 C 0.27 0.86 0.34 0.94 0.46 0.94 C 0.60 0.94 0.70 0.86 0.70 0.74 C 0.70 0.60 0.66 0.50 0.68 0.40 C 0.70 0.30 0.74 0.22 0.70 0.16 C 0.66 0.10 0.60 0.12 0.56 0.10 C 0.50 0.06 0.44 0.06 0.40 0.07 C 0.37 0.075 0.35 0.085 0.34 0.10 Z",
  "groups": [
    { "id": "head-sinus", "label": "Head and sinuses (toes)" },
    { "id": "neck-thyroid", "label": "Neck and thyroid" },
    { "id": "chest-lung", "label": "Chest, lungs and heart" },
    { "id": "digestive", "label": "Digestive organs" },
    { "id": "urinary", "label": "Urinary and adrenal" },
    { "id": "spine", "label": "Spine (medial edge)" },
    { "id": "pelvis-elimination", "label": "Pelvis and elimination (heel)" },
    { "id": "limb", "label": "Shoulder, arm and leg (lateral edge)" }
  ],
  "anchors": [
    { "key": "big-toe-tip", "template": {"x": 0.33, "y": 0.05}, "hint": "Tap the tip of your big toe." },
    { "key": "heel-center", "template": {"x": 0.46, "y": 0.90}, "hint": "Tap the center of your heel." },
    { "key": "little-toe-base", "template": {"x": 0.70, "y": 0.18}, "hint": "Tap the base of your little toe." }
  ],
  "zones": [ ... one per reflex row below ... ]
}
```

Each zone has this shape (fill from the table; `meaning_glen` is always `""`, `layers` always the three-null object):
```json
{ "id": "foot-<L|R>-<slug>", "side": "<left|right>", "group": "<group id>", "geometry": {"type":"point","x":<x>,"y":<y>}, "anatomy": "<name>", "meaning_standard": "<one-line reflex description>", "meaning_glen": "", "layers": {"embryological_depth": null, "stress_affirmation": null, "touch_for_health": null} }
```

**Reflex table.** `Sides` column: `both` → create TWO zones (one `side:"left"` id `foot-L-<slug>`, one `side:"right"` id `foot-R-<slug>`) at the same x,y; `left`/`right` → one zone on that side only. Use the given x,y verbatim. Write a short, plain `meaning_standard` for each (what body part it reflects); no em dashes, no ALL CAPS.

| slug | anatomy | sides | group | x | y |
|---|---|---|---|---|---|
| brain | Brain and head | both | head-sinus | 0.32 | 0.07 |
| pituitary | Pituitary gland | both | head-sinus | 0.34 | 0.11 |
| pineal | Pineal and hypothalamus | both | head-sinus | 0.30 | 0.10 |
| temples | Side of head and temples | both | head-sinus | 0.28 | 0.12 |
| sinuses | Sinuses | both | head-sinus | 0.52 | 0.06 |
| teeth-jaw | Teeth and jaw | both | head-sinus | 0.50 | 0.13 |
| eyes | Eyes | both | head-sinus | 0.47 | 0.17 |
| ears | Ears | both | head-sinus | 0.64 | 0.17 |
| neck-throat | Neck and throat | both | neck-thyroid | 0.34 | 0.16 |
| thyroid | Thyroid | both | neck-thyroid | 0.36 | 0.22 |
| parathyroid | Parathyroid | both | neck-thyroid | 0.34 | 0.20 |
| trapezius | Trapezius and shoulder line | both | neck-thyroid | 0.64 | 0.22 |
| lung | Lungs and bronchi | both | chest-lung | 0.50 | 0.30 |
| chest-breast | Chest and breast | both | chest-lung | 0.56 | 0.27 |
| heart | Heart | left | chest-lung | 0.40 | 0.30 |
| diaphragm | Diaphragm | both | chest-lung | 0.50 | 0.38 |
| solar-plexus | Solar plexus | both | chest-lung | 0.45 | 0.40 |
| stomach | Stomach | left | digestive | 0.40 | 0.45 |
| pancreas | Pancreas | left | digestive | 0.43 | 0.47 |
| spleen | Spleen | left | digestive | 0.60 | 0.47 |
| liver | Liver | right | digestive | 0.62 | 0.45 |
| gallbladder | Gallbladder | right | digestive | 0.60 | 0.49 |
| small-intestine | Small intestine | both | digestive | 0.50 | 0.65 |
| transverse-colon | Transverse colon | both | digestive | 0.50 | 0.58 |
| ascending-colon | Ascending colon | right | digestive | 0.62 | 0.63 |
| ileocecal | Ileocecal valve | right | digestive | 0.64 | 0.69 |
| descending-colon | Descending colon | left | digestive | 0.62 | 0.63 |
| sigmoid-colon | Sigmoid colon | left | digestive | 0.55 | 0.71 |
| adrenal | Adrenal gland | both | urinary | 0.44 | 0.50 |
| kidney | Kidney | both | urinary | 0.45 | 0.55 |
| ureter | Ureter | both | urinary | 0.42 | 0.63 |
| bladder | Bladder | both | urinary | 0.36 | 0.69 |
| cervical-spine | Cervical spine | both | spine | 0.30 | 0.17 |
| thoracic-spine | Thoracic spine | both | spine | 0.29 | 0.35 |
| lumbar-spine | Lumbar spine | both | spine | 0.29 | 0.55 |
| sacral-coccyx | Sacrum and coccyx | both | spine | 0.31 | 0.78 |
| shoulder-arm | Shoulder and arm | both | limb | 0.72 | 0.26 |
| elbow | Elbow | both | limb | 0.74 | 0.40 |
| knee-leg | Knee and leg | both | limb | 0.72 | 0.60 |
| sciatic | Sciatic nerve | both | pelvis-elimination | 0.50 | 0.85 |
| hip-pelvis | Hip and pelvis | both | pelvis-elimination | 0.45 | 0.80 |
| lower-back | Lower back | both | pelvis-elimination | 0.34 | 0.80 |
| uterus-prostate | Uterus or prostate | both | pelvis-elimination | 0.28 | 0.84 |
| ovary-testis | Ovary or testis | both | pelvis-elimination | 0.72 | 0.84 |
| rectum | Rectum | both | pelvis-elimination | 0.40 | 0.83 |

(Reflexes marked `both` yield a left and a right zone → total zones ≈ 39 rows: 34 `both` × 2 + 5 single = 73 zones.)

- [ ] **Step 4: Run to verify it passes**

Run: `pytest tests/test_bodymap_store.py::test_shipped_foot_seed_valid -v` then the full `pytest tests/test_bodymap_store.py -v`.
Expected: PASS (both soles present, all zones valid, groups resolve, anchors + lateralized organs correct).

- [ ] **Step 5: Commit**

```bash
git add data/bodymap-foot.json tests/test_bodymap_store.py
git commit -m "feat(bodymap): seed foot reflexology chart (both soles, full standard set)"
```

(Controller render-verify: Foot draws the sole outline with the reflex points; Side selector offers Left and Right and switches soles; the left sole shows heart/spleen/stomach, the right shows liver/gallbladder/ileocecal; body-correspondence toggles filter; panel reads "Liver, digestive, right sole ..."; sole-photo overlay warps via big-toe/heel/little-toe; `?system=foot&side=right&zone=foot-R-liver` focuses; iris + ear unchanged.)

---

## Self-review notes

- **Spec coverage:** engine outline generalization → Task 1; foot registration + dropdown → Task 1; both-soles full chart grouped by body correspondence → Task 2; anchors → Task 2; iris/ear invariant → Task 1 (frame test keeps `unit_circle` and `ear_outline` on their paths). All covered.
- **Deferred (per spec):** per-side mirrored outline, dorsal/medial/lateral views, iris/ear full-map backfill, polygon/path types.
- **Testing discipline:** store via `pytest tests/test_bodymap_store.py` only; client via headless Chrome (controller).
- **Content caveat:** normalized coordinates are chart-approximate; Glen refines placements via the admin editor. "Full standard map" = the complete standard organ/gland/structure set, not every finest sub-point variant.
