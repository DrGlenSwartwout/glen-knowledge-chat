# Body Map — Foot Reflexology System — Design Spec

**Date:** 2026-07-16
**Status:** Approved design, pre-implementation
**App:** deploy-chat (`glen-knowledge-chat`)
**Builds on:** Body Map iris/sclera (#945), Atlas deep-link (#948), ear + typed-geometry generalization (#950) — all merged/live.

---

## Goal

Add **foot reflexology** as the third Body Map system: both soles (plantar surface), the full standard reflex map grouped by body correspondence, on the already-generalized point engine. Requires only a small additional engine generalization (make the outline-render path work for any outline system, not just the ear) plus the content.

## Locked decisions (from brainstorm)

1. **View = the sole (plantar surface).** That is where reflexology lives.
2. **Both feet, full standard map — not a starter subset.** Per Glen: "all the maps should show the full standard maps." The standard reflexology chart is inherently two feet because organs are lateralized (heart/spleen/pancreas weighted left; liver/gallbladder/ileocecal weighted right). So the foot ships **left and right soles**, each with its complete standard reflex set, correctly lateralized.
3. **Grouping = body correspondence** (what each zone reflects), not foot region: head & sinuses (toes), neck & thyroid (toe bases/ball), chest & lungs (ball), digestive organs (arch), urinary (arch to heel), spine (medial arch edge), pelvis & elimination (heel). This is reflexology's organizing principle — the body mapped onto the sole — and reads naturally in the panel ("Liver, digestive-organs, right sole").
4. **Overlay anchors** = big-toe tip, heel center, base of the little toe — three landmarks for the same similarity warp.
5. **Source = a standard reflexology sole chart** as the scaffold, refined by Glen via the admin editor (normalized coordinates by hand are approximate).

**Follow-on (separate slice, not this one):** backfill the iris and ear maps — shipped as starters — to their full standard charts, to honor "all the maps show the full standard maps" everywhere.

---

## Architecture

The point engine (typed geometry, drawn outline surface, region grouping, `side` laterality, per-system anchors + `fitSimilarity`, focus params, admin editor) already exists from the ear slice. The foot needs one generalization and the content.

### 1. Engine — generalize the outline dispatch (`static/body-map.js`)

Today the outline-vs-rings dispatch is hardcoded to the ear:
- `renderChart`: `if (state.frame === "ear_outline") { draw outline } else { draw rings }`
- `refToScreen`: `if (state.frame === "ear_outline") { [0,1]→canvas } else { unit-circle }`

Generalize both to **any outline frame** so foot/hand/face work without further code:
- Introduce `state.frame`-based test `isOutlineFrame()` = `state.frame !== "unit_circle"` (equivalently `state.frame.endsWith("_outline")`).
- `renderChart` and `refToScreen` use `isOutlineFrame()` instead of the literal `=== "ear_outline"`.

This is additive: iris (`unit_circle`) → rings/unit-circle unchanged; ear (`ear_outline`) → outline path unchanged (still `!== "unit_circle"`); foot (`foot_outline`) → outline path, same [0,1] mapping. No other engine change (point rendering, `side`, groups, anchors, `fitSimilarity`, focus, admin all already generic).

Also: add `foot` to the `wire()` initial-`?system` allow-list.

### 2. Store — register the foot system (`bodymap_store.py`)

Add `"foot": DATA_DIR / "bodymap-foot.json"` to `SYSTEMS` and `"bodymap-foot.json"` to `_SEED_NAMES`. `validate_zone` already accepts point zones — no change. (An optional test asserts the foot seed validates.)

### 3. Page — dropdown option (`static/body-map.html`)

Add `<option value="foot">Foot (reflexology)</option>` to `#bm-system`.

### 4. Seed — `data/bodymap-foot.json`

- `reference_frame: "foot_outline"`, `outline`: a normalized `[0,1]` sole outline path (a generic sole shape; the same outline is shown for both sides — the mirror is cosmetic, the points carry the meaning; a per-side mirrored outline is a later refinement).
- `groups`: the body-correspondence groups (head-sinus, neck-thyroid, chest-lung, digestive, urinary, spine, pelvis-elimination, and a lateral-limb/shoulder group).
- `anchors`: `big-toe-tip`, `heel-center`, `little-toe-base` with `template` coords and hints.
- `zones`: the **full standard reflex set for BOTH soles** — point zones with `side` (left/right), `group`, `geometry:{type:point,x,y}`, `anatomy`, `meaning_standard`, empty `meaning_glen`. Lateralized correctly (e.g., heart/spleen/stomach/pancreas emphasis on the left sole; liver/gallbladder/ileocecal-valve/ascending-colon on the right sole; shared midline and bilateral organs — kidneys, adrenals, lungs, sinuses, spine, bladder — on both).
- "Full standard map" here means the complete set of commonly-charted reflexes across every body system (roughly 30-45 per sole), not every micro-variant point (charts differ on the finest sub-points). Finer points accrete via the admin editor.

---

## Scope

### In scope
- Engine: generalize the outline dispatch to any outline frame (`renderChart`, `refToScreen`), `wire()` allow-list += foot.
- Store: register the foot system; a foot-seed-validity test.
- Page: Foot dropdown option.
- Seed: full both-soles standard reflexology chart, grouped by body correspondence, with anchors.
- The iris and ear render and behave identically (the frame generalization keeps `unit_circle` and `ear_outline` on their existing paths).

### Explicitly deferred
- Per-side mirrored outline (one shared sole outline for now).
- Dorsal/medial/lateral foot views (sole only).
- The iris/ear full-map backfill (separate follow-on slice).
- polygon/path geometry types; ML anchoring; Atlas cluster targets on foot zones.

## Success criteria
- Selecting **Foot** in the dropdown draws the sole outline with the reflex points as labeled dots; the Side selector offers Left and Right and switches soles; the body-correspondence toggles filter points; clicking a point opens its panel ("Liver, digestive, right sole ...").
- Both soles are populated with their correct lateralized reflex sets.
- Uploading a sole photo and tapping big-toe/heel/little-toe warps the points onto the foot (photo stays in-browser, no upload).
- `/body-map?system=foot&side=right&zone=<id>` opens the right sole centered on that reflex; unknown params fall back without error.
- The iris and ear are unchanged; `pytest tests/test_bodymap_store.py` passes including the foot-seed-validity test.
