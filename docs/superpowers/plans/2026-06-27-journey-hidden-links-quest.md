# Journey "Find the Hidden Links" Quest — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Turn the journey-shell map into a gamified, ordered "Find the Hidden Links" onboarding quest over the blessed Shire scene — invisible audio-proximity hotspots, two-step gating per stage, progressive coupons — all behind `JOURNEY_SHELL_ENABLED`.

**Architecture:** Interface-only over the existing injected shell. The four `begin_funnel` engine keys (`scan/find/heal/give`) and all routes/points are untouched; this adds (a) rebuilt static art + a `scene` config block in `shell-map.json`, (b) a new self-contained `static/journey-quest.{js,css}` module that the shell loads to render the hunt overlay + audio + rewards, and (c) client-side `localStorage` progress (server mirror is a later increment). The working reference implementation is `~/Downloads/journey-ribbon-samples/landmarks/hunt-prototype.html`; Phase-2 tasks **port** its functions into the module rather than re-deriving them.

**Tech Stack:** Python 3 + Flask (existing app), vanilla ES5-style JS (match `shell.js` — no build step, no framework), Web Audio API + `speechSynthesis` for the prototype's synth audio stand-ins, Pillow (build-time only) for asset generation, pytest for unit tests, Playwright/claude-in-chrome for render-verify.

## Global Constraints

- **Flag:** all user-visible behavior ships behind the existing `JOURNEY_SHELL_ENABLED` (app already gates injection on it). Add a second nested flag `JOURNEY_QUEST_ENABLED` (default False) so the quest can ship dark independently of the plain shell.
- **No engine change:** do not edit `begin_funnel.JOURNEY_STEPS`, `/begin/state`, points, or routing. Display-only + additive endpoints.
- **Engine keys stay:** `scan / find / heal / give`. `home` is a wayfinding entry that exists ONLY in the `scene` block, never in `lands` (the validator checks `lands` keys against `JOURNEY_STEPS`).
- **Trademark display names (exact, verbatim):** `scan` → `Wellness Whispering`; `find` → `Remedy Match`; `heal` → `Accelerated Self Healing™` (note the ™); `give` → `Healing Oasis`; the home entry → `Home`.
- **Blessed scene source:** `~/Downloads/journey-ribbon-samples/landmarks/5-journey-unified-v12-trumpet.png` (1328×800).
- **Hotspot coordinates** (% of scene; center x / center y / width / height): home `14/57/13/23`; scan `43/35/12/19`; find `64/56/13/17`; heal `62/44/14/13`; give `72/18/16/27`.
- **Quest order:** `home → scan → find → heal → give`.
- **Reward model:** single growing personal coupon, 5/10/15% as 1/2/3 distinct paths (chat/video/hunt) are completed across the journey, cap 15%; coupon #1 issued at first stage completion; coupon #2 (giftable) at full completion. Never penalized by sequence; finding a hidden link always pays out.
- **Sound quality is the top priority** (Glen). Audio is asset-driven via a manifest; the Web-Audio synth from the prototype is the *fallback* until real recordings land (Glen's voice lines + 172 Hz Tibetan bowl confirmed for real recording).
- **Render-verify is mandatory** for any frontend task (a prior journey-shell change shipped a runtime `ui is not defined` break to prod that injection-only checks missed). Assert DOM + zero console errors in a headless browser, not just that the script is served.
- **Worktree:** all work in `/tmp/wt-deploy-chat-d7c6fc53` on branch `sess/d7c6fc53`.
- **Local test command** (app imports validate Pinecone over the network at import, and prd's `DATA_DIR=/data` isn't writable locally):
  `doppler run -p remedy-match -c prd -- env DATA_DIR=/tmp/qtest python3 -m pytest <args>` (run `mkdir -p /tmp/qtest` first). Pure-module tests (`shell_nav`, config) also run under that wrapper; they don't need network but the wrapper is harmless.

---

# Phase 1 — Assets + Config (foundation, fully unit-tested)

## Task 1: Rebuild journey art from the blessed v12 scene

**Files:**
- Modify: `scripts/build_journey_assets.py` (source, dims, crops, anchors)
- Create (build output, committed): `static/journey/scene.webp` (overwrite), `static/journey/thumb-{home,scan,find,heal,give}.webp` (overwrite)
- Test: `tests/test_journey_assets.py`

**Interfaces:**
- Produces: `static/journey/scene.webp` at aspect ≈ 1328/800 (1.66), < 300 KB; five `thumb-<key>.webp` files. Phase 2/the config consume `/static/journey/scene.webp` + `/static/journey/thumb-<key>.webp`.

- [ ] **Step 1: Write the failing test**

```python
# tests/test_journey_assets.py
from pathlib import Path
from PIL import Image

OUT = Path(__file__).resolve().parent.parent / "static" / "journey"
KEYS = ["home", "scan", "find", "heal", "give"]

def test_scene_is_v12_aspect_and_small():
    p = OUT / "scene.webp"
    assert p.exists(), "scene.webp missing — run scripts/build_journey_assets.py"
    w, h = Image.open(p).size
    assert abs((w / h) - (1328 / 800)) < 0.02, f"scene aspect {w}x{h} is not the v12 1328x800 scene"
    assert p.stat().st_size < 300 * 1024, "scene.webp over 300 KB"

def test_all_five_thumbs_exist():
    for k in KEYS:
        t = OUT / f"thumb-{k}.webp"
        assert t.exists(), f"thumb-{k}.webp missing"
        assert t.stat().st_size < 30 * 1024
```

- [ ] **Step 2: Run it; verify it fails**

Run: `mkdir -p /tmp/qtest && doppler run -p remedy-match -c prd -- env DATA_DIR=/tmp/qtest python3 -m pytest tests/test_journey_assets.py -v`
Expected: `test_scene_is_v12_aspect_and_small` FAILS (committed scene is 1344×768, aspect 1.75).

- [ ] **Step 3: Update the build script** — repoint to v12 and replace the crop/anchor tables.

In `scripts/build_journey_assets.py`:
- Change `DEFAULT_SRC`:
```python
DEFAULT_SRC = Path.home() / "Downloads" / "journey-ribbon-samples" / "landmarks" / "5-journey-unified-v12-trumpet.png"
```
- Change `SCENE_MAX_W = 1328`.
- Replace `THUMB_CROPS` (square-ish context boxes around each hotspot, fractions l,t,r,b):
```python
THUMB_CROPS = {
    "home": (0.03, 0.46, 0.25, 0.68),   # the green hobbit door
    "scan": (0.32, 0.24, 0.54, 0.46),   # Glendalf's ear & cupped hand
    "find": (0.53, 0.45, 0.75, 0.67),   # the remedy bottle & hands
    "heal": (0.51, 0.33, 0.73, 0.55),   # the path into the distance
    "give": (0.60, 0.07, 0.84, 0.31),   # the glowing cathedral
}
```
- Replace `LABEL_ANCHORS` with the spec hotspots (kept for the printout that feeds shell-map's scene block):
```python
HOTSPOTS = {
    "home": {"x": 14, "y": 57, "w": 13, "h": 23},
    "scan": {"x": 43, "y": 35, "w": 12, "h": 19},
    "find": {"x": 64, "y": 56, "w": 13, "h": 17},
    "heal": {"x": 62, "y": 44, "w": 14, "h": 13},
    "give": {"x": 72, "y": 18, "w": 16, "h": 27},
}
```
- In `main()`, change the docstring/printout line `LABEL_ANCHORS` → `HOTSPOTS` (the final `print(json.dumps(...))` should dump `HOTSPOTS`).

- [ ] **Step 4: Run the build script**

Run: `cd /tmp/wt-deploy-chat-d7c6fc53 && python3 scripts/build_journey_assets.py`
Expected: prints `source: …5-journey-unified-v12-trumpet.png (1328x800)`, `scene.webp: 1328x800 …KB` (under 300), five `thumb-*.webp` lines, then the `HOTSPOTS` JSON.

- [ ] **Step 5: Run the test; verify it passes**

Run: `doppler run -p remedy-match -c prd -- env DATA_DIR=/tmp/qtest python3 -m pytest tests/test_journey_assets.py -v`
Expected: PASS (both tests).

- [ ] **Step 6: Eyeball the assets** — open `static/journey/scene.webp` and the five thumbs; confirm scene = the v12 wizard/woman/cathedral scene and each thumb frames its hotspot (home=door, etc.). This is the Phase-1 art gate.

- [ ] **Step 7: Commit**

```bash
git add scripts/build_journey_assets.py static/journey/scene.webp static/journey/thumb-*.webp tests/test_journey_assets.py
git commit -m "feat(journey): rebuild shell art from blessed v12 scene + asset guard test"
```

---

## Task 2: Trademark names + `scene` block in shell-map.json + validation

**Files:**
- Modify: `static/shell-map.json` (land names → trademarks; add `thumb` per land; add top-level `scene` block)
- Modify: `shell_nav.py` (`validate_shell_map` — validate names, thumbs, scene block)
- Test: `tests/test_shell_map_config.py` (add assertions)

**Interfaces:**
- Produces: `shell-map.json` gains `lands.<key>.name` (trademark), `lands.<key>.thumb`, and a top-level `scene = {image, order, hotspots:{<home+4 keys>:{x,y,w,h,sound}}}`. Phase 2 reads `scene` to build the overlay.
- `validate_shell_map(cfg, land_keys)` keeps its signature; now also returns errors for empty names, missing thumbs, missing/!numeric scene fields.

- [ ] **Step 1: Write the failing tests** (append to `tests/test_shell_map_config.py`)

```python
def test_trademark_names_present():
    cfg = json.loads(CFG.read_text())
    names = {k: v["name"] for k, v in cfg["lands"].items()}
    assert names["scan"] == "Wellness Whispering"
    assert names["find"] == "Remedy Match"
    assert names["heal"] == "Accelerated Self Healing™"  # ™
    assert names["give"] == "Healing Oasis"

def test_scene_block_image_and_thumbs_exist_on_disk():
    base = CFG.parent  # static/
    cfg = json.loads(CFG.read_text())
    img = cfg["scene"]["image"].lstrip("/").split("static/", 1)[-1]
    assert (base / img).exists(), f"scene image missing: {img}"
    for k, land in cfg["lands"].items():
        rel = land["thumb"].lstrip("/").split("static/", 1)[-1]
        assert (base / rel).exists(), f"thumb missing for {k}: {land['thumb']}"

def test_scene_hotspots_numeric_for_home_and_four_lands():
    cfg = json.loads(CFG.read_text())
    hs = cfg["scene"]["hotspots"]
    for key in ["home", "scan", "find", "heal", "give"]:
        spot = hs[key]
        for f in ("x", "y", "w", "h"):
            assert isinstance(spot[f], (int, float)), f"{key}.{f} not numeric"

def test_validator_flags_bad_scene():
    bad = {"lands": {"scan": {"name": "x", "category": "scan", "intrigue": "y", "thumb": "/t.webp"}},
           "categories": {"scan": {"icon": "🌀"}},
           "scene": {"image": "", "order": [], "hotspots": {}}}
    errs = shell_nav.validate_shell_map(bad, ["scan", "find", "heal", "give"])
    assert any("scene.image" in e for e in errs)
    assert any("hotspot" in e for e in errs)
```

- [ ] **Step 2: Run; verify failure**

Run: `doppler run -p remedy-match -c prd -- env DATA_DIR=/tmp/qtest python3 -m pytest tests/test_shell_map_config.py -v`
Expected: the four new tests FAIL (old config has poetic names, no `scene`, no `thumb`; validator has no scene logic).

- [ ] **Step 3: Update `static/shell-map.json`**

```json
{
  "lands": {
    "scan": {"name": "Wellness Whispering", "category": "scan", "thumb": "/static/journey/thumb-scan.webp",
             "intrigue": "Your body is already speaking. Step in and listen.",
             "featured": {"product_slug": "terrain-restore", "product_name": "Terrain Restore", "healing_power": "supports your foundational terrain"}},
    "find": {"name": "Remedy Match", "category": "find", "thumb": "/static/journey/thumb-find.webp",
             "intrigue": "See the one remedy your body is asking for.",
             "featured": {"product_slug": "nous-energy", "product_name": "Nous Energy", "healing_power": "supports clear, energized focus"}},
    "heal": {"name": "Accelerated Self Healing™", "category": "heal", "thumb": "/static/journey/thumb-heal.webp",
             "intrigue": "Where the root causes finally settle.",
             "featured": {"product_slug": "microbiome", "product_name": "Microbiome", "healing_power": "supports your gut and inner terrain"}},
    "give": {"name": "Healing Oasis", "category": "give", "thumb": "/static/journey/thumb-give.webp",
             "intrigue": "the gift of healing"}
  },
  "categories": {
    "scan": {"icon": "🌀", "hue": "#4aa3a2"},
    "find": {"icon": "🔮", "hue": "#7a6cc4"},
    "heal": {"icon": "🌿", "hue": "#5aa36a"},
    "give": {"icon": "✨", "hue": "#caa64a"}
  },
  "scene": {
    "image": "/static/journey/scene.webp",
    "home": {"name": "Home", "thumb": "/static/journey/thumb-home.webp", "href": "/"},
    "order": ["home", "scan", "find", "heal", "give"],
    "hotspots": {
      "home": {"x": 14, "y": 57, "w": 13, "h": 23, "sound": "creak"},
      "scan": {"x": 43, "y": 35, "w": 12, "h": 19, "sound": "whisper"},
      "find": {"x": 64, "y": 56, "w": 13, "h": 17, "sound": "chaching"},
      "heal": {"x": 62, "y": 44, "w": 14, "h": 13, "sound": "doppler"},
      "give": {"x": 72, "y": 18, "w": 16, "h": 27, "sound": "oasis"}
    }
  }
}
```

- [ ] **Step 4: Extend `validate_shell_map` in `shell_nav.py`** — add, before `return errors`:

```python
    # land display fields
    for key, land in lands.items():
        if not (land or {}).get("name"):
            errors.append(f"land '{key}' has empty name")
        if not (land or {}).get("thumb"):
            errors.append(f"land '{key}' missing thumb")
    # scene block (optional, but if present must be well-formed)
    scene = (cfg or {}).get("scene")
    if scene is not None:
        if not scene.get("image"):
            errors.append("scene.image is empty")
        spots = scene.get("hotspots") or {}
        expected = set(valid) | {"home"}
        for key in expected:
            spot = spots.get(key)
            if not spot:
                errors.append(f"scene hotspot '{key}' missing")
                continue
            for f in ("x", "y", "w", "h"):
                if not isinstance(spot.get(f), (int, float)):
                    errors.append(f"scene hotspot '{key}.{f}' not numeric")
```

- [ ] **Step 5: Run; verify pass**

Run: `doppler run -p remedy-match -c prd -- env DATA_DIR=/tmp/qtest python3 -m pytest tests/test_shell_map_config.py -v`
Expected: PASS (all, including `test_shipped_config_is_valid` and the four new ones).

- [ ] **Step 6: Commit**

```bash
git add static/shell-map.json shell_nav.py tests/test_shell_map_config.py
git commit -m "feat(journey): trademark names + scene/hotspot block + validation"
```

---

# Phase 2 — The quest module (port the prototype)

> These tasks build the interactive experience by porting `hunt-prototype.html` into a shell-loaded module. They are **render-verified** (DOM + zero console errors in a headless browser), not unit-tested — the logic is DOM/Web-Audio/timing. Each task ends with a render-verify command and a screenshot for Glen.

## Task 3: Quest module scaffold + scene overlay (no audio yet)

**Files:**
- Create: `static/journey-quest.js`, `static/journey-quest.css`
- Modify: `shell_nav.py` (`inject_shell_html` — also emit the quest assets + `questEnabled` flag when on), `app.py` (pass a new `JOURNEY_QUEST_ENABLED` into the injector), `tests/test_journey_shell_inject.py` (assert quest assets present only when enabled)

**Interfaces:**
- Consumes: `window.__SHELL__.questEnabled` (bool); `shell-map.json` `scene` block (Task 2).
- Produces: `window.__JQUEST__` with `{ open(), close(), state }`; renders a `.jq-overlay` containing `<img class="jq-scene">` + five invisible `<button class="jq-hot" data-key=…>` positioned from `scene.hotspots`. Clicking a revealed/unlocked hotspot calls the land's href (reuse `shell.js`'s external/internal rule). The shell's existing map button (`ui.mapBtn`) opens this overlay instead of the pavilion overlay when `questEnabled`.

- [ ] **Step 1: Add the flag in `app.py`.** Find where `inject_shell_html(...)` is called in the `after_request` hook (search `inject_shell_html`). Add a module-level `JOURNEY_QUEST_ENABLED = os.environ.get("JOURNEY_QUEST_ENABLED", "").lower() in ("1","true","yes")` near `JOURNEY_SHELL_ENABLED`, and pass `quest_enabled=JOURNEY_QUEST_ENABLED` to the inject call.

- [ ] **Step 2: Write/extend the inject test (TDD for the server bit)** in `tests/test_journey_shell_inject.py`:

```python
def test_inject_adds_quest_assets_when_enabled():
    out = shell_nav.inject_shell_html("<head></head>", "funnel", quest_enabled=True)
    assert "/static/journey-quest.js" in out
    assert '"questEnabled":true' in out

def test_inject_omits_quest_assets_when_disabled():
    out = shell_nav.inject_shell_html("<head></head>", "funnel", quest_enabled=False)
    assert "/static/journey-quest.js" not in out
    assert '"questEnabled":false' in out
```

Run: `doppler run -p remedy-match -c prd -- env DATA_DIR=/tmp/qtest python3 -m pytest tests/test_journey_shell_inject.py -v` → these two FAIL.

- [ ] **Step 3: Extend `inject_shell_html`** signature to `(html, mode, rewards1b=False, rewards_gift=False, quest_enabled=False)`. Add `qe = "true" if quest_enabled else "false"`, include `"questEnabled":{qe}` in the `window.__SHELL__` object, and when `quest_enabled` append `<link rel="stylesheet" href="/static/journey-quest.css"><script defer src="/static/journey-quest.js"></script>` to `tags`. Re-run the test → PASS (and the existing inject tests still pass — default `quest_enabled=False` keeps `__SHELL__` shape additive).

- [ ] **Step 4: Create `static/journey-quest.css`** — port the prototype's `.stage`/`.hot`/`.pop`/`#engage`/`#reward`/`.navitem` styles, renaming the prefix to `.jq-` and scoping under `.jq-overlay`. Use the prototype file as the source of truth for the visual rules (background image cover, invisible hotspots, cream popup card, reward toast, locked/current/unlocked nav states with grayscale filter). Include `@media (hover:none)` rules for the mobile hint (Task 6).

- [ ] **Step 5: Create `static/journey-quest.js`** scaffold (IIFE, ES5 style like shell.js):
  - Guard `if (!(window.__SHELL__ && window.__SHELL__.questEnabled)) return;`
  - Fetch `/static/shell-map.json`, read `scene`.
  - Build `.jq-overlay` with the scene `<img>` and five `.jq-hot` buttons positioned `left/top/width/height` from `scene.hotspots` (% values, `transform: translate(-50%,-50%)`).
  - Expose `window.__JQUEST__ = { open, close, state }`. For this task, hotspots simply navigate to the land href on click (audio/locking come next).
  - Port the prototype's `isExternal`/navigation choice (or reuse — the shell already tags external links).

- [ ] **Step 6: Hook the shell's map button.** In `static/shell.js` `boot()`, after building the overlay, add: `if (window.__JQUEST__) { ui.mapBtn.onclick = function(){ window.__JQUEST__.open(); }; }` (guard so the pavilion overlay remains when quest disabled). Keep it minimal and idempotent.

- [ ] **Step 7: Render-verify** (local app, both flags on):

```bash
mkdir -p /tmp/qtest
JOURNEY_SHELL_ENABLED=1 JOURNEY_QUEST_ENABLED=1 \
  doppler run -p remedy-match -c prd -- env DATA_DIR=/tmp/qtest python3 -m flask --app app run -p 5055 &
# then headless-load http://127.0.0.1:5055/begin, click the map button, and assert:
#  - .jq-overlay.open exists, .jq-scene <img> has naturalWidth>0
#  - exactly 5 .jq-hot buttons exist with correct data-key
#  - console has ZERO errors
```
Use the claude-in-chrome MCP (navigate, click, `read_console_messages`) or a small Playwright script modeled on `tests/test_atlas_e2e.py`. Capture a screenshot for Glen. Kill the server when done.

- [ ] **Step 8: Commit**

```bash
git add static/journey-quest.js static/journey-quest.css shell_nav.py app.py static/shell.js tests/test_journey_shell_inject.py
git commit -m "feat(journey): quest scaffold + scene overlay behind JOURNEY_QUEST_ENABLED"
```

## Task 4: Ribbon lock states + ordered progress (localStorage)

**Files:** Modify `static/journey-quest.js`, `static/journey-quest.css`, `static/shell.js` (let the quest annotate the ribbon lands).

**Interfaces:**
- Produces: `__JQUEST__.state` = per-key `{found, done}` + `paths` set, persisted to `localStorage["jquest.v1"]`. A `render()` that toggles `.jq-locked/.jq-current/.jq-unlocked` on both the ribbon lands and the overlay hotspots; `curIdx()` (first not-done in `scene.order`). Ordering enforced: clicking a non-current hotspot is a no-op (gentle).
- Port `load/save/curIdx/render/tryFind` from the prototype (adapt selectors to `.jq-` + the real ribbon land elements rendered by `shell.js renderLands`).

- [ ] Steps: port the state + render logic; have `shell.js renderLands` add a stable `data-key` to each land element so the quest can find+style them; gray out locked lands, gold-pulse the current, full-color unlocked; persist; **render-verify** the lock→unlock transition (click home hotspot → Home land loses `.jq-locked`); zero console errors; screenshot; commit.

## Task 5: Audio engine — approach + arrival, asset-driven with synth fallback

**Files:** Create `static/journey-audio.js` (loaded by the quest); Create `static/journey/audio/manifest.json` (maps `<key>` → `{approach, arrival}` asset URLs, initially empty/absent); Modify `static/journey-quest.js` to call it.

**Interfaces:**
- Produces: `window.__JQAUDIO__ = { init(), setTarget(key), proximity(p), arrival(key), stopAll() }`. `init()` lazily creates the `AudioContext` on first user gesture. If a real asset exists in `manifest.json` for a cue it plays that (HTMLAudio/buffer); otherwise it falls back to the prototype's **synth** generator for that cue (`creak/whisper/chaching/doppler/oasis`, approach + arrival). `proximity(p)` (0..1) drives the current target's approach gain (faint floor + swell), exactly as the prototype's `approach.setProx`.
- Port `loopNoise/burst/tone/say/startApproach/stopApproach/playSound` from the prototype verbatim into this file; wrap them behind the manifest check.

- [ ] Steps: port the synth engine; add the manifest lookup + `<audio>` playback path; wire the overlay `mousemove` → compute proximity to the current hotspot → `proximity()`; on a successful find call `arrival(key)`; add a mute/volume control in the overlay and respect a stored `jquest.muted`; ensure the hunt is completable with sound off (visible "reveal zones"-style fallback hint already in prototype, gated to keyboard/touch/mute). **Render-verify** that no console errors occur and audio nodes start only after a gesture; screenshot. Commit.
- **Note:** real recordings (Glen's voice lines + 172 Hz bowl, etc.) are a separate production workstream; dropping files into `static/journey/audio/` + adding manifest entries upgrades cues with no code change.

## Task 6: Two-step gating, entry flow, progressive rewards (client-side)

**Files:** Modify `static/journey-quest.js`, `static/journey-quest.css`.

**Interfaces:**
- Produces: the full loop from the prototype — clean entry (UI appears after first video/chat touch), ordered find → `pop` card → `engage` panel (video|chat) → mark `done` → advance; rewards: `huntReward` (every first-find, fires the coupon + arrival fanfare), `gateReward` (per gate; finale on all-done with the giftable coupon); a single growing personal `couponPct` (5/10/15 by distinct `paths` count, cap 15) stored in state and shown in the toast; mobile subtle hint (current target name + one-time line) via the `@media (hover:none)` path.
- Port `enter/tryFind/showPop/openEngage/engageDone/showReward/huntReward/gateReward` from the prototype; replace the simulated engage buttons' targets with the real land hrefs (video/chat rails open the actual stage page in a new context, then return marks the rail done — for v1 the "engage" can mark done on click-through; full rail-completion detection is Phase 3).

- [ ] Steps: port the loop; compute `couponPct` from `paths`; wire the entry gate; **render-verify** a full 5-stage playthrough (DOM: all five lands reach `.jq-unlocked`, finale toast shows the giftable coupon, zero console errors); screenshot the finale for Glen; commit.

---

# Phase 3 — Persistence & coupons (server, additive)

> Optional for a dark-flag pilot (Phase 2 works client-side). Build when moving toward go-live so progress + coupons survive across devices for known members.

## Task 7: Member progress mirror

**Files:** Create `dashboard/journey_quest.py` (pure helpers: serialize/merge quest state); Modify `app.py` (add `GET/POST /api/journey/quest-state`, console-key-free, session/member-scoped, fail-open); Test `tests/test_journey_quest_state.py`.

**Interfaces:**
- Produces: `POST /api/journey/quest-state` (body = client state) upserts for the authenticated member; `GET` returns merged state. `journey-quest.js` calls these when a member session exists, else stays localStorage-only. Storage reuses the existing per-member store pattern (follow `dashboard/journal_store.py` sqlite pattern; do NOT invent a new DB).
- TDD: pure merge helper (`merge_quest(local, server) -> dict`, union of found/done/paths, max couponPct) gets real unit tests; the route gets a Flask test-client test mirroring `test_journey_shell_inject.py`'s fixture.

- [ ] Steps: TDD the merge helper; add the routes; wire the client to sync on open/close for members; run pytest; commit.

## Task 8: Progressive coupon issuance

**Files:** Modify `app.py` / the existing `/api/journey/claim-coupon` + `/api/journey/wallet` path (already present in `shell.js`), or add `/api/journey/quest-coupon`; reuse the existing wallet/coupon mechanism rather than a parallel one.

**Interfaces:**
- Produces: at first stage completion, issue coupon #1 at the current `couponPct`; upgrade its pct as `paths` grow (cap 15); at full completion issue coupon #2 as a giftable (mirror the existing `REWARDS_GIFT` / `activate-gifting` flow). Coordinate exact redemption with the existing store/membership coupon mechanism (spec §10.3 — confirm before building).

- [ ] Steps: confirm coupon redemption mechanics with Glen; TDD the issuance helper; wire to the quest; pytest; commit.

---

# Phase 4 — Render-verify sweep + go-live

- [ ] Run the full unit suite: `doppler run -p remedy-match -c prd -- env DATA_DIR=/tmp/qtest python3 -m pytest tests/test_journey_assets.py tests/test_shell_map_config.py tests/test_journey_shell_inject.py -v` — all green (proves names/assets/inject are display-only and the flag gating holds).
- [ ] **Full render-verify** with both flags on: load a shell page, open the quest, play through all five stages, assert scene + five trademark labels-on-hover (incl. ™) + ribbon lock→unlock + reward toasts render and **console has zero errors**. Capture ribbon + open-map + finale screenshots for Glen.
- [ ] Confirm `/begin/state` JSON is byte-unchanged (display-only) and that with `JOURNEY_QUEST_ENABLED` unset the plain shell behaves exactly as before (pavilion overlay, no quest assets).
- [ ] Commit, push `sess/d7c6fc53`, open PR. Before `gh pr create`, run `git diff --name-only origin/main..HEAD | grep superpowers` and `git rm --cached` any leaked `.superpowers/sdd/*` scratch. Imagery + names + quest ship together, dark behind `JOURNEY_QUEST_ENABLED`.

---

## Self-review notes (coverage)

- Spec §1 scene → Task 1. §2 names/keys/hotspots → Task 2 (+ Global Constraints). §3 mechanic/entry/ordering → Tasks 4 & 6. §4 audio two-phase + proximity + asset-driven + mute fallback → Task 5. §5 progressive single coupon → Task 6 (display) + Task 8 (issuance). §6 ribbon states + thumb icons → Tasks 1, 4. §7 persistence/mobile/a11y → Tasks 6 (mobile/a11y), 7 (member mirror). §8 assets → Tasks 1, 5. §9 out-of-scope respected (no engine change; full ambient bed deferred). §10 open items: reward scope RESOLVED; LOTR-line rights + coupon redemption flagged in Tasks 5/8 as confirm-before-ship.
- Flag strategy keeps every increment shippable dark; Phase 2 alone is a complete client-side pilot.
- Audio realism (Glen's #1 priority) is decoupled via the manifest so recordings land without code change.
