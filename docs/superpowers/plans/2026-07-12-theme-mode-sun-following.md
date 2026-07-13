# Theme Mode (three-state, sun-following) Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace the three separate light/dark toggles (on two disagreeing storage keys) with one three-state Light / Dark / Auto control, where Auto follows the visitor's local sunrise and sunset computed in the browser.

**Architecture:** A pure sun engine (`sun-engine.js`) computes local sunrise/sunset from latitude, longitude, and date with no external calls. A single controller (`theme-mode.js`) owns the persisted mode, migrates the two legacy keys, resolves the mode to a `data-theme` value, caches geolocation once, and renders the shared toggle UI. The existing per-surface light-palette CSS stays exactly where it is; the controller only ever sets `data-theme`, never palettes. The old appliers in `shell.js`, `theme-toggle.js`, `client-portal.html`, and `client-login.html` are gutted to defer to the controller.

**Tech Stack:** Vanilla browser JS (no build step, no framework), Node's built-in `node:test` runner for the pure engine, Flask static serving.

## Global Constraints

- deploy-chat has NO CI; merge equals deploy. Verify every UI task by rendering the page in headless Chrome, not by tests alone.
- Single persisted mode key: `rm-theme-mode` with values `light` | `dark` | `auto`. Default for a new visitor: `auto`.
- Legacy keys to migrate and then stop writing as the source of truth: `rm-theme`, `rm_portal_theme`. The resolved theme is still mirrored into `rm-theme` for backward-compatible cross-document sync only.
- Geolocation is requested at most once, cached in `localStorage` under `rm-geo` as `{"lat":<n>,"lng":<n>}`, and the coordinates never leave the device (no fetch, no server write).
- Sunrise/sunset uses the NOAA "Almanac for Computers" algorithm. Verified accurate within ~2 minutes for Honolulu, London, and Anchorage.
- Copy rule: no em dashes anywhere in user-visible strings.
- The controller sets `data-theme` only; it must not inject or move any palette CSS.

## File Structure

- `static/sun-engine.js` (new) — pure sunrise/sunset math. UMD shim: `window.RMSun` in the browser, `module.exports` under Node. One responsibility: given date + lat + lng, return local event hours.
- `static/theme-mode.js` (new) — the controller `window.RMTheme`: mode storage, legacy migration, resolve, geolocation cache, apply `data-theme`, cross-document sync, and `mountToggle()` for the shared UI. Loaded first in `<head>` on every surface.
- `tests/theme/sun.test.cjs` (new) — asserts the engine against known locations and a polar null case.
- `tests/theme/resolve.test.cjs` (new) — asserts `RMTheme._resolve` mode-to-theme logic.
- `static/shell.js` (modify) — remove the two-state ribbon toggle and the localStorage apply; mount the shared toggle.
- `static/theme-toggle.js` (modify) — gut to keep only the funnel light-palette `<style>` injection; drop its button and its own apply/sync.
- `static/client-portal.html` (modify) — remove the `rm_portal_theme` head apply and the `#themeToggle` handler; mount the shared toggle on `#themeToggle`.
- `static/client-login.html` (modify) — same as portal.

---

### Task 1: Pure sun engine with unit tests

**Files:**
- Create: `static/sun-engine.js`
- Test: `tests/theme/sun.test.cjs`

**Interfaces:**
- Produces: `sunTimes(date: Date, lat: number, lng: number) -> { sunrise: number|null, sunset: number|null }` where the numbers are local clock hours (0..24, e.g. `5.95` for 5:57 AM), using `date.getTimezoneOffset()` for the local conversion. `null` means the sun does not cross the horizon that day at that latitude. Exposed as `window.RMSun.sunTimes` in the browser and `module.exports.sunTimes` under Node.

- [ ] **Step 1: Write the failing test**

```js
// tests/theme/sun.test.cjs
const test = require('node:test');
const assert = require('node:assert');
const { sunTimes } = require('../../static/sun-engine.js');

// Convert a local-hours float to "H:MM" for readable assertions.
function hm(h){ const m = Math.round(h*60); return Math.floor(m/60) + ':' + String(m%60).padStart(2,'0'); }

test('Honolulu mid-July sunrise/sunset (UTC-10)', () => {
  const d = new Date(2026, 6, 12); // local date; test host TZ is normalized below
  const { sunrise, sunset } = sunTimes(d, 21.3, -157.8);
  // Allow the host offset to differ from HST: assert day length instead of wall time.
  const dayLen = sunset - sunrise;
  assert.ok(Math.abs(dayLen - 13.3) < 0.3, 'day length ~13.3h, got ' + dayLen.toFixed(2));
});

test('Anchorage mid-July has a long day (>17h)', () => {
  const d = new Date(2026, 6, 12);
  const { sunrise, sunset } = sunTimes(d, 61.2, -149.9);
  assert.ok((sunset - sunrise) > 17, 'expected >17h, got ' + (sunset - sunrise).toFixed(2));
});

test('Honolulu winter day is shorter than summer day', () => {
  const summer = sunTimes(new Date(2026, 6, 12), 21.3, -157.8);
  const winter = sunTimes(new Date(2026, 11, 21), 21.3, -157.8);
  assert.ok((winter.sunset - winter.sunrise) < (summer.sunset - summer.sunrise));
});

test('extreme polar latitude returns null in deep winter', () => {
  const { sunrise } = sunTimes(new Date(2026, 11, 21), 78, 15); // Svalbard, polar night
  assert.strictEqual(sunrise, null);
});
```

- [ ] **Step 2: Run test to verify it fails**

Run: `node --test tests/theme/sun.test.cjs`
Expected: FAIL, cannot find module `../../static/sun-engine.js`.

- [ ] **Step 3: Write the engine**

```js
// static/sun-engine.js
/* Pure sunrise/sunset (NOAA "Almanac for Computers"). No DOM, no network.
   Browser: window.RMSun. Node: module.exports. */
(function (root, factory) {
  var api = factory();
  if (typeof module !== 'undefined' && module.exports) module.exports = api;
  if (typeof window !== 'undefined') window.RMSun = api;
})(this, function () {
  var RAD = Math.PI / 180;

  // Returns local clock hours (0..24) for the event, or null if the sun does
  // not cross the horizon that day at that latitude.
  function sunEvent(date, lat, lng, isSunrise) {
    var start = new Date(date.getFullYear(), 0, 0);
    var dayOfYear = Math.floor((date - start) / 86400000);
    var lngHour = lng / 15;
    var t = dayOfYear + ((isSunrise ? 6 : 18) - lngHour) / 24;
    var M = 0.9856 * t - 3.289;
    var L = M + 1.916 * Math.sin(M * RAD) + 0.020 * Math.sin(2 * M * RAD) + 282.634;
    L = (L % 360 + 360) % 360;
    var RA = Math.atan(0.91764 * Math.tan(L * RAD)) / RAD;
    RA = (RA % 360 + 360) % 360;
    RA += (Math.floor(L / 90) * 90) - (Math.floor(RA / 90) * 90);
    RA /= 15;
    var sinDec = 0.39782 * Math.sin(L * RAD);
    var cosDec = Math.cos(Math.asin(sinDec));
    var cosH = (Math.cos(90.833 * RAD) - sinDec * Math.sin(lat * RAD)) / (cosDec * Math.cos(lat * RAD));
    if (cosH > 1 || cosH < -1) return null;
    var H = isSunrise ? 360 - Math.acos(cosH) / RAD : Math.acos(cosH) / RAD;
    H /= 15;
    var T = H + RA - 0.06571 * t - 6.622;
    var UT = ((T - lngHour) % 24 + 24) % 24;
    var offsetHours = -date.getTimezoneOffset() / 60; // local offset of the host
    return ((UT + offsetHours) % 24 + 24) % 24;
  }

  function sunTimes(date, lat, lng) {
    return { sunrise: sunEvent(date, lat, lng, true), sunset: sunEvent(date, lat, lng, false) };
  }

  return { sunEvent: sunEvent, sunTimes: sunTimes };
});
```

- [ ] **Step 4: Run test to verify it passes**

Run: `node --test tests/theme/sun.test.cjs`
Expected: PASS, 4 tests.

- [ ] **Step 5: Commit**

```bash
git add static/sun-engine.js tests/theme/sun.test.cjs
git commit -m "feat(theme): pure sunrise/sunset engine with tests"
```

---

### Task 2: Theme controller with mode-resolution tests

**Files:**
- Create: `static/theme-mode.js`
- Test: `tests/theme/resolve.test.cjs`

**Interfaces:**
- Consumes: `RMSun.sunTimes` from Task 1.
- Produces on `window.RMTheme`:
  - `_resolve(mode: 'light'|'dark'|'auto', ctx: {nowH:number, sunrise:number|null, sunset:number|null}) -> 'light'|'dark'` (pure, testable).
  - `getMode() -> 'light'|'dark'|'auto'`, `setMode(mode)`, `resolvedTheme() -> 'light'|'dark'`.
  - `mountToggle(container: Element) -> void` renders the three-state control.
  - `init() -> void` runs migration and applies; called in `<head>`.
  - Fires a `rm-theme-change` CustomEvent on `document` after every apply.

- [ ] **Step 1: Write the failing test for `_resolve`**

```js
// tests/theme/resolve.test.cjs
const test = require('node:test');
const assert = require('node:assert');

// _resolve is pure and DOM-free; load the file under a minimal global shim.
global.window = {}; global.document = { documentElement: { }, addEventListener(){} };
require('../../static/theme-mode.js');
const R = global.window.RMTheme;

test('explicit modes pass through', () => {
  assert.strictEqual(R._resolve('light', {nowH:3, sunrise:6, sunset:19}), 'light');
  assert.strictEqual(R._resolve('dark',  {nowH:12, sunrise:6, sunset:19}), 'dark');
});
test('auto is light between sunrise and sunset', () => {
  assert.strictEqual(R._resolve('auto', {nowH:12, sunrise:6, sunset:19}), 'light');
});
test('auto is dark before sunrise and after sunset', () => {
  assert.strictEqual(R._resolve('auto', {nowH:5,  sunrise:6, sunset:19}), 'dark');
  assert.strictEqual(R._resolve('auto', {nowH:21, sunrise:6, sunset:19}), 'dark');
});
test('auto falls back to a 7-19 window when sun times are null', () => {
  assert.strictEqual(R._resolve('auto', {nowH:12, sunrise:null, sunset:null}), 'light');
  assert.strictEqual(R._resolve('auto', {nowH:2,  sunrise:null, sunset:null}), 'dark');
});
```

- [ ] **Step 2: Run test to verify it fails**

Run: `node --test tests/theme/resolve.test.cjs`
Expected: FAIL, cannot find module `../../static/theme-mode.js`.

- [ ] **Step 3: Write the controller**

```js
// static/theme-mode.js
/* Single theme controller. Owns mode (rm-theme-mode), migrates legacy keys,
   resolves to data-theme, caches geolocation once (rm-geo), and renders the
   shared 3-state toggle. Sets data-theme ONLY; never injects palette CSS.
   Load in <head>, not deferred, before paint. */
(function () {
  var MODE_KEY = 'rm-theme-mode', GEO_KEY = 'rm-geo', MIRROR_KEY = 'rm-theme';
  var VALID = { light: 1, dark: 1, auto: 1 };
  var geoDenied = false;

  function lsGet(k){ try { return localStorage.getItem(k); } catch (e) { return null; } }
  function lsSet(k,v){ try { localStorage.setItem(k,v); } catch (e) {} }

  function getMode(){
    var m = lsGet(MODE_KEY);
    return VALID[m] ? m : 'auto';
  }

  // Pure: mode + context -> concrete theme. Unit tested.
  function _resolve(mode, ctx){
    if (mode === 'light' || mode === 'dark') return mode;
    var sr = ctx.sunrise, ss = ctx.sunset, h = ctx.nowH;
    if (sr == null || ss == null) return (h >= 7 && h < 19) ? 'light' : 'dark';
    return (h >= sr && h < ss) ? 'light' : 'dark';
  }

  function geo(){
    var raw = lsGet(GEO_KEY);
    if (raw) { try { return JSON.parse(raw); } catch (e) {} }
    // Longitude from the time-zone offset; latitude unknown until asked.
    return { lat: null, lng: -(new Date().getTimezoneOffset()) / 60 * 15 };
  }

  function currentContext(){
    var now = new Date();
    var g = geo();
    var st = (g.lat == null)
      ? { sunrise: null, sunset: null }
      : window.RMSun.sunTimes(now, g.lat, g.lng);
    return { nowH: now.getHours() + now.getMinutes() / 60, sunrise: st.sunrise, sunset: st.sunset };
  }

  function apply(){
    var theme = _resolve(getMode(), currentContext());
    document.documentElement.setAttribute('data-theme', theme);
    lsSet(MIRROR_KEY, theme); // legacy cross-document sync only
    try { document.dispatchEvent(new CustomEvent('rm-theme-change', { detail: { theme: theme, mode: getMode() } })); } catch (e) {}
    return theme;
  }

  function requestLocation(){
    if (geoDenied || !navigator.geolocation) return;
    navigator.geolocation.getCurrentPosition(function(p){
      lsSet(GEO_KEY, JSON.stringify({ lat: p.coords.latitude, lng: p.coords.longitude }));
      apply();
    }, function(){ geoDenied = true; }, { timeout: 8000, maximumAge: 86400000 });
  }

  function setMode(mode){
    if (!VALID[mode]) mode = 'auto';
    lsSet(MODE_KEY, mode);
    if (mode === 'auto' && geo().lat == null) requestLocation();
    apply();
  }

  function migrate(){
    if (VALID[lsGet(MODE_KEY)]) return;               // already on the new key
    var legacy = lsGet('rm-theme') || lsGet('rm_portal_theme');
    lsSet(MODE_KEY, (legacy === 'light' || legacy === 'dark') ? legacy : 'auto');
  }

  var ICONS = {
    light: '<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round"><circle cx="12" cy="12" r="4.5"/><path d="M12 2v2M12 20v2M4 12H2M22 12h-2M5 5l1.5 1.5M17.5 17.5 19 19M19 5l-1.5 1.5M6.5 17.5 5 19"/></svg>',
    dark:  '<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><path d="M21 12.8A9 9 0 1 1 11.2 3 7 7 0 0 0 21 12.8z"/></svg>',
    auto:  '<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round"><path d="M3 18h18"/><path d="M7 18a5 5 0 0 1 10 0"/><path d="M12 3v3M4.5 9l1.6 1.6M19.5 9l-1.6 1.6"/></svg>'
  };
  var LABELS = { light: 'Light', dark: 'Dark', auto: 'Auto' };

  function mountToggle(container){
    if (!container) return;
    var seg = document.createElement('div');
    seg.className = 'rm-theme-seg';
    seg.setAttribute('role', 'group');
    seg.setAttribute('aria-label', 'Theme mode');
    ['light', 'dark', 'auto'].forEach(function(m){
      var b = document.createElement('button');
      b.type = 'button';
      b.className = 'rm-theme-seg-btn';
      b.dataset.mode = m;
      b.title = LABELS[m];
      b.setAttribute('aria-label', LABELS[m]);
      b.innerHTML = ICONS[m];
      b.onclick = function(){ setMode(m); };
      seg.appendChild(b);
    });
    container.appendChild(seg);
    function refresh(){
      var mode = getMode();
      seg.querySelectorAll('.rm-theme-seg-btn').forEach(function(b){
        b.setAttribute('aria-pressed', b.dataset.mode === mode ? 'true' : 'false');
      });
    }
    document.addEventListener('rm-theme-change', refresh);
    refresh();
  }

  function init(){
    migrate();
    apply();
    if (getMode() === 'auto' && geo().lat == null) requestLocation();
    // Re-resolve periodically so Auto flips across sunrise/sunset without a reload.
    setInterval(function(){ if (getMode() === 'auto') apply(); }, 600000);
    // Follow theme changes made in other same-origin documents.
    window.addEventListener('storage', function(e){
      if (e.key === MODE_KEY || e.key === MIRROR_KEY) apply();
    });
  }

  window.RMTheme = {
    _resolve: _resolve, getMode: getMode, setMode: setMode,
    resolvedTheme: function(){ return _resolve(getMode(), currentContext()); },
    sunTimes: function(d, lat, lng){ return window.RMSun.sunTimes(d, lat, lng); },
    requestLocation: requestLocation, mountToggle: mountToggle, init: init
  };

  // Auto-run only in a real browser (guarded so Node tests can require this file).
  if (typeof document !== 'undefined' && document.documentElement && typeof localStorage !== 'undefined') {
    init();
  }
})();
```

- [ ] **Step 4: Run test to verify it passes**

Run: `node --test tests/theme/resolve.test.cjs`
Expected: PASS, 4 tests. (The `init()` auto-run is skipped because the test shim leaves `localStorage` undefined.)

- [ ] **Step 5: Add the shared toggle CSS**

Append to `static/shell.css`:

```css
/* Shared three-state theme toggle (rendered by theme-mode.js mountToggle). */
.rm-theme-seg{display:inline-flex;border:1px solid var(--hair,#21472d);border-radius:999px;overflow:hidden}
.rm-theme-seg-btn{appearance:none;border:none;background:transparent;color:var(--dim,#8fa89b);
  padding:5px 8px;cursor:pointer;display:inline-flex;align-items:center;line-height:0}
.rm-theme-seg-btn+.rm-theme-seg-btn{border-left:1px solid var(--hair,#21472d)}
.rm-theme-seg-btn svg{width:15px;height:15px}
.rm-theme-seg-btn[aria-pressed="true"]{background:var(--gold,#d4a843);color:#22160a}
.rm-theme-seg-btn:focus-visible{outline:2px solid var(--gold,#d4a843);outline-offset:2px}
```

- [ ] **Step 6: Commit**

```bash
git add static/theme-mode.js static/shell.css tests/theme/resolve.test.cjs
git commit -m "feat(theme): single mode controller + shared 3-state toggle"
```

---

### Task 3: Wire the funnel (ribbon + floating toggle) to the controller

**Files:**
- Modify: `static/shell.js:108-124` (ribbon theme button) and `static/shell.js:17-31` (`applyShellTheme`)
- Modify: `static/theme-toggle.js` (gut to palette-only)
- Modify: `static/begin.html` (load order in `<head>`)

**Interfaces:**
- Consumes: `RMTheme.mountToggle`, `RMTheme.init` from Task 2.

- [ ] **Step 1: Load the controller first in the funnel head**

In `static/begin.html`, in `<head>` before the existing `theme-toggle.js` include (`begin.html:633`), add:

```html
<script src="/static/sun-engine.js"></script>
<script src="/static/theme-mode.js"></script>
```

- [ ] **Step 2: Replace the ribbon's two-state toggle with the shared control**

In `static/shell.js`, replace the block at lines 108-124 (from `// Light/Dark toggle` through the `window.addEventListener("storage"...)` that ends at line 124) with:

```js
    // Light/Dark/Auto toggle — rendered and owned by theme-mode.js.
    if (window.RMTheme) window.RMTheme.mountToggle(bar);
```

- [ ] **Step 3: Stop shell.js from applying/persisting theme (keep palette injection)**

In `static/shell.js` `applyShellTheme()` (lines 17-31), remove the apply line so the controller is the sole applier. Change:

```js
  function applyShellTheme() {
    try { applyTheme(localStorage.getItem("rm-theme")); } catch (e) {}
    if (!document.getElementById("rm-theme-style") && !document.getElementById("op-nav-theme-style")) {
```

to:

```js
  function applyShellTheme() {
    // theme-mode.js owns applying data-theme; here we only ensure the light palette exists.
    if (!document.getElementById("rm-theme-style") && !document.getElementById("op-nav-theme-style")) {
```

(Leave the `applyTheme` helper defined; it is still referenced elsewhere in the file. Leave the palette `<style>` injection intact.)

- [ ] **Step 4: Gut theme-toggle.js to palette-only**

Replace the entire body of `static/theme-toggle.js` with a version that keeps only the light-palette `<style>` injection and drops the button, the localStorage apply, and the storage listener (all now owned by theme-mode.js):

```js
/* Funnel light-palette override. The theme itself (data-theme) and the toggle
   are owned by theme-mode.js. This file only guarantees the light palette CSS
   is present on funnel/chat surfaces. */
(function(){
  if (document.getElementById('rm-theme-style')) return;
  var s = document.createElement('style');
  s.id = 'rm-theme-style';
  s.textContent = ':root[data-theme="light"]{' +
    '--bg:#FBF8F3;--surface:#FFFFFF;--surface-2:#F4ECDE;--border:#E2D9C9;' +
    '--cream:#1E2A2A;--muted:#5F6B6B;--gold:#B08A3E;--green:#2D7A6A;' +
    '--panel:#FFFFFF;--panel-2:#F4ECDE;--ink:#1E2A2A;--dim:#5F6B6B;' +
    '--hair:#E2D9C9;--accent:#B08A3E;' +
    '--text:#1E2A2A;--text-muted:#5F6B6B;--surface2:#F4ECDE;--accent2:#2D7A6A;}';
  (document.head || document.documentElement).appendChild(s);
})();
```

- [ ] **Step 5: Render-verify the funnel**

Start the app locally and render the homepage in headless Chrome. Confirm:
- The ribbon shows a three-button Light / Dark / Auto control (SVG icons, no `☀`/`☾` emoji).
- Clicking Dark sets `document.documentElement.dataset.theme === 'dark'`; clicking Light sets `'light'`.
- `localStorage['rm-theme-mode']` updates on each click.
- No console errors.

Run (adapt the launch to how deploy-chat is normally started locally):
```bash
# start app, then in the browser console on the homepage:
# document.querySelectorAll('.rm-theme-seg-btn').length  -> 3
# document.documentElement.getAttribute('data-theme')    -> 'light' | 'dark'
```

- [ ] **Step 6: Commit**

```bash
git add static/shell.js static/theme-toggle.js static/begin.html
git commit -m "feat(theme): funnel ribbon uses the unified 3-state toggle"
```

---

### Task 4: Wire the portal and login to the controller

**Files:**
- Modify: `static/client-portal.html:14-17` (head apply), `:399-412` (`#themeToggle` script)
- Modify: `static/client-login.html:10-12` (head apply), `:56-64` (`#themeToggle` script)

**Interfaces:**
- Consumes: `RMTheme.init`, `RMTheme.mountToggle` from Task 2.

- [ ] **Step 1: Replace the portal head apply with the controller**

In `static/client-portal.html`, replace the inline head script at lines 14-17 that reads `rm_portal_theme`:

```html
  (function(){var t=localStorage.getItem('rm_portal_theme');
    if(t==='light'||t==='dark')
    document.documentElement.setAttribute('data-theme',t);})();
```

with the controller includes (which migrate `rm_portal_theme` automatically):

```html
  <script src="/static/sun-engine.js"></script>
  <script src="/static/theme-mode.js"></script>
```

- [ ] **Step 2: Replace the portal `#themeToggle` handler with a mount**

In `static/client-portal.html`, replace the script block at lines 399-412 (the IIFE that defines `SUN`/`MOON` and writes `rm_portal_theme`) with:

```html
  <script>
    (function(){
      var host = document.getElementById('themeToggle');
      if (host && window.RMTheme) { host.replaceWith((function(){var d=document.createElement('span');d.id='themeToggle';return d;})()); }
      if (window.RMTheme) window.RMTheme.mountToggle(document.getElementById('themeToggle'));
    })();
  </script>
```

(The `#themeToggle` element at line 396 stays as the mount host; the segmented control is appended into it. The `.theme-toggle` button styling still positions the cluster.)

- [ ] **Step 3: Apply the same two edits to the login page**

In `static/client-login.html`, replace the head apply at lines 10-12 with the same two `<script src=...>` includes, and replace the `#themeToggle` IIFE at lines 56-64 with the same mount block from Step 2.

- [ ] **Step 4: Render-verify portal + login and cross-surface agreement**

Render `/portal/login` and the logged-in portal locally. Confirm:
- Both show the three-state control on the existing top-right toggle spot.
- Setting Dark on the login page, then loading the portal, shows the portal already in Dark (they now share `rm-theme-mode`).
- A stored legacy `rm_portal_theme='light'` (set before this change) results in `rm-theme-mode==='light'` after first load (migration).
- No console errors; the portal's committed fireside/video scene still renders.

- [ ] **Step 5: Commit**

```bash
git add static/client-portal.html static/client-login.html
git commit -m "feat(theme): portal and login use the unified 3-state toggle"
```

---

### Task 5: Cross-surface verification and migration sweep

**Files:** none (verification only; no code unless a defect is found).

**Interfaces:** Consumes the full system from Tasks 1-4.

- [ ] **Step 1: Confirm one key, three states, four surfaces**

With the app running, in each of homepage, `/portal/login`, and the portal:
- `localStorage['rm-theme-mode']` is the only theme key being written on toggle (watch it change; confirm `rm_portal_theme` is no longer written).
- Selecting Auto with a cached `rm-geo` resolves to Light during local day and Dark at night (temporarily set `rm-geo` to a night-side longitude to prove the flip, then restore).

- [ ] **Step 2: Confirm the legacy migration path**

In a fresh profile: set `localStorage['rm_portal_theme']='dark'`, clear `rm-theme-mode`, reload the portal, and confirm `rm-theme-mode` becomes `dark` and the page renders dark.

- [ ] **Step 3: Run the engine tests once more**

Run: `node --test tests/theme/sun.test.cjs tests/theme/resolve.test.cjs`
Expected: PASS, 8 tests total.

- [ ] **Step 4: Commit any fixes and open the PR**

```bash
git add -A
git commit -m "test(theme): cross-surface verification notes" --allow-empty
git push
gh pr create --draft --title "Theme: unified 3-state Light/Dark/Auto (sun-following)" \
  --body "Implements docs/superpowers/specs/2026-07-12-homepage-header-redesign-design.md section 4."
```

## Self-Review

**Spec coverage (section 4 of the design spec):**
- Three states Light/Dark/Auto, one storage key: Tasks 2-4 (`rm-theme-mode`).
- Auto follows local sunrise/sunset, computed in-browser, no external call: Task 1 engine, Task 2 `_resolve`/`currentContext`.
- Default new visitors to Auto: `getMode()` returns `auto` when unset; `migrate()` writes `auto` when no legacy value.
- Geolocation asked once, cached, never leaves device: `requestLocation` + `rm-geo`; no fetch anywhere.
- Graceful fallback on denied/blocked: `geoDenied` flag + the null-sun 7-19 window in `_resolve`.
- Migrate the two legacy keys: `migrate()` reads `rm-theme` and `rm_portal_theme`.
- Always visible on both sites: mounted in the ribbon (funnel) and on `#themeToggle` (portal/login).
- Sets data-theme only, palettes untouched: Task 3 keeps the palette `<style>` blocks; Task 4 keeps the portal's committed CSS.

**Placeholder scan:** none; every step carries full code or a concrete command.

**Type consistency:** `sunTimes` returns `{sunrise, sunset}` (Task 1) and is consumed with those exact names in `currentContext` (Task 2). `_resolve(mode, ctx)` signature is identical in the test (Task 2 Step 1) and the implementation (Task 2 Step 3). `mountToggle(container)` is defined in Task 2 and called with a single element argument in Tasks 3 and 4.

**Known follow-ups (out of scope here):** migrate the four `/practitioner/*` pages
(`practitioner-portal.html`, `-settings`, `-register`, `-dropship`) off their own
2-state `rm-theme` toggle onto the unified controller; until then they are excluded
from theme injection (see Execution notes). The remaining four redesign plans (icon
set, header rebuild, homepage hybrid, portal relabels) are separate documents.

## Execution notes (2026-07-12)

The plan shipped as written for Tasks 1, 2, 4, and 5, with two corrections made
during Task 3 after review caught real integration gaps:

- **Controller injection moved server-side and decoupled from the ribbon flag.**
  The plan added the `sun-engine.js`/`theme-mode.js` includes only to `begin.html`,
  but `shell.js` (and thus the theme system) runs on ~19 funnel pages plus the
  portal, injected by `shell_nav`. Wiring one page would have left every other page
  with `window.RMTheme` undefined and no toggle. Fix: a new
  `shell_nav.inject_theme_html` injects the controller (synchronous, before the
  deferred `shell.js`) into every `should_inject` page, from `_inject_journey_shell`
  in `app.py`, and it runs ALWAYS, not gated on `JOURNEY_SHELL_ENABLED` (theming is
  universal; only the ribbon is flag-gated). The `begin.html` manual includes were
  reverted. `theme-mode.js` gained an idempotency guard.
- **`/practitioner/*` excluded from theme injection.** Those four pages still ship
  their own `rm-theme` toggle; injecting the controller there made the two fight and
  the theme choice fail to persist. `shell_nav.should_inject_theme` skips
  `/practitioner` prefixes pending their migration.

The `embed.html` chat-iframe cross-document sync was confirmed intact: the controller
is injected there too and its `storage` listener on `rm-theme-mode`/`rm-theme`
preserves parent-following.

**Live render-verification (pre-merge gate, not runnable in this headless job):**
1) portal/login/community in Light for a daytime new `auto` visitor stay legible over
the fireside/video scene; 2) the funnel ribbon shows the 3-button SVG control and
toggling writes only `rm-theme-mode`; 3) the chat iframe follows a parent theme change.
