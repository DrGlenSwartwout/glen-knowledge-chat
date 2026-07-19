# Landing-page Fireside Invitation Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Put a small muted welcome video on the landing page whose tap plays Dr. Glen's invitation, unlocks browser audio so chat replies speak themselves, and offers a door to `/begin/fireside` — plus real fullscreen on the fireside stage.

**Architecture:** All logic lives in one new ES module, `static/begin/invitation.js`, which is pure enough to unit-test in Node against fake DOM objects. A thin, untested mount script fetches the fireside manifest and wires that module to real elements. The tap posts a same-origin `postMessage` into the existing `#begin-chat` iframe; `embed.html` listens and switches from `window.TTS.attach` to `window.TTS.attachAndSpeak`. Fullscreen is a self-contained block appended to the existing fireside IIFE.

**Tech Stack:** Vanilla ES modules (no build step, no bundler), Node's built-in `node:test` runner with zero dependencies, Flask test client for served-HTML wiring assertions.

## Global Constraints

- **No new server routes, and no changes to `/chat`, `/chat/tts`, `/transcribe`, or `/begin/fireside/agent`.** This work is frontend-only.
- **No new video or audio assets.** Every clip referenced already exists in `static/fireside/fireside-manifest.json` and serves 200 in production.
- **Graceful degradation is mandatory, following the Director precedent:** if the manifest fetch fails or yields no clip, the tile never appears and the page behaves exactly as it does today. No method may throw on an empty manifest.
- **The "🔊 Listen" button stays rendered in both locked and unlocked states.** Unlocking adds automatic playback; it never removes the manual control.
- **Never run the bare full pytest suite.** It sends real email. Run only the specific test files named in each task.
- **Run pytest through Doppler.** Under bare `pytest`, app-importing tests silently skip rather than fail, so a green bare run proves nothing. Use the exact commands given.
- **ES module style matches `static/fireside/director.js`:** named `export`, imported in tests by relative path from `tests/`, no transpilation.
- **Copy is fixed and exact.** The two CTAs read `Come sit by the fire` and `Ask here instead`. The hint reads `tap to hear`. Do not reword.

---

## File Structure

| File | Responsibility |
|---|---|
| `static/begin/invitation.js` | **Create.** All invitation logic: clip selection, muted start, tap-to-unmute, unlock postMessage, end-of-clip choices. Exported and unit-tested. |
| `static/begin/invitation-mount.js` | **Create.** Thin DOM wiring: fetch manifest, resolve elements, construct `Invitation`, bind listeners. Deliberately untested — it contains no branching logic. |
| `static/begin.html` | **Modify.** Tile markup, scoped styles, module script tag. |
| `static/embed.html` | **Modify.** Unlock listener + auto-speak branch. |
| `static/begin-fireside.html` | **Modify.** Fullscreen button + handler. |
| `tests/begin_js/invitation.test.mjs` | **Create.** Node unit tests for `invitation.js`. |
| `tests/test_begin_invitation_wiring.py` | **Create.** Flask-client assertions that the served HTML actually carries the wiring. |
| `docs/superpowers/verification/2026-07-19-landing-fireside-invitation.md` | **Create.** Manual verification record for the browser behaviors machines cannot assert. |

---

### Task 1: Invitation module

**Files:**
- Create: `static/begin/invitation.js`
- Test: `tests/begin_js/invitation.test.mjs`

**Interfaces:**
- Consumes: nothing from earlier tasks.
- Produces:
  - `UNLOCK_MSG: string` — the literal `'begin:audio-unlocked'`, used as `{type: UNLOCK_MSG}` by Task 3's listener.
  - `pickWelcomeClip(manifest: object|null, rand?: () => number) => string|null`
  - `class Invitation` with constructor `Invitation({video, root, choices, hint, frame, origin, clip, restingClip})` and methods `start() => boolean`, `tap() => boolean`, `notifyUnlock() => boolean`, `onEnded() => void`, `dismiss() => void`.

- [ ] **Step 1: Write the failing test**

Create `tests/begin_js/invitation.test.mjs`:

```js
import { test } from 'node:test';
import assert from 'node:assert/strict';
import { pickWelcomeClip, Invitation, UNLOCK_MSG } from '../../static/begin/invitation.js';

const M = {
  intro_welcomes: ['/a.mp4', '/b.mp4', '/c.mp4'],
  intro_welcome: '/w.mp4',
  intro_video: '/v.mp4',
};

function fakeVideo() {
  return {
    src: '', muted: true, loop: false, currentTime: 0, onended: null, _played: 0,
    play() { this._played++; return { catch() {} }; },
  };
}

function fakeEl(initiallyHidden = true) {
  const cls = new Set(initiallyHidden ? ['hidden'] : []);
  return {
    classList: {
      add: (c) => cls.add(c),
      remove: (c) => cls.delete(c),
      contains: (c) => cls.has(c),
    },
  };
}

function fakeFrame() {
  const sent = [];
  return { sent, contentWindow: { postMessage: (msg, origin) => sent.push({ msg, origin }) } };
}

function build(over = {}) {
  const video = fakeVideo();
  const root = fakeEl();
  const choices = fakeEl();
  const hint = fakeEl(false);
  const frame = fakeFrame();
  const inv = new Invitation({
    video, root, choices, hint, frame,
    origin: 'https://illtowell.com',
    clip: '/a.mp4',
    restingClip: '/rest-1.mp4',
    ...over,
  });
  return { inv, video, root, choices, hint, frame };
}

// ── pickWelcomeClip ─────────────────────────────────────────────────────────
test('picks from intro_welcomes', () => {
  assert.equal(pickWelcomeClip(M, () => 0), '/a.mp4');
  assert.equal(pickWelcomeClip(M, () => 0.99), '/c.mp4');
});

test('falls back to intro_welcome when the list is empty', () => {
  assert.equal(pickWelcomeClip({ intro_welcomes: [], intro_welcome: '/w.mp4' }), '/w.mp4');
});

test('falls back to intro_video when neither list nor intro_welcome', () => {
  assert.equal(pickWelcomeClip({ intro_video: '/v.mp4' }), '/v.mp4');
});

test('returns null on an empty or missing manifest', () => {
  assert.equal(pickWelcomeClip({}), null);
  assert.equal(pickWelcomeClip(null), null);
});

// ── start ───────────────────────────────────────────────────────────────────
test('start plays muted and looping, and reveals the tile', () => {
  const { inv, video, root } = build();
  assert.equal(inv.start(), true);
  assert.equal(video.src, '/a.mp4');
  assert.equal(video.muted, true);
  assert.equal(video.loop, true);
  assert.equal(video._played, 1);
  assert.equal(root.classList.contains('hidden'), false);
});

test('start is a no-op with no clip and leaves the tile hidden', () => {
  const { inv, video, root } = build({ clip: null });
  assert.equal(inv.start(), false);
  assert.equal(video._played, 0);
  assert.equal(root.classList.contains('hidden'), true);
});

// ── tap ─────────────────────────────────────────────────────────────────────
test('tap unmutes, restarts, stops looping, and hides the hint', () => {
  const { inv, video, hint } = build();
  inv.start();
  assert.equal(inv.tap(), true);
  assert.equal(video.muted, false);
  assert.equal(video.loop, false);
  assert.equal(video.currentTime, 0);
  assert.equal(hint.classList.contains('hidden'), true);
});

test('tap posts the unlock message once, to the given origin', () => {
  const { inv, frame } = build();
  inv.start();
  inv.tap();
  assert.equal(frame.sent.length, 1);
  assert.deepEqual(frame.sent[0].msg, { type: UNLOCK_MSG });
  assert.equal(frame.sent[0].origin, 'https://illtowell.com');
});

test('tapping twice still posts only one unlock message', () => {
  const { inv, frame } = build();
  inv.start();
  inv.tap();
  inv.tap();
  assert.equal(frame.sent.length, 1);
});

test('tap with no frame does not throw and still marks unlocked', () => {
  const { inv } = build({ frame: null });
  inv.start();
  assert.doesNotThrow(() => inv.tap());
  assert.equal(inv.unlocked, true);
});

// ── end of clip ─────────────────────────────────────────────────────────────
test('onEnded reveals the choices and returns to a muted resting loop', () => {
  const { inv, video, choices } = build();
  inv.start();
  inv.tap();
  video.onended();
  assert.equal(choices.classList.contains('hidden'), false);
  assert.equal(video.src, '/rest-1.mp4');
  assert.equal(video.loop, true);
  assert.equal(video.muted, true);
});

test('onEnded with no resting clip still reveals the choices', () => {
  const { inv, choices } = build({ restingClip: null });
  inv.start();
  inv.tap();
  assert.doesNotThrow(() => inv.onEnded());
  assert.equal(choices.classList.contains('hidden'), false);
});

// ── dismiss ─────────────────────────────────────────────────────────────────
test('dismiss hides the whole tile', () => {
  const { inv, root } = build();
  inv.start();
  inv.dismiss();
  assert.equal(root.classList.contains('hidden'), true);
});

// ── graceful degradation ────────────────────────────────────────────────────
test('every method is safe with an entirely empty construction', () => {
  const inv = new Invitation({});
  assert.doesNotThrow(() => { inv.start(); inv.tap(); inv.notifyUnlock(); inv.onEnded(); inv.dismiss(); });
});
```

- [ ] **Step 2: Run the test to verify it fails**

```bash
cd /tmp/wt-deploy-chat-b9535446 && node --test tests/begin_js/*.test.mjs
```

Expected: FAIL — `Cannot find module .../static/begin/invitation.js`.

- [ ] **Step 3: Write the implementation**

Create `static/begin/invitation.js`:

```js
/* invitation.js — the fireside welcome tile on the landing page.
 *
 * A small muted clip autoplays beside the chat panel. Tapping it unmutes and
 * plays Glendalf's invitation; that tap is also the browser gesture that
 * unlocks audio, so it is forwarded to the chat iframe, which from then on
 * speaks its replies aloud instead of waiting for a Listen click.
 *
 * Every method is a safe no-op when its dependencies are missing, mirroring
 * the Director's degradation contract: a failed manifest fetch must leave the
 * page exactly as it was.
 */

export const UNLOCK_MSG = 'begin:audio-unlocked';

export function pickWelcomeClip(m, rand = Math.random) {
  if (!m) return null;
  const list = Array.isArray(m.intro_welcomes) && m.intro_welcomes.length ? m.intro_welcomes : [];
  if (list.length) return list[Math.floor(rand() * list.length)] || list[0];
  return m.intro_welcome || m.intro_video || null;
}

export class Invitation {
  constructor(opts = {}) {
    this.video       = opts.video || null;
    this.root        = opts.root || null;
    this.choices     = opts.choices || null;
    this.hint        = opts.hint || null;
    this.frame       = opts.frame || null;
    this.origin      = opts.origin || '*';
    this.clip        = opts.clip || null;
    this.restingClip = opts.restingClip || null;
    this.unlocked    = false;
  }

  start() {
    if (!this.clip || !this.video) return false;
    this.video.src = this.clip;
    this.video.muted = true;
    this.video.loop = true;
    this.video.play();
    if (this.root) this.root.classList.remove('hidden');
    return true;
  }

  tap() {
    if (!this.clip || !this.video) return false;
    this.video.muted = false;
    this.video.loop = false;
    this.video.currentTime = 0;
    this.video.onended = () => this.onEnded();
    this.video.play();
    if (this.hint) this.hint.classList.add('hidden');
    this.notifyUnlock();
    return true;
  }

  notifyUnlock() {
    if (this.unlocked) return false;
    this.unlocked = true;
    if (this.frame && this.frame.contentWindow) {
      this.frame.contentWindow.postMessage({ type: UNLOCK_MSG }, this.origin);
    }
    return true;
  }

  onEnded() {
    if (this.choices) this.choices.classList.remove('hidden');
    if (this.restingClip && this.video) {
      this.video.src = this.restingClip;
      this.video.muted = true;
      this.video.loop = true;
      this.video.currentTime = 0;
      this.video.play();
    }
  }

  dismiss() {
    if (this.root) this.root.classList.add('hidden');
  }
}
```

- [ ] **Step 4: Run the test to verify it passes**

```bash
cd /tmp/wt-deploy-chat-b9535446 && node --test tests/begin_js/*.test.mjs
```

Expected: PASS, 15 tests.

- [ ] **Step 5: Commit**

```bash
cd /tmp/wt-deploy-chat-b9535446
git add static/begin/invitation.js tests/begin_js/invitation.test.mjs
git commit -m "feat: fireside invitation module with unlock postMessage"
```

---

### Task 2: Mount the tile on the landing page

**Files:**
- Create: `static/begin/invitation-mount.js`
- Modify: `static/begin.html` (inside `<section class="shell ask reveal">`, around line 832)
- Test: `tests/test_begin_invitation_wiring.py`

**Interfaces:**
- Consumes: `pickWelcomeClip`, `Invitation` from `static/begin/invitation.js` (Task 1).
- Produces: DOM contract relied on by nothing else, but Task 3's listener depends on the `postMessage` this mount causes. Element ids: `#fireside-invite`, `#fs-invite-video`, `#fs-invite-hint`, `#fs-invite-choices`, `#fs-invite-tap`, `.fs-invite-stay`.

- [ ] **Step 1: Write the failing test**

Create `tests/test_begin_invitation_wiring.py`:

```python
"""The served landing page must actually carry the invitation wiring.

These are deliberately shallow assertions on served HTML: the browser
behaviour (autoplay policy, audio unlock, fullscreen) cannot be asserted
headlessly and is covered by the manual verification record instead.
"""
import importlib


def _reload_app(monkeypatch, tmp_path):
    monkeypatch.setenv("DATA_DIR", str(tmp_path))
    monkeypatch.setenv("FIRESIDE_ENABLED", "true")
    import app as appmod
    importlib.reload(appmod)
    return appmod


def test_landing_page_carries_invitation_tile(monkeypatch, tmp_path):
    appmod = _reload_app(monkeypatch, tmp_path)
    body = appmod.app.test_client().get("/begin").get_data(as_text=True)
    assert 'id="fireside-invite"' in body
    assert 'id="fs-invite-video"' in body
    assert 'id="fs-invite-choices"' in body
    assert "/static/begin/invitation-mount.js" in body


def test_invitation_tile_starts_hidden(monkeypatch, tmp_path):
    """It must not occupy layout until a clip actually resolves."""
    appmod = _reload_app(monkeypatch, tmp_path)
    body = appmod.app.test_client().get("/begin").get_data(as_text=True)
    idx = body.index('id="fireside-invite"')
    tag = body[idx - 200 : idx + 200]
    assert "hidden" in tag


def test_invitation_ctas_use_exact_copy(monkeypatch, tmp_path):
    appmod = _reload_app(monkeypatch, tmp_path)
    body = appmod.app.test_client().get("/begin").get_data(as_text=True)
    assert "Come sit by the fire" in body
    assert "Ask here instead" in body
    assert 'href="/begin/fireside"' in body


def test_invitation_module_is_served(monkeypatch, tmp_path):
    appmod = _reload_app(monkeypatch, tmp_path)
    c = appmod.app.test_client()
    assert c.get("/static/begin/invitation.js").status_code == 200
    assert c.get("/static/begin/invitation-mount.js").status_code == 200
```

- [ ] **Step 2: Run the test to verify it fails**

```bash
cd /tmp/wt-deploy-chat-b9535446 && doppler run -- python3 -m pytest tests/test_begin_invitation_wiring.py -v
```

Expected: FAIL — `assert 'id="fireside-invite"' in body`.

- [ ] **Step 3a: Add the tile markup to `static/begin.html`**

Find this exact block (around line 831):

```html
    <!-- ASK A QUESTION — multi-turn funnel-mode chat (piece 3) -->
    <section class="shell ask reveal">
      <iframe id="begin-chat" src="/embed?mode=funnel" title="Ask Dr. Glen" loading="eager"
```

Insert the tile between the `<section>` and the `<iframe>` so it reads as part of the same surface:

```html
    <!-- ASK A QUESTION — multi-turn funnel-mode chat (piece 3) -->
    <section class="shell ask reveal">
      <!-- Fireside invitation — muted autoplay; the tap is also the audio-unlock gesture -->
      <div id="fireside-invite" class="fs-invite hidden">
        <button id="fs-invite-tap" class="fs-invite-tap" type="button"
                aria-label="Hear Dr. Glen's invitation">
          <video id="fs-invite-video" class="fs-invite-video"
                 playsinline muted loop preload="metadata"
                 poster="/static/fireside/video/intro-poster.jpg"></video>
          <span id="fs-invite-hint" class="fs-invite-hint">tap to hear</span>
        </button>
        <div id="fs-invite-choices" class="fs-invite-choices hidden">
          <a class="fs-invite-go" href="/begin/fireside">Come sit by the fire</a>
          <button class="fs-invite-stay" type="button">Ask here instead</button>
        </div>
      </div>
      <iframe id="begin-chat" src="/embed?mode=funnel" title="Ask Dr. Glen" loading="eager"
```

- [ ] **Step 3b: Add the scoped styles**

Immediately before the closing `</head>` of `static/begin.html`, add:

```html
<style>
  /* Fireside invitation tile — deliberately small so it never pushes the chat
     input below the fold on a phone. 160px, not a hero video. */
  .fs-invite { display:flex; align-items:center; gap:14px; margin:0 0 14px;
    flex-wrap:wrap; }
  .fs-invite.hidden { display:none !important; }
  .fs-invite-tap { position:relative; width:160px; height:160px; padding:0;
    border:1px solid var(--border); border-radius:var(--radius);
    background:var(--surface); overflow:hidden; cursor:pointer; flex:0 0 auto; }
  .fs-invite-video { width:100%; height:100%; object-fit:cover; display:block; }
  .fs-invite-hint { position:absolute; left:0; right:0; bottom:0;
    padding:6px 8px; font-size:12px; letter-spacing:.02em; color:#f3e6cf;
    background:linear-gradient(transparent, rgba(20,12,6,.82)); }
  .fs-invite-hint.hidden { display:none !important; }
  .fs-invite-choices { display:flex; flex-direction:column; gap:8px;
    align-items:flex-start; }
  .fs-invite-choices.hidden { display:none !important; }
  .fs-invite-go { display:inline-block; padding:10px 18px; border-radius:24px;
    background:linear-gradient(#caa86a,#8a6a38); color:#1a0f06;
    text-decoration:none; font-size:15px; }
  .fs-invite-stay { background:none; border:none; color:var(--muted);
    font:inherit; font-size:14px; text-decoration:underline; cursor:pointer;
    padding:0; }
  @media (prefers-reduced-motion: reduce) { .fs-invite-video { display:none; } }
</style>
```

- [ ] **Step 3c: Create the mount script**

Create `static/begin/invitation-mount.js`:

```js
/* invitation-mount.js — wires invitation.js to the real landing-page DOM.
 * Intentionally thin and untested: all branching logic lives in invitation.js.
 */
import { pickWelcomeClip, Invitation } from './invitation.js';

(function () {
  var root = document.getElementById('fireside-invite');
  if (!root) return;

  fetch('/static/fireside/fireside-manifest.json')
    .then(function (r) { return r.ok ? r.json() : null; })
    .then(function (m) {
      var clip = pickWelcomeClip(m);
      if (!clip) return;                       // no clip: tile stays hidden

      var resting = (m && Array.isArray(m.resting_loops) && m.resting_loops.length)
        ? m.resting_loops[0] : null;

      var inv = new Invitation({
        video:       document.getElementById('fs-invite-video'),
        root:        root,
        choices:     document.getElementById('fs-invite-choices'),
        hint:        document.getElementById('fs-invite-hint'),
        frame:       document.getElementById('begin-chat'),
        origin:      window.location.origin,
        clip:        clip,
        restingClip: resting,
      });

      document.getElementById('fs-invite-tap')
        .addEventListener('click', function () { inv.tap(); });

      var stay = root.querySelector('.fs-invite-stay');
      if (stay) stay.addEventListener('click', function () { inv.dismiss(); });

      inv.start();
    })
    .catch(function () { /* manifest unavailable: leave the page as it was */ });
})();
```

- [ ] **Step 3d: Load the module**

Immediately before the closing `</body>` of `static/begin.html`, add:

```html
<script type="module" src="/static/begin/invitation-mount.js"></script>
```

- [ ] **Step 4: Run the test to verify it passes**

```bash
cd /tmp/wt-deploy-chat-b9535446 && doppler run -- python3 -m pytest tests/test_begin_invitation_wiring.py -v
```

Expected: PASS, 4 tests.

- [ ] **Step 5: Commit**

```bash
cd /tmp/wt-deploy-chat-b9535446
git add static/begin.html static/begin/invitation-mount.js tests/test_begin_invitation_wiring.py
git commit -m "feat: mount fireside invitation tile on the landing page"
```

---

### Task 3: Spoken replies after unlock

**Files:**
- Modify: `static/embed.html:1167-1171` (the `window.TTS.attach` call) and the top-level script scope
- Test: `tests/test_begin_invitation_wiring.py` (append)

**Interfaces:**
- Consumes: the `{type: 'begin:audio-unlocked'}` message posted by `Invitation.notifyUnlock()` (Task 1).
- Produces: nothing consumed by later tasks.

- [ ] **Step 1: Write the failing test**

Append to `tests/test_begin_invitation_wiring.py`:

```python
def test_embed_listens_for_the_unlock_message(monkeypatch, tmp_path):
    appmod = _reload_app(monkeypatch, tmp_path)
    body = appmod.app.test_client().get("/embed?mode=funnel").get_data(as_text=True)
    assert "begin:audio-unlocked" in body
    assert "__audioUnlocked" in body


def test_embed_can_auto_speak_and_keeps_the_listen_button(monkeypatch, tmp_path):
    appmod = _reload_app(monkeypatch, tmp_path)
    body = appmod.app.test_client().get("/embed?mode=funnel").get_data(as_text=True)
    # both paths must survive: automatic playback AND the manual control
    assert "TTS.attachAndSpeak" in body
    assert "TTS.attach(" in body


def test_embed_unlock_listener_checks_origin(monkeypatch, tmp_path):
    """A cross-origin frame must not be able to force audio playback."""
    appmod = _reload_app(monkeypatch, tmp_path)
    body = appmod.app.test_client().get("/embed?mode=funnel").get_data(as_text=True)
    idx = body.index("begin:audio-unlocked")
    assert "location.origin" in body[idx - 400 : idx + 400]
```

- [ ] **Step 2: Run the test to verify it fails**

```bash
cd /tmp/wt-deploy-chat-b9535446 && doppler run -- python3 -m pytest tests/test_begin_invitation_wiring.py -v
```

Expected: FAIL on `test_embed_listens_for_the_unlock_message` — `assert "begin:audio-unlocked" in body`.

- [ ] **Step 3a: Add the listener**

In `static/embed.html`, inside the same top-level script scope that defines `appendMessage`, add near the other top-level listeners:

```js
  // The landing page's invitation tile forwards its tap here. That tap is the
  // browser gesture that permits audio, so from this point on replies speak
  // themselves instead of waiting for a Listen click.
  window.__audioUnlocked = false;
  window.addEventListener('message', function (e) {
    if (e.origin !== window.location.origin) return;
    var d = e.data;
    if (d && d.type === 'begin:audio-unlocked') window.__audioUnlocked = true;
  });
```

- [ ] **Step 3b: Branch the TTS call**

Replace this exact block at `static/embed.html:1167-1171`:

```js
    if (role === 'assistant' && window.TTS) {
      const bar  = div.querySelector('.msg-actions');
      const body = div.querySelector('.message-body');
      if (bar && body) window.TTS.attach(bar, body.innerText);
    }
```

with:

```js
    if (role === 'assistant' && window.TTS) {
      const bar  = div.querySelector('.msg-actions');
      const body = div.querySelector('.message-body');
      if (bar && body) {
        // attachAndSpeak renders the same Listen button, then speaks once.
        if (window.__audioUnlocked) window.TTS.attachAndSpeak(bar, body.innerText);
        else                        window.TTS.attach(bar, body.innerText);
      }
    }
```

- [ ] **Step 4: Run the test to verify it passes**

```bash
cd /tmp/wt-deploy-chat-b9535446 && doppler run -- python3 -m pytest tests/test_begin_invitation_wiring.py -v
```

Expected: PASS, 7 tests.

- [ ] **Step 5: Commit**

```bash
cd /tmp/wt-deploy-chat-b9535446
git add static/embed.html tests/test_begin_invitation_wiring.py
git commit -m "feat: speak chat replies automatically once audio is unlocked"
```

---

### Task 4: Fullscreen on the fireside stage

**Files:**
- Modify: `static/begin-fireside.html` (button near `#muteBtn` around line 71; handler in the main IIFE)
- Test: `tests/test_begin_invitation_wiring.py` (append)

**Interfaces:**
- Consumes: nothing.
- Produces: nothing.

Note: fullscreen targets `#fireside` (the whole `<main>`), not `#stage-wrap`, so the subtitle, composer and mic go fullscreen with the video rather than being cropped away.

- [ ] **Step 1: Write the failing test**

Append to `tests/test_begin_invitation_wiring.py`:

```python
def test_fireside_has_a_fullscreen_button(monkeypatch, tmp_path):
    appmod = _reload_app(monkeypatch, tmp_path)
    body = appmod.app.test_client().get("/begin/fireside").get_data(as_text=True)
    assert 'id="fsBtn"' in body
    assert "requestFullscreen" in body


def test_fireside_fullscreen_is_feature_detected(monkeypatch, tmp_path):
    """iOS Safari cannot fullscreen arbitrary elements; the button must hide
    there rather than sit on the page doing nothing."""
    appmod = _reload_app(monkeypatch, tmp_path)
    body = appmod.app.test_client().get("/begin/fireside").get_data(as_text=True)
    assert "webkitRequestFullscreen" in body
    idx = body.index("webkitRequestFullscreen")
    assert "hidden" in body[idx : idx + 600]
```

- [ ] **Step 2: Run the test to verify it fails**

```bash
cd /tmp/wt-deploy-chat-b9535446 && doppler run -- python3 -m pytest tests/test_begin_invitation_wiring.py -v
```

Expected: FAIL — `assert 'id="fsBtn"' in body`.

- [ ] **Step 3a: Add the button**

In `static/begin-fireside.html`, immediately after the existing `#muteBtn` element inside `#stage-wrap`, add:

```html
    <button id="fsBtn" aria-label="Enter fullscreen"
      style="position:absolute;right:56px;bottom:12px;background:rgba(20,12,6,.72);border:1px solid #6b5436;color:var(--warm);border-radius:50%;width:36px;height:36px;font-size:15px;cursor:pointer;z-index:2;">⛶</button>
```

- [ ] **Step 3b: Add the handler**

Inside the main IIFE of `static/begin-fireside.html`, after the other `getElementById` lookups (near line 137), add:

```js
  // ── fullscreen ────────────────────────────────────────────────────────────
  // The page already fakes immersion with position:fixed + 100dvh + a
  // visualViewport handler. This makes it real where the browser allows it.
  // iOS Safari only fullscreens <video> elements, never arbitrary containers,
  // so there the button hides and the CSS treatment remains the fallback.
  var fsBtn = document.getElementById('fsBtn');
  var fsTarget = document.getElementById('fireside');
  var fsEnter = fsTarget && (fsTarget.requestFullscreen || fsTarget.webkitRequestFullscreen);
  var fsExit  = document.exitFullscreen || document.webkitExitFullscreen;

  function fsCurrent() {
    return document.fullscreenElement || document.webkitFullscreenElement || null;
  }

  function fsSync() {
    if (!fsBtn) return;
    var on = !!fsCurrent();
    fsBtn.innerHTML = on ? '✕' : '⛶';
    fsBtn.setAttribute('aria-label', on ? 'Exit fullscreen' : 'Enter fullscreen');
  }

  if (fsBtn) {
    if (!fsEnter) {
      fsBtn.classList.add('hidden');
    } else {
      fsBtn.addEventListener('click', function () {
        try {
          if (fsCurrent()) { if (fsExit) fsExit.call(document); }
          else             { fsEnter.call(fsTarget); }
        } catch (e) { /* denied by the browser: stay windowed */ }
      });
      document.addEventListener('fullscreenchange', fsSync);
      document.addEventListener('webkitfullscreenchange', fsSync);
      fsSync();
    }
  }
```

- [ ] **Step 4: Run the test to verify it passes**

```bash
cd /tmp/wt-deploy-chat-b9535446 && doppler run -- python3 -m pytest tests/test_begin_invitation_wiring.py -v
```

Expected: PASS, 9 tests.

- [ ] **Step 5: Commit**

```bash
cd /tmp/wt-deploy-chat-b9535446
git add static/begin-fireside.html tests/test_begin_invitation_wiring.py
git commit -m "feat: real fullscreen on the fireside stage, hidden on iOS"
```

---

### Task 5: Verification record

**Files:**
- Create: `docs/superpowers/verification/2026-07-19-landing-fireside-invitation.md`

**Interfaces:**
- Consumes: the working implementation from Tasks 1-4.
- Produces: nothing.

This task exists because the three behaviors that matter most — muted autoplay, the audio unlock, and fullscreen — are all governed by per-browser policy that headless Chrome does not reproduce. The existing `tests/fireside_render_verify.md` sets the precedent for recording this rather than pretending a machine asserted it.

- [ ] **Step 1: Run the full automated set**

```bash
cd /tmp/wt-deploy-chat-b9535446
node --test tests/begin_js/*.test.mjs
doppler run -- python3 -m pytest tests/test_begin_invitation_wiring.py tests/test_fireside_routes.py tests/test_begin_routes.py tests/test_chat_tts.py -v
```

Expected: all green. Record the counts.

- [ ] **Step 2: Headless layout check**

```bash
cd /private/tmp/claude-501/-Users-remedymatch-AI-Training/b9535446-3393-4f7a-b288-6d490e769750/scratchpad
"/Applications/Google Chrome.app/Contents/MacOS/Google Chrome" --headless --disable-gpu \
  --screenshot=invite-mobile.png --window-size=390,844 --virtual-time-budget=9000 \
  "http://localhost:5000/begin"
```

Confirm from the screenshot that the tile renders at roughly 160px and the chat input is still above the fold. Note that headless Chrome will not decode the video, so a poster frame is the expected appearance.

- [ ] **Step 3: Manual browser passes**

Perform each and record pass/fail with the browser version:

1. **Desktop Chrome:** tile autoplays muted; tap unmutes and plays the invitation; both CTAs appear when it ends; the next chat reply speaks with no click; the "🔊 Listen" button still works and still toggles off.
2. **iOS Safari:** tile autoplays muted; tap unlocks audio; the fullscreen button is absent, not present-and-broken; the `100dvh` treatment still pins the scene when the keyboard opens.
3. **Desktop Chrome, fireside:** the fullscreen button enters and exits, the icon swaps both ways, and pressing Escape also swaps the icon back.
4. **Never-tapped control:** load `/begin`, ignore the tile entirely, ask a question by typing. Confirm the reply does *not* speak and the page behaves exactly as before this change.
5. **Degradation:** block `/static/fireside/fireside-manifest.json` in devtools, reload. Confirm the tile never appears and the chat is unaffected.

- [ ] **Step 4: Write the record and commit**

Write the results into `docs/superpowers/verification/2026-07-19-landing-fireside-invitation.md`, following the structure of `tests/fireside_render_verify.md`: what is machine-asserted, what is manually verified and why it cannot be automated, and the observed results per browser.

```bash
cd /tmp/wt-deploy-chat-b9535446
git add docs/superpowers/verification/2026-07-19-landing-fireside-invitation.md
git commit -m "docs: verification record for the landing fireside invitation"
```

---

## Self-Review

**Spec coverage:**

| Spec section | Task |
|---|---|
| 1. Welcome tile (poster, muted autoplay, 160px, tap, two CTAs) | 1, 2 |
| 2. Audio unlock (session flag, postMessage, attachAndSpeak, Listen retained) | 1, 3 |
| 3. Handoff (plain navigation, no context carried) | 2 (`href="/begin/fireside"`) |
| 4. Fullscreen (requestFullscreen, webkit prefix, fullscreenchange, iOS hidden) | 4 |
| Testing 1-2 (headless renders) | 5 steps 2 |
| Testing 3-5 (manual passes) | 5 step 3 |
| Risk: attention competition | 2 step 3b (160px cap, reduced-motion) |
| Risk: autoplay drift | 2 step 3a (`poster` attribute) + 5 step 3.5 |
| Risk: unexpected speech | 3 step 3b (Listen button retained) + 5 step 3.4 |

No gaps.

**Placeholder scan:** none. Every code step carries complete code; every command is exact.

**Type consistency:** `UNLOCK_MSG` is `'begin:audio-unlocked'` in Task 1 and the same literal appears in Task 3's listener and its test. `Invitation` constructor keys (`video, root, choices, hint, frame, origin, clip, restingClip`) match exactly between Task 1's class, Task 1's test helper, and Task 2's mount. Element ids `#fireside-invite`, `#fs-invite-video`, `#fs-invite-hint`, `#fs-invite-choices`, `#fs-invite-tap`, `.fs-invite-stay` are identical across Task 2's markup, styles, mount, and tests. `#fsBtn` and `#fireside` match between Task 4's markup, handler, and tests.
