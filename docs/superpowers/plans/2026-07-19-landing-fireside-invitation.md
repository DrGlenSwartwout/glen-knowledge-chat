# Landing-page Fireside Invitation Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a speaker button to the landing page's existing hero avatar whose tap plays Dr. Glen's spoken invitation and unlocks browser audio so chat replies speak themselves, plus real fullscreen on the fireside stage.

**Architecture:** All logic lives in one ES module, `static/begin/invitation.js`, pure enough to unit-test in Node against fake DOM objects. A thin, untested mount script fetches the fireside manifest and wires it to a speaker button added inside the page's existing hero-avatar anchor. The tap plays a voice-over mp3 (the avatar's own video has no audio track) and posts a same-origin `postMessage` into the `#begin-chat` iframe; `embed.html` listens and switches from `window.TTS.attach` to `window.TTS.attachAndSpeak`. Fullscreen is a self-contained block appended to the existing fireside IIFE.

**Tech Stack:** Vanilla ES modules (no build step, no bundler), Node's built-in `node:test` runner with zero dependencies, Flask test client for served-HTML wiring assertions.

## Global Constraints

- **No new server routes, and no changes to `/chat`, `/chat/tts`, `/transcribe`, or `/begin/fireside/agent`.** This work is frontend-only.
- **No new video or audio assets.** Every clip referenced already exists in `static/fireside/fireside-manifest.json` and serves 200 in production.
- **Graceful degradation is mandatory, following the Director precedent:** if the manifest fetch fails or yields no audio source, the speaker button never appears and the page behaves exactly as it does today. No method may throw on an empty manifest.
- **The "🔊 Listen" button stays rendered in both locked and unlocked states.** Unlocking adds automatic playback; it never removes the manual control.
- **Never run the bare full pytest suite.** It sends real email. Run only the specific test files named in each task.
- **Run pytest through Doppler.** Under bare `pytest`, app-importing tests silently skip rather than fail, so a green bare run proves nothing. Use the exact commands given.
- **ES module style matches `static/fireside/director.js`:** named `export`, imported in tests by relative path from `tests/`, no transpilation.
- **Do not add a second fireside entry.** Commit `ec233b56` consolidated the landing page to one door, the hero avatar at `static/begin.html:761`. The speaker rides on that avatar; the page must still contain exactly one `href="/begin/fireside"`.
- **The speaker must not navigate.** It is a sibling of the anchor inside `.avatar-wrap`, not a descendant, so the click cannot reach the anchor's handler; `stopPropagation()` is unnecessary. `preventDefault()` is kept only as a guard against future re-nesting.
- **Run pytest as `doppler run --config dev`.** Never `prd` — that config holds live payment credentials.

---

## File Structure

| File | Responsibility |
|---|---|
| `static/begin/invitation.js` | **Create.** All invitation logic: voice-over selection, play/stop, unlock postMessage, button labelling. Exported and unit-tested. |
| `static/begin/invitation-mount.js` | **Create.** Thin DOM wiring: fetch manifest, resolve elements, construct `Invitation`, bind listeners. Deliberately untested — it contains no branching logic. |
| `static/begin.html` | **Modify.** Speaker button inside the existing hero-avatar anchor, its styles, the module script tag, and deletion of the orphaned `#fireside-invite` CSS left by ec233b56. |
| `static/embed.html` | **Modify.** Unlock listener + auto-speak branch. |
| `static/begin-fireside.html` | **Modify.** Fullscreen button + handler. |
| `tests/begin_js/invitation.test.mjs` | **Create.** Node unit tests for `invitation.js`. |
| `tests/test_begin_invitation_wiring.py` | **Create.** Flask-client assertions that the served HTML actually carries the wiring. |
| `docs/superpowers/verification/2026-07-19-landing-fireside-invitation.md` | **Create.** Manual verification record for the browser behaviors machines cannot assert. |

---

### Task 1: Invitation module (audio) and speaker button

**Files:**
- Rewrite: `static/begin/invitation.js` (exists from a superseded design — replace its contents)
- Rewrite: `tests/begin_js/invitation.test.mjs`
- Create: `static/begin/invitation-mount.js`
- Modify: `static/begin.html` — speaker button inside the existing `<a class="avatar">` (around line 761), styles near the existing `.avatar` rules (around line 586), module script before `</body>`, and deletion of the orphaned `#fireside-invite` / `.fi-*` / `fiGlow` CSS block
- Test: `tests/test_begin_invitation_wiring.py`

**Interfaces:**
- Consumes: nothing from earlier tasks.
- Produces:
  - `UNLOCK_MSG: string` — the literal `'begin:audio-unlocked'`, consumed by Task 3's listener in `embed.html`.
  - `pickInvitationAudio(manifest: object|null, rand?: () => number) => string|null`
  - `class Invitation` with constructor `Invitation({audio, button, frame, origin, src})` and methods `play() => boolean`, `stop() => void`, `toggle() => boolean`, `notifyUnlock() => boolean`, `mount() => boolean`.

**Why this replaces the previous Task 1 and Task 2:** an earlier design added a standalone
welcome tile. That was wrong — `static/begin.html:761` already carries a hero avatar (a
muted autoplaying `/static/media/glendalf-welcome.mp4` linking to `/begin/fireside`), and
commit `ec233b56` deliberately consolidated the page to that single fireside door. The
previous Task 2 commit has been reverted. This task hangs the invitation off the existing
avatar instead. `glendalf-welcome.mp4` has no audio track, so the voice-over is a separate
mp3.

- [ ] **Step 1: Write the failing test**

Create `tests/begin_js/invitation.test.mjs` (replacing its current contents entirely):

```js
import { test } from 'node:test';
import assert from 'node:assert/strict';
import { pickInvitationAudio, Invitation, UNLOCK_MSG } from '../../static/begin/invitation.js';

const M = {
  intro_welcome_audios: ['/a.mp3', '/b.mp3', '/c.mp3'],
  intro_welcome_audio: '/w.mp3',
};

function fakeAudio() {
  return {
    src: '', currentTime: 0, paused: true, onended: null, _played: 0, _paused: 0,
    play() { this._played++; this.paused = false; return { catch() {} }; },
    pause() { this._paused++; this.paused = true; },
  };
}

function fakeBtn() {
  const cls = new Set(['hidden']);
  return {
    innerHTML: '', _label: '',
    classList: { add: (c) => cls.add(c), remove: (c) => cls.delete(c), contains: (c) => cls.has(c) },
    setAttribute(k, v) { if (k === 'aria-label') this._label = v; },
  };
}

function fakeFrame() {
  const sent = [];
  return { sent, contentWindow: { postMessage: (msg, origin) => sent.push({ msg, origin }) } };
}

function build(over = {}) {
  const audio = fakeAudio();
  const button = fakeBtn();
  const frame = fakeFrame();
  const inv = new Invitation({
    audio, button, frame,
    origin: 'https://illtowell.com',
    src: '/a.mp3',
    ...over,
  });
  return { inv, audio, button, frame };
}

// ── pickInvitationAudio ─────────────────────────────────────────────────────
test('picks from intro_welcome_audios', () => {
  assert.equal(pickInvitationAudio(M, () => 0), '/a.mp3');
  assert.equal(pickInvitationAudio(M, () => 0.99), '/c.mp3');
});

test('falls back to intro_welcome_audio when the list is empty', () => {
  assert.equal(pickInvitationAudio({ intro_welcome_audios: [], intro_welcome_audio: '/w.mp3' }), '/w.mp3');
});

test('returns null on an empty or missing manifest', () => {
  assert.equal(pickInvitationAudio({}), null);
  assert.equal(pickInvitationAudio(null), null);
});

// ── mount ───────────────────────────────────────────────────────────────────
test('mount reveals the button when there is a source', () => {
  const { inv, button } = build();
  assert.equal(inv.mount(), true);
  assert.equal(button.classList.contains('hidden'), false);
});

test('mount is a no-op with no source and leaves the button hidden', () => {
  const { inv, button } = build({ src: null });
  assert.equal(inv.mount(), false);
  assert.equal(button.classList.contains('hidden'), true);
});

// ── play ────────────────────────────────────────────────────────────────────
test('play sets the source, restarts, and plays', () => {
  const { inv, audio } = build();
  assert.equal(inv.play(), true);
  assert.equal(audio.src, '/a.mp3');
  assert.equal(audio.currentTime, 0);
  assert.equal(audio._played, 1);
});

test('play is a no-op with no source', () => {
  const { inv, audio } = build({ src: null });
  assert.equal(inv.play(), false);
  assert.equal(audio._played, 0);
});

test('play posts the unlock message once, to the given origin', () => {
  const { inv, frame } = build();
  inv.play();
  assert.equal(frame.sent.length, 1);
  assert.deepEqual(frame.sent[0].msg, { type: UNLOCK_MSG });
  assert.equal(frame.sent[0].origin, 'https://illtowell.com');
});

test('playing twice still posts only one unlock message', () => {
  const { inv, frame } = build();
  inv.play();
  inv.stop();
  inv.play();
  assert.equal(frame.sent.length, 1);
});

test('play with no frame does not throw and still marks unlocked', () => {
  const { inv } = build({ frame: null });
  assert.doesNotThrow(() => inv.play());
  assert.equal(inv.unlocked, true);
});

// ── stop and toggle ─────────────────────────────────────────────────────────
test('stop pauses the audio', () => {
  const { inv, audio } = build();
  inv.play();
  inv.stop();
  assert.equal(audio._paused, 1);
});

test('toggle plays when idle and stops when playing', () => {
  const { inv, audio } = build();
  assert.equal(inv.toggle(), true);
  assert.equal(audio._played, 1);
  assert.equal(inv.toggle(), false);
  assert.equal(audio._paused, 1);
});

test('the audio ending returns the button to its idle label', () => {
  const { inv, audio, button } = build();
  inv.play();
  const playingLabel = button._label;
  audio.onended();
  assert.notEqual(button._label, playingLabel);
  assert.equal(inv.playing, false);
});

// ── graceful degradation ────────────────────────────────────────────────────
test('every method is safe with an entirely empty construction', () => {
  const inv = new Invitation({});
  assert.doesNotThrow(() => { inv.mount(); inv.play(); inv.stop(); inv.toggle(); inv.notifyUnlock(); });
});
```

- [ ] **Step 2: Run the test to verify it fails**

```bash
cd /tmp/wt-deploy-chat-b9535446 && node --test tests/begin_js/*.test.mjs
```

Expected: FAIL — `pickInvitationAudio` is not exported (the file currently exports the superseded video-tile interface).

- [ ] **Step 3: Write the implementation**

Replace the entire contents of `static/begin/invitation.js`:

```js
/* invitation.js — the spoken invitation behind the hero avatar's speaker button.
 *
 * The landing page already shows a muted, looping Glendalf video that links to
 * /begin/fireside. That clip carries no audio track, so this module plays a
 * separate voice-over on demand. The tap that starts it is also the browser
 * gesture that permits audio, so it is forwarded to the chat iframe, which from
 * then on speaks its replies instead of waiting for a Listen click.
 *
 * Every method is a safe no-op when its dependencies are missing, mirroring the
 * Director's degradation contract: a failed manifest fetch must leave the page
 * exactly as it was.
 */

export const UNLOCK_MSG = 'begin:audio-unlocked';

const ICON_IDLE = '🔈';
const ICON_PLAYING = '⏹';
const LABEL_IDLE = "Hear Dr. Glen's invitation";
const LABEL_PLAYING = 'Stop the invitation';

export function pickInvitationAudio(m, rand = Math.random) {
  if (!m) return null;
  const list = Array.isArray(m.intro_welcome_audios) && m.intro_welcome_audios.length
    ? m.intro_welcome_audios : [];
  if (list.length) return list[Math.floor(rand() * list.length)] || list[0];
  return m.intro_welcome_audio || null;
}

export class Invitation {
  constructor(opts = {}) {
    this.audio    = opts.audio || null;
    this.button   = opts.button || null;
    this.frame    = opts.frame || null;
    this.origin   = opts.origin || '*';
    this.src      = opts.src || null;
    this.unlocked = false;
    this.playing  = false;
  }

  _label(state) {
    if (!this.button) return;
    const on = state === 'playing';
    this.button.innerHTML = on ? ICON_PLAYING : ICON_IDLE;
    this.button.setAttribute('aria-label', on ? LABEL_PLAYING : LABEL_IDLE);
  }

  mount() {
    if (!this.src || !this.button) return false;
    this._label('idle');
    this.button.classList.remove('hidden');
    return true;
  }

  play() {
    if (!this.src || !this.audio) return false;
    this.audio.src = this.src;
    this.audio.currentTime = 0;
    this.audio.onended = () => { this.playing = false; this._label('idle'); };
    this.audio.play();
    this.playing = true;
    this._label('playing');
    this.notifyUnlock();
    return true;
  }

  stop() {
    if (!this.audio) return;
    this.audio.pause();
    this.playing = false;
    this._label('idle');
  }

  toggle() {
    if (this.playing) { this.stop(); return false; }
    return this.play();
  }

  notifyUnlock() {
    if (this.unlocked) return false;
    this.unlocked = true;
    if (this.frame && this.frame.contentWindow) {
      this.frame.contentWindow.postMessage({ type: UNLOCK_MSG }, this.origin);
    }
    return true;
  }
}
```

- [ ] **Step 4: Run the test to verify it passes**

```bash
cd /tmp/wt-deploy-chat-b9535446 && node --test tests/begin_js/*.test.mjs
```

Expected: PASS, 14 tests.

- [ ] **Step 5: Commit**

```bash
cd /tmp/wt-deploy-chat-b9535446
git add static/begin/invitation.js tests/begin_js/invitation.test.mjs
git commit -m "feat: spoken invitation module for the hero avatar speaker"
```

- [ ] **Step 6: Write the failing wiring test**

Create `tests/test_begin_invitation_wiring.py`:

```python
"""The served landing page must actually carry the speaker wiring.

These are deliberately shallow assertions on served HTML: the browser
behaviour (audio playback, click-through suppression, the audio unlock)
cannot be asserted headlessly and is covered by the manual verification
record instead.
"""
import importlib


def _reload_app(monkeypatch, tmp_path):
    monkeypatch.setenv("DATA_DIR", str(tmp_path))
    monkeypatch.setenv("FIRESIDE_ENABLED", "true")
    import app as appmod
    importlib.reload(appmod)
    return appmod


def test_landing_page_carries_the_speaker_button(monkeypatch, tmp_path):
    appmod = _reload_app(monkeypatch, tmp_path)
    body = appmod.app.test_client().get("/begin").get_data(as_text=True)
    assert 'id="avatar-speaker"' in body
    assert "/static/begin/invitation-mount.js" in body


def test_speaker_starts_hidden(monkeypatch, tmp_path):
    """It must not appear until a manifest audio source actually resolves."""
    appmod = _reload_app(monkeypatch, tmp_path)
    body = appmod.app.test_client().get("/begin").get_data(as_text=True)
    idx = body.index('id="avatar-speaker"')
    assert "hidden" in body[idx - 200 : idx + 200]


def test_speaker_lives_inside_the_existing_avatar_anchor(monkeypatch, tmp_path):
    """The single fireside door is the avatar; the speaker rides on it rather
    than becoming a second entry (see commit ec233b56)."""
    appmod = _reload_app(monkeypatch, tmp_path)
    body = appmod.app.test_client().get("/begin").get_data(as_text=True)
    anchor = body.index('<a class="avatar"')
    speaker = body.index('id="avatar-speaker"')
    closing = body.index("</a>", anchor)
    assert anchor < speaker < closing


def test_no_second_fireside_cta_was_added(monkeypatch, tmp_path):
    appmod = _reload_app(monkeypatch, tmp_path)
    body = appmod.app.test_client().get("/begin").get_data(as_text=True)
    assert body.count('href="/begin/fireside"') == 1


def test_orphaned_fireside_invite_css_is_gone(monkeypatch, tmp_path):
    """ec233b56 removed the standalone section but left its CSS behind."""
    appmod = _reload_app(monkeypatch, tmp_path)
    body = appmod.app.test_client().get("/begin").get_data(as_text=True)
    assert "fiGlow" not in body
    assert ".fi-inner" not in body


def test_invitation_modules_are_served(monkeypatch, tmp_path):
    appmod = _reload_app(monkeypatch, tmp_path)
    c = appmod.app.test_client()
    assert c.get("/static/begin/invitation.js").status_code == 200
    assert c.get("/static/begin/invitation-mount.js").status_code == 200
```

- [ ] **Step 7: Run it to verify it fails**

```bash
cd /tmp/wt-deploy-chat-b9535446 && doppler run --config dev -- python3 -m pytest tests/test_begin_invitation_wiring.py -v
```

Expected: FAIL — `assert 'id="avatar-speaker"' in body`.

- [ ] **Step 8a: Add the speaker button inside the existing avatar anchor**

In `static/begin.html`, find this exact block (around line 761):

```html
        <a class="avatar" href="/begin/fireside" aria-label="Sit by the fire with Dr. Glen">
          <span class="ember" aria-hidden="true"></span>
          <video class="avatar-video" autoplay muted loop playsinline preload="auto" poster="/static/media/glendalf-poster.jpg" aria-hidden="true">
            <source src="/static/media/glendalf-welcome.mp4" type="video/mp4">
          </video>
          <span class="avcap">Sit by the fire with Dr. Glen &rarr;</span>
        </a>
```

Add the speaker button as the last child of the anchor, immediately before `</a>`:

```html
          <button id="avatar-speaker" class="avatar-speaker hidden" type="button"
                  aria-label="Hear Dr. Glen&rsquo;s invitation">&#128264;</button>
        </a>
```

- [ ] **Step 8b: Add the styles and delete the orphaned block**

In `static/begin.html`, immediately after the existing `.avatar .avcap` rule (around line 589), add:

```css
  /* Speaker rides on the avatar. 44px minimum tap target, cornered clear of
     the caption, so a mis-tap opens fireside rather than doing nothing. */
  .avatar .avatar-speaker { position: absolute; top: 8px; right: 8px; z-index: 3;
    width: 44px; height: 44px; border-radius: 50%; cursor: pointer;
    background: rgba(20,12,6,0.72); border: 1px solid #6b5436; color: var(--cream);
    font-size: 17px; line-height: 1; display: flex; align-items: center;
    justify-content: center; }
  .avatar .avatar-speaker:hover { background: rgba(20,12,6,0.9); }
  .avatar .avatar-speaker.hidden { display: none !important; }
```

Then delete the orphaned block left by `ec233b56` — every rule for `#fireside-invite`,
`.fi-inner`, `.fi-title`, `.fi-sub`, `.fi-cta`, the `@keyframes fiGlow` block, and the
`@media (max-width: 760px)` rule that targets `#fireside-invite .fi-title`. The section
they styled no longer exists. Delete only those rules; leave everything around them
untouched.

- [ ] **Step 8c: Create the mount script**

Create `static/begin/invitation-mount.js`:

```js
/* invitation-mount.js — wires invitation.js to the hero avatar's speaker button.
 * Intentionally thin and untested: all branching logic lives in invitation.js.
 */
import { pickInvitationAudio, Invitation } from './invitation.js';

(function () {
  var button = document.getElementById('avatar-speaker');
  if (!button) return;

  fetch('/static/fireside/fireside-manifest.json')
    .then(function (r) { return r.ok ? r.json() : null; })
    .then(function (m) {
      var src = pickInvitationAudio(m);
      if (!src) return;                        // no voice-over: button stays hidden

      var inv = new Invitation({
        audio:  new Audio(),
        button: button,
        frame:  document.getElementById('begin-chat'),
        origin: window.location.origin,
        src:    src,
      });

      // The speaker lives inside the fireside anchor. Both calls are required:
      // preventDefault stops the navigation, stopPropagation keeps the click
      // away from the anchor's engagement handler.
      button.addEventListener('click', function (e) {
        e.preventDefault();
        e.stopPropagation();
        inv.toggle();
      });

      inv.mount();
    })
    .catch(function () { /* manifest unavailable: leave the page as it was */ });
})();
```

- [ ] **Step 8d: Load the module**

Immediately before the closing `</body>` of `static/begin.html`, add:

```html
<script type="module" src="/static/begin/invitation-mount.js"></script>
```

- [ ] **Step 9: Run the wiring test to verify it passes**

```bash
cd /tmp/wt-deploy-chat-b9535446 && doppler run --config dev -- python3 -m pytest tests/test_begin_invitation_wiring.py -v
```

Expected: PASS, 6 tests.

- [ ] **Step 10: Commit**

```bash
cd /tmp/wt-deploy-chat-b9535446
git add static/begin.html static/begin/invitation-mount.js tests/test_begin_invitation_wiring.py
git commit -m "feat: speaker button on the hero avatar plays the invitation and unlocks audio"
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
cd /tmp/wt-deploy-chat-b9535446 && doppler run --config dev -- python3 -m pytest tests/test_begin_invitation_wiring.py -v
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
cd /tmp/wt-deploy-chat-b9535446 && doppler run --config dev -- python3 -m pytest tests/test_begin_invitation_wiring.py -v
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
cd /tmp/wt-deploy-chat-b9535446 && doppler run --config dev -- python3 -m pytest tests/test_begin_invitation_wiring.py -v
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
cd /tmp/wt-deploy-chat-b9535446 && doppler run --config dev -- python3 -m pytest tests/test_begin_invitation_wiring.py -v
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
doppler run --config dev -- python3 -m pytest tests/test_begin_invitation_wiring.py tests/test_fireside_routes.py tests/test_begin_routes.py tests/test_chat_tts.py -v
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
| 1. Speaker button on the existing hero avatar (sibling in `.avatar-wrap`, mute-video passthrough, idle/playing states) | 1 |
| 2. Audio unlock (session flag, postMessage, attachAndSpeak, Listen retained) | 1, 3 |
| 3. Handoff (plain navigation, no context carried) | 1 (`href="/begin/fireside"` unchanged on the existing anchor) |
| 4. Fullscreen (requestFullscreen, webkit prefix, fullscreenchange, iOS hidden) | 4 |
| Testing 1-2 (headless renders) | 5 steps 2 |
| Testing 3-5 (manual passes) | 5 step 3 |
| Risk: swallowed navigation | Not applicable as shipped — the button is a sibling inside `.avatar-wrap`, not a descendant of the anchor, so the click cannot reach the anchor's handler regardless of `preventDefault`/`stopPropagation` |
| Risk: unexpected speech | 1 (Listen button retained) + 5 step 3.4 |

No gaps.

**Placeholder scan:** none. Every code step carries complete code; every command is exact.

**Type consistency:** `UNLOCK_MSG` is `'begin:audio-unlocked'` in Task 1 and the same literal appears in Task 3's listener and its test. `Invitation` constructor keys (`audio, button, frame, origin, src`) match exactly between Task 1's class, Task 1's test helper, and `invitation-mount.js`. The element id `#avatar-speaker` and the `.avatar-wrap` sibling structure are identical across `static/begin.html`'s markup, styles, and `invitation-mount.js`'s `getElementById` lookup. `#fsBtn` and `#fireside` match between Task 4's markup, handler, and tests.
