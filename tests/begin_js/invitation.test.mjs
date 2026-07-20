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

test('replaying re-posts the unlock (the receiver is idempotent)', () => {
  const { inv, frame } = build();
  inv.play();
  inv.stop();
  inv.play();
  assert.equal(frame.sent.length, 2, 'a lost first post must be recoverable by replaying');
  assert.deepEqual(frame.sent[1].msg, { type: UNLOCK_MSG });
});

test('notifyUnlock reports the state change only once', () => {
  const { inv } = build();
  assert.equal(inv.notifyUnlock(), true);
  assert.equal(inv.notifyUnlock(), false);
  assert.equal(inv.unlocked, true);
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

// ── one voice at a time ─────────────────────────────────────────────────────
test('play() silences a chat reply that is already speaking', () => {
  let stopped = 0;
  globalThis.window = { TTS: { stop: () => { stopped++; } } };
  try {
    const { inv } = build();
    inv.play();
    assert.equal(stopped, 1, 'the invitation must take the audio channel');
  } finally { delete globalThis.window; }
});

test('play() still works when TTS is absent', () => {
  const { inv, audio } = build();
  assert.doesNotThrow(() => inv.play());
  assert.equal(audio._played, 1);
});
