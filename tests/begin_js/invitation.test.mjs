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

test('tap is a no-op with no clip', () => {
  const { inv, video } = build({ clip: null });
  assert.equal(inv.tap(), false);
  assert.equal(video._played, 0);
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
