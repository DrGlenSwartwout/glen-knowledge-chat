// Integration test: drive the real Director through the full dance against a
// fake DOM, for both an empty manifest (graceful degradation) and a fixture
// manifest (the dance). Verifies no-throws + correct video-state transitions
// without a browser. Visual/pixel quality is reviewed manually (see
// tests/fireside_render_verify.md).
import { test } from 'node:test';
import assert from 'node:assert/strict';
import { normalizeManifest } from '../../static/fireside/manifest-normalize.js';
import { Director } from '../../static/fireside/director.js';

// The Director schedules safety/pause timeouts; make them inert so the test
// stays synchronous and leaks no open handles. We drive `onended` manually.
const realSetTimeout = globalThis.setTimeout;
globalThis.setTimeout = () => 0;

function fakeVideo() {
  return {
    src: '', loop: false, currentTime: 0, _paused: false, onended: null,
    style: {},
    play() { return { catch() {} }; },
    pause() { this._paused = true; },
  };
}
function mkDirector(rawManifest) {
  const a = fakeVideo(), b = fakeVideo();
  const interjAudio = [];
  const d = new Director(normalizeManifest(rawManifest), {
    videoA: a, videoB: b,
    onInterjectionAudio: (clip) => interjAudio.push(clip),
  });
  return { d, a, b, interjAudio, srcs: () => [a.src, b.src] };
}
const R = (o) => ({ id: o.id, family: o.family, form: o.form || 'silent', gaze: o.gaze || null,
  hand: null, intensity: o.intensity || 'med', tier: o.tier, duration_s: 2, loopable: !!o.loopable,
  file: '/v/' + o.id + '.mp4', audio: null });

test('empty manifest: every Director method is a no-op that never throws', () => {
  const { d, srcs } = mkDirector({});
  assert.doesNotThrow(() => d.toResting());
  assert.doesNotThrow(() => d.onType('I feel heavy pain', { nowMs: 1000, turn: 2 }));
  assert.doesNotThrow(() => d.onSubmit('I feel heavy pain'));
  assert.doesNotThrow(() => d.maybeInterject({ idleMs: 5000, sessionCount: 0, turn: 3 }));
  assert.doesNotThrow(() => d.maybeInterruption({ turn: 3, idleMs: 3000 }));
  assert.doesNotThrow(() => d.onReplyReady());
  // No clips anywhere → no video src was ever set.
  assert.deepEqual(srcs(), ['', '']);
});

test('fixture manifest: toResting plays a looping resting clip', () => {
  const { d, srcs } = mkDirector({ resting_loops: ['/v/rest.mp4'] });
  d.toResting();
  assert.ok(srcs().includes('/v/rest.mp4'));
  // the now-front video loops
  const front = d.front;
  assert.equal(front.src, '/v/rest.mp4');
  assert.equal(front.loop, true);
});

const FIXTURE = {
  resting_loops: ['/v/rest.mp4'],
  pondering_loops: ['/v/ponder.mp4'],
  speaking_loop: '/v/speak.mp4',
  reactions: [
    R({ id: 'bc-attend', family: 'attending', tier: 'backchannel' }),
    R({ id: 'bc-concern', family: 'empathic_concern', tier: 'backchannel' }),
    R({ id: 'gaze-dr', family: 'attending', tier: 'gaze', gaze: 'down_right' }),
    R({ id: 'hero-concern', family: 'empathic_concern', tier: 'hero', intensity: 'high' }),
    R({ id: 'pon-1', family: 'pondering', tier: 'ponder', loopable: true }),
    R({ id: 'inter-1', family: 'invitation', tier: 'interruption' }),
    R({ id: 'inter-2', family: 'invitation', tier: 'interruption' }),
  ],
};
const FILES = new Set(FIXTURE.reactions.map((r) => r.file));

test('onType crosses to a reaction clip, then onended returns to resting', () => {
  const { d } = mkDirector(FIXTURE);
  d.toResting();
  d.onType('I feel such heavy pain', { nowMs: 10000, turn: 2 });
  // a reaction (gaze or backchannel) is now the front clip
  assert.ok(FILES.has(d.front.src), `front should be a reaction clip, got ${d.front.src}`);
  // simulate the clip ending → returns to resting
  assert.equal(typeof d.front.onended, 'function');
  d.front.onended();
  assert.equal(d.front.src, '/v/rest.mp4');
});

test('onType is rate-gated: a second call within 5s does not start a new reaction', () => {
  const { d } = mkDirector(FIXTURE);
  d.toResting();
  d.onType('I feel heavy pain', { nowMs: 10000, turn: 2 });
  const firstSrc = d.front.src;
  d.onType('still in pain', { nowMs: 11000, turn: 2 }); // 1s later — gated
  assert.equal(d.front.src, firstSrc, 'no new reaction within the 5s window');
});

test('onSubmit plays a hero clip, then pondering on its end', () => {
  const { d } = mkDirector(FIXTURE);
  d.toResting();
  d.onSubmit('I feel such heavy pain');
  assert.equal(d.front.src, '/v/hero-concern.mp4');
  assert.equal(typeof d.front.onended, 'function');
  d.front.onended(); // → _ponder
  assert.equal(d.front.src, '/v/pon-1.mp4');
  assert.equal(d.front.loop, true);
});

test('onSubmit does NOT fire a weighty hero on light/unrelated content (strictFamily gate)', () => {
  // hero-concern is empathic_concern; a delight message must not trigger it — it pondering instead.
  const { d } = mkDirector(FIXTURE);
  d.toResting();
  d.onSubmit('what a wonderful delightful joyful day');
  assert.equal(d.front.src, '/v/pon-1.mp4', 'unrelated family → no hero, straight to ponder');
  assert.equal(d.front.loop, true);
});

test('onReplyReady crosses to the speaking loop', () => {
  const { d } = mkDirector(FIXTURE);
  d.toResting();
  d.onReplyReady();
  assert.equal(d.front.src, '/v/speak.mp4');
  assert.equal(d.front.loop, true);
});

test('maybeInterruption fires once per client, then never again', () => {
  const { d, interjAudio } = mkDirector(FIXTURE);
  d.toResting();
  const first = d.maybeInterruption({ turn: 3, idleMs: 3000 });
  assert.ok(first, 'first eligible call plays an interruption');
  assert.equal(d._interruptionSeen, true);
  assert.equal(interjAudio.length, 1);
  const second = d.maybeInterruption({ turn: 4, idleMs: 4000 });
  assert.equal(second, null, 'second call is suppressed (once per client)');
  assert.equal(interjAudio.length, 1);
});

test('maybeInterruption respects the gate (never turn 1)', () => {
  const { d } = mkDirector(FIXTURE);
  assert.equal(d.maybeInterruption({ turn: 1, idleMs: 5000 }), null);
  assert.equal(d._interruptionSeen ?? false, false);
});

test('missing-clip safety: reactions present but no hero → onSubmit falls back, no throw', () => {
  const noHero = { ...FIXTURE, reactions: FIXTURE.reactions.filter((r) => r.tier !== 'hero') };
  const { d } = mkDirector(noHero);
  d.toResting();
  assert.doesNotThrow(() => d.onSubmit('I feel heavy pain'));
  // falls through to pondering
  assert.equal(d.front.src, '/v/pon-1.mp4');
});

globalThis.setTimeout = realSetTimeout;
