import { test } from 'node:test';
import assert from 'node:assert/strict';
import { normalizeManifest } from '../../static/fireside/manifest-normalize.js';

test('null/garbage input yields safe empty manifest, never throws', () => {
  for (const bad of [null, undefined, 42, 'x', []]) {
    const m = normalizeManifest(bad);
    assert.deepEqual(m.pondering_loops, []);
    assert.deepEqual(m.resting_loops, []);
    assert.deepEqual(m.reactions, []);
    assert.equal(m.ambience.bed, null);
    assert.deepEqual(m.ambience.oneshots, []);
    assert.equal(m.ambience.bed_volume, 0.18);
  }
});

test('resting_loops falls back to pondering_loops then speaking_loop', () => {
  const a = normalizeManifest({ pondering_loops: ['/p1.mp4'], speaking_loop: '/s.mp4' });
  assert.deepEqual(a.resting_loops, ['/p1.mp4']);
  const b = normalizeManifest({ speaking_loop: '/s.mp4' });
  assert.deepEqual(b.resting_loops, ['/s.mp4']);
  const c = normalizeManifest({ resting_loops: ['/r.mp4'], pondering_loops: ['/p.mp4'] });
  assert.deepEqual(c.resting_loops, ['/r.mp4']);
});

test('reactions: bad entries skipped, defaults filled', () => {
  const m = normalizeManifest({ reactions: [
    { id: 'ok', family: 'surprise', form: 'silent', tier: 'backchannel', file: '/r.mp4' },
    { id: 'nofile', family: 'surprise' },         // dropped: no file
    { family: 'surprise', file: '/x.mp4' },        // dropped: no id
    'garbage',
  ]});
  assert.equal(m.reactions.length, 1);
  const r = m.reactions[0];
  assert.equal(r.gaze, null);
  assert.equal(r.hand, null);
  assert.equal(r.intensity, 'med');
  assert.equal(r.loopable, false);
  assert.equal(typeof r.duration_s, 'number');
});

test('ambience oneshots: defaults + bad entries skipped', () => {
  const m = normalizeManifest({ ambience: { bed: '/b.mp3', oneshots: [
    { id: 'pop', file: '/p.mp3', spark: true, min_gap_s: 25, max_gap_s: 70 },
    { id: 'nofile' },          // dropped
  ]}});
  assert.equal(m.ambience.bed, '/b.mp3');
  assert.equal(m.ambience.oneshots.length, 1);
  assert.equal(m.ambience.oneshots[0].volume, 0.2);   // default
  assert.equal(m.ambience.oneshots[0].spark, true);
});

test('speaking_loops: array supported; falls back to single speaking_loop', () => {
  // explicit array
  const a = normalizeManifest({ speaking_loops: ['/s1.mp4', '/s2.mp4', 42, ''] });
  assert.deepEqual(a.speaking_loops, ['/s1.mp4', '/s2.mp4']);
  assert.equal(a.speaking_loop, '/s1.mp4');           // first, for back-compat
  // fallback: only single speaking_loop -> becomes a 1-element array
  const b = normalizeManifest({ speaking_loop: '/only.mp4' });
  assert.deepEqual(b.speaking_loops, ['/only.mp4']);
  assert.equal(b.speaking_loop, '/only.mp4');
  // neither -> empty array, null loop
  const c = normalizeManifest({});
  assert.deepEqual(c.speaking_loops, []);
  assert.equal(c.speaking_loop, null);
});

test('ambience oneshot: loop flag defaults false, honored when true', () => {
  const m = normalizeManifest({ ambience: { oneshots: [
    { id: 'rain', file: '/rain.mp3', loop: true, volume: 0.1 },
    { id: 'pop', file: '/pop.mp3' },        // loop defaults false
  ]}});
  assert.equal(m.ambience.oneshots[0].loop, true);
  assert.equal(m.ambience.oneshots[1].loop, false);
});

test('intro book-flow fields: read / welcome / welcome_audio pass through, default null', () => {
  const a = normalizeManifest({ intro_read: '/r.mp4', intro_welcome: '/w.mp4', intro_welcome_audio: '/w.mp3' });
  assert.equal(a.intro_read, '/r.mp4');
  assert.equal(a.intro_welcome, '/w.mp4');
  assert.equal(a.intro_welcome_audio, '/w.mp3');
  const b = normalizeManifest({});
  assert.equal(b.intro_read, null);
  assert.equal(b.intro_welcome, null);
});
