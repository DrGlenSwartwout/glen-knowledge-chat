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
