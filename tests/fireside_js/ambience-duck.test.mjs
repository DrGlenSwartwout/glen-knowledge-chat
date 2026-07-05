import { test } from 'node:test';
import assert from 'node:assert/strict';
import { Ambience, pickSrc } from '../../static/fireside/ambience.js';

test('pickSrc: single file, files-array variants, or nothing', () => {
  assert.equal(pickSrc({ file: '/a.mp3' }), '/a.mp3');
  const files = ['/x.mp3', '/y.mp3', '/z.mp3'];
  assert.equal(pickSrc({ files }, () => 0), '/x.mp3');       // rng->first
  assert.equal(pickSrc({ files }, () => 0.99), '/z.mp3');    // rng->last
  assert.ok(files.includes(pickSrc({ files })));             // real rng stays in set
  assert.equal(pickSrc({ files: [], file: '/f.mp3' }), '/f.mp3'); // empty array -> file
  assert.equal(pickSrc({}), null);
  assert.equal(pickSrc(null), null);
});

const CFG = { bed: '/b.mp3', bed_volume: 0.1, oneshots: [] };

test('voice ducking is opt-in: no duck opt -> never ducks the bed level', () => {
  const a = new Ambience(CFG, {});          // no duck -> disabled
  assert.equal(a.duck, null);
  a._ducked = true;                          // even if flagged ducked
  assert.equal(a._levelFactor(), 1);         // level unchanged
});

test('duck opt sets the ducked level factor; composes with mute', () => {
  const a = new Ambience(CFG, { duck: 0.35 });
  assert.equal(a.duck, 0.35);
  assert.equal(a._levelFactor(), 1);         // idle
  a._ducked = true;
  assert.equal(a._levelFactor(), 0.35);      // ducked under voice
  a.muted = true;
  assert.equal(a._levelFactor(), 0);         // mute wins over duck
});

test('duck opt is clamped to (0,1); junk disables ducking', () => {
  for (const bad of [0, 1, 1.5, -0.2, 'x', null, undefined, NaN]) {
    const a = new Ambience(CFG, { duck: bad });
    assert.equal(a.duck, null, `duck=${bad} should disable`);
    a._ducked = true;
    assert.equal(a._levelFactor(), 1);
  }
});

test('alternate one-shot cycles variants round-robin at each item volume (never overlaps)', () => {
  const plays = [];
  global.Audio = class { constructor(src) { this.src = src; this.volume = 1; }
    play() { plays.push({ src: this.src, volume: this.volume }); return { catch() {} }; } };
  try {
    const a = new Ambience(CFG, {});
    const o = { alternate: [{ file: '/bowl.mp3', volume: 0.05 }, { file: '/chant.mp3', volume: 0.2 }] };
    a._play(o); a._play(o); a._play(o); a._play(o);
    // one clip per fire, strictly alternating (bowl first), each at its own volume
    assert.deepEqual(plays.map(p => p.src), ['/bowl.mp3', '/chant.mp3', '/bowl.mp3', '/chant.mp3']);
    assert.deepEqual(plays.map(p => p.volume), [0.05, 0.2, 0.05, 0.2]);
  } finally { delete global.Audio; }
});
