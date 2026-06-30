import { test } from 'node:test';
import assert from 'node:assert/strict';
import { canFireBackchannel, canInterject, canInterrupt, nextGapMs, shouldDuck } from '../../static/fireside/governance.js';

test('backchannel respects 5s min gap', () => {
  assert.equal(canFireBackchannel(10000, 4000), true);   // 6s gap
  assert.equal(canFireBackchannel(10000, 6000), false);  // 4s gap
  assert.equal(canFireBackchannel(10000, null), true);   // never fired
});

test('interjection gate: idle>=3.5s, <3 this session, never turn 1', () => {
  assert.equal(canInterject({ idleMs: 4000, sessionCount: 0, turn: 3 }), true);
  assert.equal(canInterject({ idleMs: 2000, sessionCount: 0, turn: 3 }), false);
  assert.equal(canInterject({ idleMs: 4000, sessionCount: 3, turn: 3 }), false);
  assert.equal(canInterject({ idleMs: 4000, sessionCount: 0, turn: 1 }), false);
});

test('interruption gate: not seen, turn>=3, idle>=2.5s', () => {
  assert.equal(canInterrupt({ seen: false, turn: 3, idleMs: 3000 }), true);
  assert.equal(canInterrupt({ seen: true, turn: 3, idleMs: 3000 }), false);
  assert.equal(canInterrupt({ seen: false, turn: 1, idleMs: 3000 }), false);
  assert.equal(canInterrupt({ seen: false, turn: 3, idleMs: 1000 }), false);
});

test('nextGapMs lands within [min,max] seconds', () => {
  const o = { min_gap_s: 25, max_gap_s: 70 };
  assert.equal(nextGapMs(o, () => 0), 25000);
  assert.equal(nextGapMs(o, () => 1), 70000);
  const mid = nextGapMs(o, () => 0.5);
  assert.ok(mid >= 25000 && mid <= 70000);
});

test('shouldDuck while voice plays', () => {
  assert.equal(shouldDuck(true), true);
  assert.equal(shouldDuck(false), false);
});
