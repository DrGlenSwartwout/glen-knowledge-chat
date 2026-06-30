import { test } from 'node:test';
import assert from 'node:assert/strict';
import { spawnEmbers } from '../../static/fireside/spark.js';

test('spawnEmbers makes N particles rising from the origin', () => {
  const ps = spawnEmbers(100, 200, 8, () => 0.5);
  assert.equal(ps.length, 8);
  for (const p of ps) {
    assert.equal(p.x, 100);
    assert.equal(p.y, 200);
    assert.ok(p.vy < 0);            // embers rise
    assert.ok(p.life > 0);
  }
});
