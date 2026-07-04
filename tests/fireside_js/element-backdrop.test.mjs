import { test } from 'node:test';
import assert from 'node:assert/strict';
import { readFileSync } from 'node:fs';
import { fileURLToPath } from 'node:url';
import { pickElement, ELEMENT_KEYS } from '../../static/fireside/element-backdrop.js';

const MANIFEST = JSON.parse(
  readFileSync(fileURLToPath(new URL('../../static/fireside/elements-manifest.json', import.meta.url)), 'utf8')
);

test('null/garbage setting yields null (graceful default: plain portal)', () => {
  for (const bad of [null, undefined, 42, '', ' ', 'sky', 'aether', [], {}]) {
    assert.equal(pickElement(bad, MANIFEST), null);
  }
});

test('each of the five elements resolves to a video + poster + ambience', () => {
  for (const key of ELEMENT_KEYS) {
    const e = pickElement(key, MANIFEST);
    assert.ok(e, `${key} should resolve`);
    assert.equal(e.key, key);
    assert.match(e.video, /\/static\/fireside\/video\/elements\//);
    assert.ok(e.poster, `${key} has a poster`);
    // bed is optional (Metal is intentionally bedless — a constant tone reads as hum);
    // every element must have at least one ambient one-shot.
    assert.ok(e.ambience && typeof e.ambience === 'object', `${key} has ambience`);
    assert.ok(Array.isArray(e.ambience.oneshots) && e.ambience.oneshots.length, `${key} has oneshots`);
  }
});

test('setting is case/space-insensitive (API sends lowercase, but be defensive)', () => {
  assert.equal(pickElement('  FIRE ', MANIFEST).key, 'fire');
  assert.equal(pickElement('Water', MANIFEST).key, 'water');
});

test('missing/garbage manifest never throws, returns null', () => {
  for (const bad of [null, undefined, {}, { elements: null }, { elements: { fire: { label: 'Fire' } } }]) {
    assert.equal(pickElement('fire', bad), null); // last case: entry present but no video
  }
});

test('manifest covers exactly the five canonical elements', () => {
  assert.deepEqual(Object.keys(MANIFEST.elements).sort(), [...ELEMENT_KEYS].sort());
});
