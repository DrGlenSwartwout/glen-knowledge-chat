import { test } from 'node:test';
import assert from 'node:assert/strict';
import { selectClip, FAMILY_AFFINITY } from '../../static/fireside/reaction-select.js';

const R = (o) => ({ id: o.id, family: o.family, form: o.form || 'silent', gaze: o.gaze || null,
  hand: null, intensity: o.intensity || 'med', tier: o.tier || 'backchannel',
  duration_s: 2, loopable: false, file: '/' + o.id + '.mp4', audio: null });

const pool = [
  R({ id: 'att', family: 'attending', tier: 'backchannel' }),
  R({ id: 'con', family: 'empathic_concern', tier: 'backchannel' }),
  R({ id: 'grav', family: 'gentle_gravity', tier: 'backchannel' }),
  R({ id: 'hero-con', family: 'empathic_concern', tier: 'hero', intensity: 'high' }),
  R({ id: 'gaze-dr', family: 'attending', tier: 'gaze', gaze: 'down_right' }),
];

test('exact family + tier match', () => {
  const c = selectClip(pool, { tier: 'backchannel', family: 'empathic_concern' });
  assert.equal(c.id, 'con');
});

test('affinity fallback when family absent (empathic_concern -> gentle_gravity)', () => {
  const noCon = pool.filter((x) => x.id !== 'con');
  const c = selectClip(noCon, { tier: 'backchannel', family: 'empathic_concern' });
  assert.equal(c.id, 'grav');
  assert.ok(FAMILY_AFFINITY.empathic_concern.includes('gentle_gravity'));
});

test('gaze preference within tier', () => {
  const c = selectClip(pool, { tier: 'gaze', gaze: 'down_right' });
  assert.equal(c.id, 'gaze-dr');
});

test('returns null on empty tier pool', () => {
  assert.equal(selectClip(pool, { tier: 'interjection' }), null);
});

test('avoids repeating lastId when an alternative exists', () => {
  const two = [R({ id: 'a', family: 'attending' }), R({ id: 'b', family: 'attending' })];
  const rng = () => 0; // deterministic: would pick index 0
  const c = selectClip(two, { tier: 'backchannel', family: 'attending' }, 'a', rng);
  assert.equal(c.id, 'b');
});
