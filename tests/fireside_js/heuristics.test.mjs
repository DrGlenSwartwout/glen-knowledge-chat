import { test } from 'node:test';
import assert from 'node:assert/strict';
import { detectGaze, detectFamily, detectIntensity, classifyTyping } from '../../static/fireside/heuristics.js';

test('detectGaze maps rep-system predicates to the right triad', () => {
  assert.equal(detectGaze('I can see it clearly, a dark picture'), 'up_right');
  assert.equal(detectGaze('it sounds loud, I hear ringing'), 'lat_right');
  assert.equal(detectGaze('I feel a heavy pain, so tense'), 'down_right');
  assert.equal(detectGaze('the cat sat on the mat'), null);
});

test('detectFamily picks dominant emotional family, defaults attending', () => {
  assert.equal(detectFamily('I am in so much pain and feel alone'), 'empathic_concern');
  assert.equal(detectFamily('wow that is amazing, I love it!'), 'delight');
  assert.equal(detectFamily('what if it is something else? I wonder'), 'curiosity');
  assert.equal(detectFamily('whoa, I never expected that!'), 'surprise');
  assert.equal(detectFamily('and then I went to the store'), 'attending');
});

test('detectIntensity reads punctuation/emphasis', () => {
  assert.equal(detectIntensity('help!'), 'high');
  assert.equal(detectIntensity('I can NEVER do this'), 'high');
  assert.equal(detectIntensity('ok'), 'low');
  assert.equal(detectIntensity('I went for a walk this morning and it was fine'), 'med');
});

test('classifyTyping combines the three', () => {
  const c = classifyTyping('I feel such heavy pain!');
  assert.equal(c.gaze, 'down_right');
  assert.equal(c.family, 'empathic_concern');
  assert.equal(c.intensity, 'high');
});
