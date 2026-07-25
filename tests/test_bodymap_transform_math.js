// tests/test_bodymap_transform_math.js
// Run: node tests/test_bodymap_transform_math.js
const assert = require('assert');
const { bmTransformFromParams, bmTransformParams } = require('../static/body-map.js');

// A saved {mx,my,tx,ty} reconstructs the SAME mapping fitSimilarity produced.
const steps = [
  { template: { x: 0, y: 0 }, key: 'a' },
  { template: { x: 1, y: 0 }, key: 'b' },
];
const anchors = { a: { x: 100, y: 100 }, b: { x: 300, y: 100 } };  // scale 200, no rotation
const p = bmTransformParams(steps, anchors);
const fn = bmTransformFromParams(p);

// template (0,0)->(100,100); (1,0)->(300,100); (0,1)-> rotated by the same similarity
const A = fn({ x: 0, y: 0 }), B = fn({ x: 1, y: 0 });
assert.ok(Math.abs(A.x - 100) < 1e-6 && Math.abs(A.y - 100) < 1e-6);
assert.ok(Math.abs(B.x - 300) < 1e-6 && Math.abs(B.y - 100) < 1e-6);

// round-trip through JSON (what the endpoint stores) is identical
const p2 = JSON.parse(JSON.stringify(p));
const fn2 = bmTransformFromParams(p2);
const C = fn({ x: 0.37, y: 0.81 }), D = fn2({ x: 0.37, y: 0.81 });
assert.ok(Math.abs(C.x - D.x) < 1e-9 && Math.abs(C.y - D.y) < 1e-9);

console.log('ok - bodymap transform math round-trips');
