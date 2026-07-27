// tests/test_ff_draft_canon_render.js
// Run: node tests/test_ff_draft_canon_render.js
const assert = require('assert');
const { renderCanonBlock } = require('../static/js/ff-draft-canon.js');

// empty / absent -> nothing (no empty section)
assert.strictEqual(renderCanonBlock(null), '');
assert.strictEqual(renderCanonBlock({}), '');
assert.strictEqual(renderCanonBlock({ conditions: [], challenges: '' }), '');

// populated -> a labeled block that is NOT an .item, with the values
const html = renderCanonBlock({
  conditions: ['glaucoma', 'ocular hypertension'],
  terrain_concerns: ['oxidative stress'],
  challenges: 'fatigue'
});
assert.ok(/class="canon"/.test(html));          // distinct class
assert.ok(!/class="item"/.test(html));          // NOT an .item (never collected/published)
assert.ok(/records/i.test(html));               // a "from the records" label
assert.ok(html.includes('glaucoma') && html.includes('ocular hypertension'));
assert.ok(html.includes('oxidative stress') && html.includes('fatigue'));
// omit empty fields
assert.ok(!/body.?systems/i.test(html));

// escaping
const evil = renderCanonBlock({ conditions: ['<script>alert(1)</script>'] });
assert.ok(!evil.includes('<script>alert(1)</script>'));
assert.ok(evil.includes('&lt;script&gt;'));

console.log('ok - ff-draft canon render');
