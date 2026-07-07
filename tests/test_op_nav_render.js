// tests/test_op_nav_render.js — run: node tests/test_op_nav_render.js
// Structural check on the exported nav data (op-nav render is DOM-only; this guards the
// ordered labels + logo Home + that every page id has a landing href).
const assert = require("assert");
const nav = require("../static/op-nav.js");

const labels = nav.PILLARS.map(p => p.logo ? "[logo]" : p.label);
assert.ok(labels.join(" ").indexOf("Market Match Sell Fill People Comm") !== -1, "front labels in order");
assert.strictEqual(nav.PILLARS[0].logo, true, "home is logo");

// every pillar has at least one page, and every page has an href
for (const id of Object.keys(nav.PILLAR_PAGES)) {
  const pages = nav.PILLAR_PAGES[id];
  assert.ok(Array.isArray(pages) && pages.length >= 1, id + " has pages");
  pages.forEach(p => { assert.ok(p.id && typeof p.href === "string" && p.href.startsWith("/"), id + "/" + p.id + " href"); });
}
console.log("op-nav render data OK:", labels.join(" · "));
