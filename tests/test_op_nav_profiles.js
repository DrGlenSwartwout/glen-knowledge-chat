// tests/test_op_nav_profiles.js — run: node tests/test_op_nav_profiles.js
const assert = require("assert");
const nav = require("../static/op-nav.js");   // op-nav.js must export when required in node

// all 11 pillars exist, ordered, with zones
const ids = nav.PILLARS.map(p => p.id);
assert.deepStrictEqual(ids, ["home","marketing","clinical","sales","fulfillment","people",
  "communication","rnd","production","finance","admin"], "pillar order");
assert.strictEqual(nav.PILLARS.find(p=>p.id==="clinical").label, "Match");
assert.strictEqual(nav.PILLARS.find(p=>p.id==="production").label, "Make");
assert.strictEqual(nav.PILLARS.find(p=>p.id==="finance").label, "Money");
assert.strictEqual(nav.PILLARS.filter(p=>p.zone==="back").map(p=>p.id).join(","),
  "rnd,production,finance,admin", "back-office zone");

// Shaira sees exactly her four; glen sees all; rae is a subset
assert.deepStrictEqual(nav.navPillarsFor("shaira"),
  ["home","marketing","communication","people"], "shaira scope");
assert.deepStrictEqual(nav.navPillarsFor("glen"), ids, "glen full");
assert.ok(nav.navPillarsFor("shaira").indexOf("finance") === -1, "shaira cannot see finance");

// sub-pages: Orders hub + Social + rnd stub present
assert.ok(nav.PILLAR_PAGES.sales.some(p=>p.id==="new-order"));
assert.ok(nav.PILLAR_PAGES.marketing.some(p=>p.id==="social"));
assert.ok(nav.PILLAR_PAGES.rnd.some(p=>p.id==="rnd"));
console.log("op-nav profiles OK");
