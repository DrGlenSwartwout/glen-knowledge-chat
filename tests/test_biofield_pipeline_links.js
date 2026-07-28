const assert = require("assert");
const fs = require("fs");
const path = require("path");

const html = fs.readFileSync(
  path.join(__dirname, "..", "static", "console-handoffs.html"),
  "utf8"
);
const scripts = [...html.matchAll(/<script(?:\s[^>]*)?>([\s\S]*?)<\/script>/g)]
  .map((match) => match[1])
  .filter(Boolean);
const source = scripts.join("\n").replace(/\bboot\(\);\s*$/, "");

global.localStorage = {
  getItem() { return "sek"; },
  setItem() {}
};
global.location = { search: "" };
global.document = { getElementById() { return {}; } };

eval(source);

const steps = {
  paid: { done: true, order_id: 42 },
  handed_off: { done: true, awaiting_publish: false },
  analysis_published: { done: true },
  invoice_published: { done: true, order_id: 84 },
  invoice_paid: { done: true },
  fulfilled: { done: true }
};
const rendered = card({
  name: "Test Client",
  email: "client@example.com",
  steps,
  done_count: 6,
  complete: true
});

const labels = [...rendered.matchAll(/<a class="slabel" href="([^"]+)">/g)]
  .map((match) => match[1]);

assert.strictEqual(labels.length, 6, "every pipeline item should be a link");
assert.match(labels[0], /^\/console\/orders\?order=42/);
assert.match(labels[1], /^\/console\/biofield-portal\?email=client%40example\.com/);
assert.match(labels[2], /^\/console\/biofield-portal\?email=client%40example\.com/);
assert.match(labels[3], /^\/console\/orders\?order=84/);
assert.match(labels[4], /^\/console\/orders\?order=84/);
assert.match(labels[5], /^\/console\/orders\?order=84/);

const unfinishedIntake = STEPS.find((step) => step.k === "handed_off");
assert.match(unfinishedIntake.href({ email: "client@example.com" }, { done: false }),
             /^\/console\/biofield-intake\?key=/);
