"use strict";

const assert = require("assert");
const clock = require("../static/organ-clock.js");

assert.strictEqual(clock.windowForHour(0).code, "GB");
assert.strictEqual(clock.windowForHour(3).code, "LU");
assert.strictEqual(clock.windowForHour(22.75).code, "TE");
assert.strictEqual(clock.angleForHour(3), 0);
assert.strictEqual(clock.angleForHour(15), 180);

let mentions = clock.extractTimeMentions("I woke around 3:30 AM and felt tired all morning.");
assert.strictEqual(mentions.length, 2);
assert.deepStrictEqual(mentions[0].codes, ["LU"]);
assert.ok(mentions[1].codes.includes("LI"));
assert.ok(mentions[1].codes.includes("HT"));

mentions = clock.extractTimeMentions("The discomfort usually starts from 7 to 9 pm.");
assert.strictEqual(mentions.length, 1);
assert.deepStrictEqual(mentions[0].codes, ["PC"]);

mentions = clock.extractTimeMentions("I was awake overnight, then ate at 8 AM.");
assert.strictEqual(mentions.length, 2);
assert.ok(mentions[0].codes.includes("GB"));
assert.ok(mentions[0].codes.includes("LR"));
assert.deepStrictEqual(mentions[1].codes, ["ST"]);

assert.deepStrictEqual(clock.extractTimeMentions("I did three exercises."), []);

console.log("organ-clock helpers: ok");
