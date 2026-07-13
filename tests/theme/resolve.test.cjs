const test = require('node:test');
const assert = require('node:assert');

// _resolve is pure and DOM-free; load the file under a minimal global shim.
global.window = {}; global.document = { documentElement: { }, addEventListener(){} };
require('../../static/theme-mode.js');
const R = global.window.RMTheme;

test('explicit modes pass through', () => {
  assert.strictEqual(R._resolve('light', {nowH:3, sunrise:6, sunset:19}), 'light');
  assert.strictEqual(R._resolve('dark',  {nowH:12, sunrise:6, sunset:19}), 'dark');
});
test('auto is light between sunrise and sunset', () => {
  assert.strictEqual(R._resolve('auto', {nowH:12, sunrise:6, sunset:19}), 'light');
});
test('auto is dark before sunrise and after sunset', () => {
  assert.strictEqual(R._resolve('auto', {nowH:5,  sunrise:6, sunset:19}), 'dark');
  assert.strictEqual(R._resolve('auto', {nowH:21, sunrise:6, sunset:19}), 'dark');
});
test('auto falls back to a 7-19 window when sun times are null', () => {
  assert.strictEqual(R._resolve('auto', {nowH:12, sunrise:null, sunset:null}), 'light');
  assert.strictEqual(R._resolve('auto', {nowH:2,  sunrise:null, sunset:null}), 'dark');
});
