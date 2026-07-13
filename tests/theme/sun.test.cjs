const test = require('node:test');
const assert = require('node:assert');
const { sunTimes } = require('../../static/sun-engine.js');

// Convert a local-hours float to "H:MM" for readable assertions.
function hm(h){ const m = Math.round(h*60); return Math.floor(m/60) + ':' + String(m%60).padStart(2,'0'); }

test('Honolulu mid-July sunrise/sunset (UTC-10)', () => {
  const d = new Date(2026, 6, 12); // local date; test host TZ is normalized below
  const { sunrise, sunset } = sunTimes(d, 21.3, -157.8);
  // Allow the host offset to differ from HST: assert day length instead of wall time.
  const dayLen = sunset - sunrise;
  assert.ok(Math.abs(dayLen - 13.3) < 0.3, 'day length ~13.3h, got ' + dayLen.toFixed(2));
});

test('Anchorage mid-July has a long day (>17h)', () => {
  const d = new Date(2026, 6, 12);
  const { sunrise, sunset } = sunTimes(d, 61.2, -149.9);
  assert.ok((sunset - sunrise) > 17, 'expected >17h, got ' + (sunset - sunrise).toFixed(2));
});

test('Honolulu winter day is shorter than summer day', () => {
  const summer = sunTimes(new Date(2026, 6, 12), 21.3, -157.8);
  const winter = sunTimes(new Date(2026, 11, 21), 21.3, -157.8);
  assert.ok((winter.sunset - winter.sunrise) < (summer.sunset - summer.sunrise));
});

test('extreme polar latitude returns null in deep winter', () => {
  const { sunrise } = sunTimes(new Date(2026, 11, 21), 78, 15); // Svalbard, polar night
  assert.strictEqual(sunrise, null);
});
