// Smoke tests for the Fire resident-cat runtime (cat-player.js).
// CatPlayer is a DOM/timing component; here we drive it with minimal fake
// <video> elements that fire 'canplay' synchronously, so the seamless-swap
// flow proceeds deterministically and we can assert the state transitions.
import { CatPlayer } from '../../static/fireside/cat-player.js';
import assert from 'node:assert';

function fakeVideo() {
  const v = {
    _src: '', style: {}, readyState: 2, currentTime: 0, onended: null,
    muted: false, loop: false, playsInline: false, poster: '',
    plays: [], paused: true,
    play() { this.plays.push(this._src); this.paused = false; return { catch() {} }; },
    pause() { this.paused = true; },
    load() {},
    addEventListener(ev, cb) { if (ev === 'canplay') cb(); },     // synchronous ready
    removeEventListener() {},
    getAttribute(k) { return k === 'src' ? this._src : null; },
  };
  Object.defineProperty(v, 'src', { get() { return this._src; }, set(x) { this._src = x; } });
  return v;
}

const cfg = {
  rest: '/rest.mp4',
  idle: ['/idle-a.mp4', '/idle-b.mp4', '/idle-c.mp4'],
  react: '/react.mp4',
  idle_min_s: 120, idle_max_s: 300,
};

// 1. Construction clamps timing and mutes/loop-disables both elements.
{
  const a = fakeVideo(), b = fakeVideo();
  const cp = new CatPlayer(a, b, { rest: '/r.mp4', idle_min_s: 1, idle_max_s: 0 });
  assert.equal(cp.min, 5000, 'idle_min floored at 5s');
  assert.ok(cp.max >= cp.min, 'idle_max never below idle_min');
  assert.equal(a.muted && b.muted, true, 'both elements muted');
  assert.equal(a.loop === false && b.loop === false, true, 'both loop disabled (manual looping)');
}

// 2. start() plays the react clip first, on the active element.
{
  const a = fakeVideo(), b = fakeVideo();
  const cp = new CatPlayer(a, b, cfg);
  cp.start();
  assert.equal(a.src, '/react.mp4', 'react plays first');
  assert.deepEqual(a.plays, ['/react.mp4'], 'react was play()ed');
  assert.equal(typeof a.onended, 'function', 'react end handler wired');
  cp.stop();
}

// 3. After react ends, it enters the rest loop (via the buffer, seamless swap).
{
  const a = fakeVideo(), b = fakeVideo();
  const cp = new CatPlayer(a, b, cfg);
  cp.start();
  a.onended();                                  // react finishes
  assert.equal(b.src, '/rest.mp4', 'rest loaded on the buffer element');
  assert.deepEqual(b.plays, ['/rest.mp4'], 'rest play()ed after react');
  assert.equal(cp.a, b, 'roles swapped: buffer is now active');
  cp.stop();
}

// 4. A queued idle plays once at the next rest boundary, then returns to rest.
{
  const a = fakeVideo(), b = fakeVideo();
  const cp = new CatPlayer(a, b, cfg);
  cp.start();
  a.onended();                                  // react -> rest (active now = b)
  cp.queued = '/idle-b.mp4';                     // pretend the idle timer fired
  const restEl = cp.a;                           // element currently playing rest
  restEl.onended();                              // rest boundary reached
  const idleEl = cp.a;                           // should be playing the idle now
  assert.equal(idleEl.src, '/idle-b.mp4', 'queued idle plays at boundary');
  assert.equal(cp.queued, null, 'queue cleared after dispatch');
  idleEl.onended();                              // idle finishes
  assert.equal(cp.a.src, '/rest.mp4', 'returns to rest after the idle');
  cp.stop();
}

// 5. stop() halts everything and clears the timer.
{
  const a = fakeVideo(), b = fakeVideo();
  const cp = new CatPlayer(a, b, cfg);
  cp.start();
  const playsBefore = a.plays.length + b.plays.length;
  cp.stop();
  assert.equal(cp.stopped, true, 'marked stopped');
  assert.equal(cp.queued, null, 'queue cleared on stop');
  if (a.onended) a.onended();                      // late events are no-ops
  if (b.onended) b.onended();
  assert.equal(a.plays.length + b.plays.length, playsBefore, 'no new playback after stop');
}

console.log('cat-player.test.mjs: all assertions passed');
