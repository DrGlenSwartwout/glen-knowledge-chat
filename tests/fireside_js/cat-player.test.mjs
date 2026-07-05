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

// 6. wake() plays the react clip once, is ignored while reacting, then rests.
{
  const a = fakeVideo(), b = fakeVideo();
  const cp = new CatPlayer(a, b, Object.assign({}, cfg, { wake_cooldown_s: 0 }));  // no cooldown
  cp.start();                    // arrival react on a
  a.onended();                   // react -> rest (active now b)
  cp.wake();                     // on-demand wake
  const reactEl = cp.a;
  assert.equal(reactEl.src, '/react.mp4', 'wake plays the react clip');
  assert.equal(cp._reacting, true, 'reacting flag set during wake');
  const playsBefore = a.plays.length + b.plays.length;
  cp.wake();                     // ignored while already reacting
  assert.equal(a.plays.length + b.plays.length, playsBefore, 'no extra playback while reacting');
  reactEl.onended();             // wake clip finishes
  assert.equal(cp._reacting, false, 'reacting cleared after the wake clip');
  assert.equal(cp.a.src, '/rest.mp4', 'settles back to rest after wake');
  cp.stop();
}

// 7. wake() respects the cooldown: a second wake within the window is ignored.
{
  const a = fakeVideo(), b = fakeVideo();
  const cp = new CatPlayer(a, b, Object.assign({}, cfg, { wake_cooldown_s: 999 }));
  cp.start(); a.onended();       // enter rest; arrival set _lastWake
  cp.wake();                     // within 999s of arrival -> cooldown blocks it
  assert.equal(cp._reacting, false, 'wake blocked by cooldown after arrival');
  cp.stop();
}

// 8. crossfade: a clip CHANGE gets an opacity transition; a same-clip re-loop cuts.
{
  const a = fakeVideo(), b = fakeVideo();
  const cp = new CatPlayer(a, b, { rest: '/rest.mp4', idle: ['/gust.mp4'], idle_min_s: 35, idle_max_s: 95, crossfade: 700 });
  cp.start();                                   // a plays rest (no react clip)
  // Same-clip re-loop (rest -> rest): hard cut, transition stays 'none'.
  cp.a.onended();
  assert.equal(cp.a.style.transition, 'none', 'same-clip re-loop hard-cuts (transition none)');
  // Clip change (rest -> gust): crossfade transition applied to the swapped element.
  cp.queued = '/gust.mp4';
  cp.a.onended();                               // rest boundary dispatches the queued gust
  assert.equal(cp.a.src, '/gust.mp4', 'gust plays on clip change');
  assert.equal(cp.a.style.transition, 'opacity 700ms linear', 'clip change crossfades');
  cp.stop();
}

// 9. onClip fires with (url, isRest): true for the rest clip, false for a gust clip.
{
  const a = fakeVideo(), b = fakeVideo();
  const seen = [];
  const cp = new CatPlayer(a, b, {
    rest: '/rest.mp4', idle: ['/gust.mp4'], idle_min_s: 35, idle_max_s: 95,
    crossfade: 700, onClip: (url, isRest) => seen.push([url, isRest]),
  });
  cp.start();
  cp.a.onended();                               // rest re-loop -> onClip(rest, true)
  cp.queued = '/gust.mp4';
  cp.a.onended();                               // gust -> onClip(gust, false)
  assert.deepEqual(seen.find(x => x[0] === '/rest.mp4'), ['/rest.mp4', true], 'rest clip reports isRest=true');
  assert.deepEqual(seen.find(x => x[0] === '/gust.mp4'), ['/gust.mp4', false], 'gust clip reports isRest=false');
  cp.stop();
}

console.log('cat-player.test.mjs: all assertions passed');
