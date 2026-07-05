import { nextGapMs, shouldDuck } from './governance.js';
import { emberBurst } from './spark.js';

// A one-shot may carry a single `file` OR a `files` array of interchangeable
// variants (e.g. the cat's three routines) — one is picked at random each fire.
export function pickSrc(o, rng = Math.random) {
  if (o && Array.isArray(o.files) && o.files.length) {
    return o.files[Math.floor(rng() * o.files.length)];
  }
  return (o && o.file) || null;
}

export class Ambience {
  constructor(ambience, opts = {}) {
    this.amb = ambience || { bed: null, bed_volume: 0.18, oneshots: [] };
    this.isVoicePlaying = opts.isVoicePlaying || (() => false);
    this.sparkCtx = opts.sparkCtx || null;
    this.sparkXY = opts.sparkXY || [0, 0];
    this.muted = !!opts.muted;
    // opt-in voice ducking of the BED + continuous loops (one-shots already duck).
    // duck = the fraction of normal volume to drop to while a voice is playing
    // (e.g. 0.4 = beds fall to 40% under Glendalf's report audio, swell back after).
    this.duck = (typeof opts.duck === 'number' && opts.duck > 0 && opts.duck < 1) ? opts.duck : null;
    this._ducked = false;
    this._monTimer = null;
    this.bedEl = null;
    this.timers = [];
    this.loopEls = [];
    this.cancelSpark = null;
    this._bedRaf = null;
  }

  start() {
    if (this._started) return;
    this._started = true;
    if (this.amb.bed) {
      this.bedEl = new Audio(this.amb.bed);
      this.bedEl.loop = true;
      this.bedEl.volume = 0;
      this.bedEl.play().catch(() => {});
      // gentle taper-in: the fire bed loops, so it can't carry a baked fade
      // (that would dip every loop) — ease it up in JS instead.
      this._rampBed(this.muted ? 0 : this.amb.bed_volume, 2000);
    }
    for (const o of this.amb.oneshots) {
      if (o.loop) this._startLoop(o);       // continuous soft layer (fills dead time)
      else this._schedule(o, true);         // random one-shot (first gap may be overridden)
    }
    // poll the voice source; duck/unduck the bed + loops when it starts/stops
    if (this.duck != null) this._monTimer = setInterval(() => this._checkDuck(), 250);
  }

  _levelFactor() {
    return this.muted ? 0 : (this._ducked && this.duck != null ? this.duck : 1);
  }

  // ramp bed + continuous loops to their current target (mute * duck) over ms
  _applyLevels(ms = 500) {
    const f = this._levelFactor();
    this._rampBed(this.amb.bed_volume * f, ms);
    for (const L of this.loopEls) if (!L.swell) this._rampEl(L.el, L.volume * f, ms);
  }

  _rampEl(el, target, ms) {
    if (!el) return;
    if (el._raf) cancelAnimationFrame(el._raf);
    const from = el.volume, t0 = performance.now();
    const tick = (now) => {
      const k = ms <= 0 ? 1 : Math.min(1, (now - t0) / ms);
      el.volume = from + (target - from) * k;
      if (k < 1) el._raf = requestAnimationFrame(tick); else el._raf = null;
    };
    el._raf = requestAnimationFrame(tick);
  }

  _checkDuck() {
    const voice = !!this.isVoicePlaying();
    if (voice !== this._ducked) {
      this._ducked = voice;
      // taper: duck in fairly quickly as he starts, swell back gently as he stops
      this._applyLevels(voice ? 700 : 1400);
    }
  }

  _rampBed(target, ms) {
    if (!this.bedEl) return;
    if (this._bedRaf) cancelAnimationFrame(this._bedRaf);
    const el = this.bedEl;
    const from = el.volume;
    const t0 = performance.now();
    const tick = (now) => {
      const k = ms <= 0 ? 1 : Math.min(1, (now - t0) / ms);
      el.volume = from + (target - from) * k;
      if (k < 1 && this.bedEl === el) this._bedRaf = requestAnimationFrame(tick);
      else this._bedRaf = null;
    };
    this._bedRaf = requestAnimationFrame(tick);
  }

  _fadeOutAndPause(el, ms) {
    const from = el.volume;
    const t0 = performance.now();
    const tick = (now) => {
      const k = ms <= 0 ? 1 : Math.min(1, (now - t0) / ms);
      el.volume = from * (1 - k);
      if (k < 1) requestAnimationFrame(tick);
      else { try { el.pause(); } catch (e) {} }
    };
    requestAnimationFrame(tick);
  }

  _startLoop(o) {
    const a = new Audio(o.file);
    a.loop = true;
    a.volume = this.muted ? 0 : (o.swell ? o.swell.min : o.volume);
    a.play().catch(() => {});
    const L = { el: a, volume: o.volume, swell: o.swell || null };
    this.loopEls.push(L);
    if (L.swell) this._startSwell(L);
  }

  // Continuous amplitude LFO for a loop layer: eases its volume between swell.min
  // and swell.max along a sine over swell.period_s, so e.g. the purr breathes (ebbs
  // and flows) instead of droning. Re-reads the mute/duck level each frame so it
  // still ducks under Dr. Glen's voice.
  _startSwell(L) {
    const sw = L.swell;
    const min = (typeof sw.min === 'number') ? sw.min : 0;
    const max = (typeof sw.max === 'number') ? sw.max : L.volume;
    const period = (Number(sw.period_s) > 0 ? Number(sw.period_s) : 14) * 1000;
    const t0 = performance.now();
    const tick = (now) => {
      if (!L.el) return;
      const phase = ((now - t0) % period) / period;
      const eased = 0.5 - 0.5 * Math.cos(2 * Math.PI * phase);   // 0 -> 1 -> 0 sine
      const v = (min + (max - min) * eased) * this._levelFactor();
      try { L.el.volume = Math.max(0, Math.min(1, v)); } catch (e) {}
      L._swellRaf = requestAnimationFrame(tick);
    };
    L._swellRaf = requestAnimationFrame(tick);
  }

  _schedule(o, isFirst = false) {
    // first_gap_s lets a signature one-shot (e.g. the Metal singing bowl) sound
    // shortly after start instead of waiting a full random gap; later fires cycle
    // on the normal min/max gap.
    const gap = (isFirst && typeof o.first_gap_s === 'number')
      ? Math.max(0, o.first_gap_s) * 1000
      : nextGapMs(o);
    const t = setTimeout(() => {
      if (!this.muted && !shouldDuck(this.isVoicePlaying())) this._play(o);
      this._schedule(o); // always reschedule, ducked or not
    }, gap);
    this.timers.push(t);
  }

  _play(o) {
    const src = pickSrc(o);
    if (!src) return;
    const a = new Audio(src);
    a.volume = o.volume;
    a.play().catch(() => {});
    if (o.spark && this.sparkCtx) {
      if (this.cancelSpark) this.cancelSpark();
      this.cancelSpark = emberBurst(this.sparkCtx, this.sparkXY[0], this.sparkXY[1]);
    }
  }

  setMuted(m) {
    this.muted = !!m;
    this._applyLevels(150);           // fast ramp; composes mute with the duck state
  }

  stop() {
    if (this._monTimer) { clearInterval(this._monTimer); this._monTimer = null; }
    this.timers.forEach(clearTimeout); this.timers = [];
    if (this._bedRaf) { cancelAnimationFrame(this._bedRaf); this._bedRaf = null; }
    if (this.bedEl) { this._fadeOutAndPause(this.bedEl, 1200); this.bedEl = null; }
    for (const L of this.loopEls) { try { if (L._swellRaf) cancelAnimationFrame(L._swellRaf); L.el.pause(); } catch (e) {} }
    this.loopEls = [];
    if (this.cancelSpark) { this.cancelSpark(); this.cancelSpark = null; }
  }
}
