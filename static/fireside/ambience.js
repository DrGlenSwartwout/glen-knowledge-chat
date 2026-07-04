import { nextGapMs, shouldDuck } from './governance.js';
import { emberBurst } from './spark.js';

export class Ambience {
  constructor(ambience, opts = {}) {
    this.amb = ambience || { bed: null, bed_volume: 0.18, oneshots: [] };
    this.isVoicePlaying = opts.isVoicePlaying || (() => false);
    this.sparkCtx = opts.sparkCtx || null;
    this.sparkXY = opts.sparkXY || [0, 0];
    this.muted = !!opts.muted;
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
    a.volume = this.muted ? 0 : o.volume;
    a.play().catch(() => {});
    this.loopEls.push({ el: a, volume: o.volume });
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
    const a = new Audio(o.file);
    a.volume = o.volume;
    a.play().catch(() => {});
    if (o.spark && this.sparkCtx) {
      if (this.cancelSpark) this.cancelSpark();
      this.cancelSpark = emberBurst(this.sparkCtx, this.sparkXY[0], this.sparkXY[1]);
    }
  }

  setMuted(m) {
    this.muted = !!m;
    if (this._bedRaf) { cancelAnimationFrame(this._bedRaf); this._bedRaf = null; }
    if (this.bedEl) this.bedEl.volume = this.muted ? 0 : this.amb.bed_volume;
    for (const L of this.loopEls) L.el.volume = this.muted ? 0 : L.volume;
  }

  stop() {
    this.timers.forEach(clearTimeout); this.timers = [];
    if (this._bedRaf) { cancelAnimationFrame(this._bedRaf); this._bedRaf = null; }
    if (this.bedEl) { this._fadeOutAndPause(this.bedEl, 1200); this.bedEl = null; }
    for (const L of this.loopEls) { try { L.el.pause(); } catch (e) {} }
    this.loopEls = [];
    if (this.cancelSpark) { this.cancelSpark(); this.cancelSpark = null; }
  }
}
