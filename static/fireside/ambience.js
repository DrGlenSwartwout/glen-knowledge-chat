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
    this.cancelSpark = null;
  }

  start() {
    if (this.amb.bed) {
      this.bedEl = new Audio(this.amb.bed);
      this.bedEl.loop = true;
      this.bedEl.volume = this.muted ? 0 : this.amb.bed_volume;
      this.bedEl.play().catch(() => {});
    }
    for (const o of this.amb.oneshots) this._schedule(o);
  }

  _schedule(o) {
    const t = setTimeout(() => {
      if (!this.muted && !shouldDuck(this.isVoicePlaying())) this._play(o);
      this._schedule(o); // always reschedule, ducked or not
    }, nextGapMs(o));
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
    if (this.bedEl) this.bedEl.volume = this.muted ? 0 : this.amb.bed_volume;
  }

  stop() {
    this.timers.forEach(clearTimeout); this.timers = [];
    if (this.bedEl) { this.bedEl.pause(); this.bedEl = null; }
    if (this.cancelSpark) { this.cancelSpark(); this.cancelSpark = null; }
  }
}
