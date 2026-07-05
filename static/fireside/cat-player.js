// cat-player.js — Fire backdrop resident-cat runtime (Phase 2).
//
// The Fire cat has three states baked into full fire-scene clips:
//   • REST  — the still cat by the fire (the element's normal loop video)
//   • IDLE  — short action clips (ear flick, paw twitch, nose twitch) played
//             rarely at random, then back to rest
//   • REACT — a wake-and-look-up clip played once when the member arrives
// Every clip starts AND ends on the identical boundary frame (fire-frame-0 +
// cat at rest), so switching at a clip's 'ended' event is seamless.
//
// We double-buffer two <video> elements. The outgoing one holds its last (rest)
// frame — pixel-identical to the incoming clip's first frame — so there is no
// flash even if the incoming clip needs a beat to start. Everything is muted;
// the purr and fire sounds come from the separate ambience layer.
//
// No dependency on the DOM structure beyond the two <video> els handed in, so it
// stays as testable as the other fireside modules.

export class CatPlayer {
  constructor(front, back, cfg) {
    this.front = front; this.back = back;         // stable DOM refs
    this.a = front; this.b = back;                // active / buffer (roles swap)
    this.rest = cfg.rest;
    this.idle = Array.isArray(cfg.idle) ? cfg.idle.slice() : [];
    this.react = cfg.react || null;
    const min = Math.max(5, Number(cfg.idle_min_s) || 120);
    const max = Math.max(min, Number(cfg.idle_max_s) || 300);
    this.min = min * 1000; this.max = max * 1000;
    this.queued = null; this.timer = null; this.stopped = true;
    for (const v of [front, back]) {
      v.muted = true; v.loop = false; v.playsInline = true;
      v.style.transition = 'opacity .12s linear';
      if (cfg.poster) v.poster = cfg.poster;
    }
  }

  _play(el) { try { const p = el.play(); if (p && p.catch) p.catch(() => {}); } catch (e) {} }

  // Play `url` on the buffer element, then reveal it over the (still-showing) active.
  _go(url, onEnded) {
    if (this.stopped) return;
    const buf = this.b;
    const begin = () => {
      buf.removeEventListener('canplay', begin);
      if (this.stopped) return;
      try { buf.currentTime = 0; } catch (e) {}
      this._play(buf);
      buf.style.opacity = '1';
      this.a.style.opacity = '0';
      const t = this.a; this.a = buf; this.b = t;          // swap roles
      this.a.onended = () => { if (!this.stopped) onEnded(); };
    };
    const cur = buf.getAttribute('src') || '';
    if (cur.indexOf(url) !== -1 && buf.readyState >= 2) begin();
    else { buf.src = url; buf.addEventListener('canplay', begin, { once: true }); try { buf.load(); } catch (e) {} }
  }

  _afterRest() {
    if (this.stopped) return;
    if (this.queued && this.idle.length) {
      const u = this.queued; this.queued = null;
      this._go(u, () => { this._schedule(); this._rest(); });   // one idle, then back to rest
    } else {
      this._rest();                                             // keep resting
    }
  }

  _rest() { this._go(this.rest, () => this._afterRest()); }

  _schedule() {
    clearTimeout(this.timer);
    if (!this.idle.length) return;
    const d = this.min + Math.random() * (this.max - this.min);
    this.timer = setTimeout(() => {
      this.queued = this.idle[Math.floor(Math.random() * this.idle.length)];
    }, d);
  }

  start() {
    this.stopped = false;
    this.a = this.front; this.b = this.back;
    this.a.style.opacity = '1'; this.b.style.opacity = '0';
    const enter = () => { this._schedule(); this._rest(); };
    if (this.react) {
      this.a.src = this.react;
      this.a.onended = () => { if (!this.stopped) enter(); };
      try { this.a.currentTime = 0; } catch (e) {}
      this._play(this.a);
    } else {
      this.a.src = this.rest;
      this.a.onended = () => { if (!this.stopped) this._afterRest(); };
      this._play(this.a);
      this._schedule();
    }
  }

  stop() {
    this.stopped = true;
    clearTimeout(this.timer);
    this.queued = null;
    for (const v of [this.front, this.back]) {
      try { v.onended = null; v.pause(); } catch (e) {}
      v.style.transition = '';
    }
    this.back.style.opacity = '0';
  }
}
