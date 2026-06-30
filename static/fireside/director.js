import { classifyTyping, detectFamily, detectIntensity } from './heuristics.js';
import { selectClip } from './reaction-select.js';
import { canFireBackchannel, canInterject, canInterrupt } from './governance.js';

const FADE_MS = 250;

export class Director {
  constructor(manifest, opts = {}) {
    this.m = manifest;
    this.a = opts.videoA; this.b = opts.videoB;       // two stacked <video>
    this.onInterjectionAudio = opts.onInterjectionAudio || (() => {});
    this.front = this.a; this.back = this.b;
    this._fadeGen = 0;
    this.lastReactionId = null;
    this.lastBackchannelMs = null;
    this.state = 'idle';
  }

  _crossfadeTo(file, { loop = false } = {}) {
    if (!file) return;
    const inc = this.back;
    inc.onended = null;                         // drop any stale handler before reuse
    inc.src = file; inc.loop = loop; inc.currentTime = 0;
    const p = inc.play(); if (p && p.catch) p.catch(() => {});
    inc.style.transition = `opacity ${FADE_MS}ms`;
    inc.style.opacity = '1';
    const out = this.front;
    out.style.transition = `opacity ${FADE_MS}ms`;   // ensure the outgoing actually fades
    out.style.opacity = '0';
    const gen = ++this._fadeGen;
    setTimeout(() => {
      // Only pause if no newer crossfade has superseded this one (else `out` was
      // recycled as the active incoming video and must keep playing).
      if (gen === this._fadeGen) { try { out.pause(); } catch (e) {} }
    }, FADE_MS + 30);
    this.front = inc; this.back = out;
  }

  toResting() {
    this.state = 'resting';
    const pool = this.m.resting_loops;
    if (!pool.length) return;
    const file = pool[Math.floor(Math.random() * pool.length)];
    this._crossfadeTo(file, { loop: true });
  }

  _playReactionThenRest(clip) {
    if (!clip) return;
    this.lastReactionId = clip.id;
    this._crossfadeTo(clip.file, { loop: false });
    let fired = false;
    const back = () => { if (fired) return; fired = true; this.toResting(); };
    this.front.onended = back;
    setTimeout(back, Math.max(1200, (clip.duration_s || 2.5) * 1000 + 200)); // safety
  }

  onType(text, ctx) {
    const now = ctx.nowMs;
    if (!canFireBackchannel(now, this.lastBackchannelMs)) return;
    const c = classifyTyping(text);
    const wantAudible = c.intensity === 'high';
    let clip = c.gaze
      ? selectClip(this.m.reactions, { tier: 'gaze', gaze: c.gaze }, this.lastReactionId)
      : null;
    if (!clip) {
      clip = selectClip(this.m.reactions,
        { tier: 'backchannel', family: c.family, intensity: c.intensity,
          form: wantAudible ? undefined : 'silent' },
        this.lastReactionId);
    }
    if (clip) { this.lastBackchannelMs = now; this._playReactionThenRest(clip); }
  }

  maybeInterject(ctx) {
    if (!canInterject(ctx)) return null;
    const clip = selectClip(this.m.reactions, { tier: 'interjection' }, this.lastReactionId);
    if (clip) { this.lastReactionId = clip.id; this._playReactionThenRest(clip); this.onInterjectionAudio(clip); }
    return clip;
  }

  // Once per client: an off-screen hobbit voice calls "Glendalf"; he waves them off.
  // `ctx = { turn, idleMs }`; the seen-flag is persisted by the caller (localStorage
  // 'fireside_interruption_seen'). Returns the clip played, or null.
  maybeInterruption(ctx) {
    const seen = this._interruptionSeen ?? false;
    if (!canInterrupt({ seen, turn: ctx.turn, idleMs: ctx.idleMs })) return null;
    const clip = selectClip(this.m.reactions, { tier: 'interruption' }, null);
    if (!clip) return null;
    this._interruptionSeen = true;
    this.lastReactionId = clip.id;
    this._playReactionThenRest(clip);     // clip carries its own off-screen + reply audio
    this.onInterjectionAudio(clip);
    return clip;
  }

  onSubmit(text) {
    const fam = detectFamily(text), intensity = detectIntensity(text);
    const hero = selectClip(this.m.reactions, { tier: 'hero', family: fam, intensity }, this.lastReactionId);
    if (hero) { this.lastReactionId = hero.id; this._crossfadeTo(hero.file, { loop: false });
      this.front.onended = () => this._ponder(); }
    else this._ponder();
  }

  _ponder() {
    const clip = selectClip(this.m.reactions, { tier: 'ponder' }, this.lastReactionId);
    if (clip) this._crossfadeTo(clip.file, { loop: true });
    else if (this.m.pondering_loops.length)
      this._crossfadeTo(this.m.pondering_loops[0], { loop: true });
  }

  onReplyReady() { this.toSpeaking(); }

  toSpeaking() {
    // v1: return to the existing speaking loop (Phase B makes this emotion-matched).
    if (this.m.speaking_loop) this._crossfadeTo(this.m.speaking_loop, { loop: true });
    else this.toResting();
  }
}
