/* invitation.js — the spoken invitation behind the hero avatar's speaker button.
 *
 * The landing page already shows a muted, looping Glendalf video that links to
 * /begin/fireside. That clip carries no audio track, so this module plays a
 * separate voice-over on demand. The tap that starts it is also the browser
 * gesture that permits audio, so it is forwarded to the chat iframe, which from
 * then on speaks its replies instead of waiting for a Listen click.
 *
 * Every method is a safe no-op when its dependencies are missing, mirroring the
 * Director's degradation contract: a failed manifest fetch must leave the page
 * exactly as it was.
 */

export const UNLOCK_MSG = 'begin:audio-unlocked';

const ICON_IDLE = '🔈';
const ICON_PLAYING = '⏹';
const LABEL_IDLE = "Hear Dr. Glen's invitation";
const LABEL_PLAYING = 'Stop the invitation';

export function pickInvitationAudio(m, rand = Math.random) {
  if (!m) return null;
  const list = Array.isArray(m.intro_welcome_audios) && m.intro_welcome_audios.length
    ? m.intro_welcome_audios : [];
  if (list.length) return list[Math.floor(rand() * list.length)] || list[0];
  return m.intro_welcome_audio || null;
}

export class Invitation {
  constructor(opts = {}) {
    this.audio    = opts.audio || null;
    this.button   = opts.button || null;
    this.frame    = opts.frame || null;
    this.origin   = opts.origin || (typeof window !== 'undefined' && window.location ? window.location.origin : '*');
    this.src      = opts.src || null;
    this.unlocked = false;
    this.playing  = false;
  }

  _label(state) {
    if (!this.button) return;
    const on = state === 'playing';
    this.button.innerHTML = on ? ICON_PLAYING : ICON_IDLE;
    this.button.setAttribute('aria-label', on ? LABEL_PLAYING : LABEL_IDLE);
  }

  mount() {
    if (!this.src || !this.button) return false;
    this._label('idle');
    this.button.classList.remove('hidden');
    return true;
  }

  play() {
    if (!this.src || !this.audio) return false;
    // One voice at a time. tts-output.js arbitrates its OWN players through a
    // single `active` slot, but this Audio object is outside that system, so
    // without this the invitation and a spoken reply talk over each other.
    if (typeof window !== 'undefined' && window.TTS && window.TTS.stop) window.TTS.stop();
    this.audio.src = this.src;
    this.audio.currentTime = 0;
    this.audio.onended = () => this._free();
    // Mark playing BEFORE starting: a refused play() can reject synchronously,
    // and _free() must not be overwritten by a later `playing = true`.
    this.playing = true;
    this._label('playing');
    clearTimeout(this._guard);
    this._guard = setTimeout(() => { if (this.playing) this._free(); }, 20000);
    // Never strand a waiting reply. If playback is refused (autoplay policy) or
    // stalls, `ended` never fires — so release the channel on rejection, and
    // again on the ceiling above, well past this clip's ~4.3s.
    const p = this.audio.play();
    if (p && p.catch) p.catch(() => this._free());
    this.notifyUnlock();
    return true;
  }

  stop() {
    if (!this.audio) return;
    this.audio.pause();
    this._free();
  }

  // Run cb once the invitation is no longer holding the audio channel — either
  // because it finished or because it was stopped. Immediate if it isn't
  // playing. Only the most recent caller is kept: if two replies land during
  // one welcome, speaking both afterwards would overlap, which is the very
  // thing this exists to prevent.
  whenFree(cb) {
    if (typeof cb !== 'function') return false;
    if (!this.playing) { cb(); return true; }
    this._pending = cb;
    return true;
  }

  _free() {
    clearTimeout(this._guard);
    this.playing = false;
    this._label('idle');
    const cb = this._pending;
    this._pending = null;
    if (cb) cb();
  }

  toggle() {
    if (this.playing) { this.stop(); return false; }
    return this.play();
  }

  notifyUnlock() {
    // Always post when a frame is available: the #begin-chat iframe may still
    // be loading in parallel with the manifest fetch, so an early tap's post
    // can be silently discarded before the iframe's listener is registered.
    // The receiver is idempotent (it just sets a boolean true), so re-sending
    // is harmless — but `unlocked` still records state so callers can observe
    // the first-time state change.
    const first = !this.unlocked;
    this.unlocked = true;
    if (this.frame && this.frame.contentWindow) {
      this.frame.contentWindow.postMessage({ type: UNLOCK_MSG }, this.origin);
    }
    return first;
  }
}
