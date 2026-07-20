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
    this.audio.src = this.src;
    this.audio.currentTime = 0;
    this.audio.onended = () => { this.playing = false; this._label('idle'); };
    this.audio.play();
    this.playing = true;
    this._label('playing');
    this.notifyUnlock();
    return true;
  }

  stop() {
    if (!this.audio) return;
    this.audio.pause();
    this.playing = false;
    this._label('idle');
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
