/* invitation.js — the fireside welcome tile on the landing page.
 *
 * A small muted clip autoplays beside the chat panel. Tapping it unmutes and
 * plays Glendalf's invitation; that tap is also the browser gesture that
 * unlocks audio, so it is forwarded to the chat iframe, which from then on
 * speaks its replies aloud instead of waiting for a Listen click.
 *
 * Every method is a safe no-op when its dependencies are missing, mirroring
 * the Director's degradation contract: a failed manifest fetch must leave the
 * page exactly as it was.
 */

export const UNLOCK_MSG = 'begin:audio-unlocked';

export function pickWelcomeClip(m, rand = Math.random) {
  if (!m) return null;
  const list = Array.isArray(m.intro_welcomes) && m.intro_welcomes.length ? m.intro_welcomes : [];
  if (list.length) return list[Math.floor(rand() * list.length)] || list[0];
  return m.intro_welcome || m.intro_video || null;
}

export class Invitation {
  constructor(opts = {}) {
    this.video       = opts.video || null;
    this.root        = opts.root || null;
    this.choices     = opts.choices || null;
    this.hint        = opts.hint || null;
    this.frame       = opts.frame || null;
    this.origin      = opts.origin || '*';
    this.clip        = opts.clip || null;
    this.restingClip = opts.restingClip || null;
    this.unlocked    = false;
  }

  start() {
    if (!this.clip || !this.video) return false;
    this.video.src = this.clip;
    this.video.muted = true;
    this.video.loop = true;
    this.video.play();
    if (this.root) this.root.classList.remove('hidden');
    return true;
  }

  tap() {
    if (!this.clip || !this.video) return false;
    this.video.muted = false;
    this.video.loop = false;
    this.video.currentTime = 0;
    this.video.onended = () => this.onEnded();
    this.video.play();
    if (this.hint) this.hint.classList.add('hidden');
    this.notifyUnlock();
    return true;
  }

  notifyUnlock() {
    if (this.unlocked) return false;
    this.unlocked = true;
    if (this.frame && this.frame.contentWindow) {
      this.frame.contentWindow.postMessage({ type: UNLOCK_MSG }, this.origin);
    }
    return true;
  }

  onEnded() {
    if (this.choices) this.choices.classList.remove('hidden');
    if (this.restingClip && this.video) {
      this.video.src = this.restingClip;
      this.video.muted = true;
      this.video.loop = true;
      this.video.currentTime = 0;
      this.video.play();
    }
  }

  dismiss() {
    if (this.root) this.root.classList.add('hidden');
  }
}
