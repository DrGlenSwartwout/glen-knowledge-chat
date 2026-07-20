/* invitation-mount.js — wires invitation.js to the hero avatar's speaker button.
 * Intentionally thin and untested: all branching logic lives in invitation.js.
 */
import { pickInvitationAudio, Invitation } from './invitation.js';

(function () {
  var button = document.getElementById('avatar-speaker');
  if (!button) return;

  fetch('/static/fireside/fireside-manifest.json')
    .then(function (r) { return r.ok ? r.json() : null; })
    .then(function (m) {
      var src = pickInvitationAudio(m);
      if (!src) return;                        // no voice-over: button stays hidden

      var inv = new Invitation({
        audio:  new Audio(),
        button: button,
        frame:  document.getElementById('begin-chat'),
        origin: window.location.origin,
        src:    src,
      });

      // The speaker is a sibling of the fireside anchor, not nested inside it,
      // so the click cannot reach the anchor's engagement handler. preventDefault
      // is kept only as a harmless guard against future re-nesting.
      // Exposed so the hero chat can silence the invitation before speaking a
      // reply — the two use different audio players and would otherwise overlap.
      window.__invitation = inv;

      button.addEventListener('click', function (e) {
        e.preventDefault();
        inv.toggle();
        // The hero chat renders on THIS page, so it reads the flag locally. The
        // postMessage inside notifyUnlock only reaches the #begin-chat iframe,
        // which is a different, further-down conversation.
        if (inv.unlocked) window.__audioUnlocked = true;
      });

      var chat = document.getElementById('begin-chat');
      if (chat) chat.addEventListener('load', function () {
        if (inv.unlocked) inv.notifyUnlock();   // re-arm: an early tap may have been discarded
      });

      inv.mount();
    })
    .catch(function () { /* manifest unavailable: leave the page as it was */ });
})();
