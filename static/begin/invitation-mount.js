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
      button.addEventListener('click', function (e) {
        e.preventDefault();
        inv.toggle();
      });

      inv.mount();
    })
    .catch(function () { /* manifest unavailable: leave the page as it was */ });
})();
