/* invitation-mount.js — wires invitation.js to the real landing-page DOM.
 * Intentionally thin and untested: all branching logic lives in invitation.js.
 */
import { pickWelcomeClip, Invitation } from './invitation.js';

(function () {
  var root = document.getElementById('fireside-invite');
  if (!root) return;

  fetch('/static/fireside/fireside-manifest.json')
    .then(function (r) { return r.ok ? r.json() : null; })
    .then(function (m) {
      var clip = pickWelcomeClip(m);
      if (!clip) return;                       // no clip: tile stays hidden

      var resting = (m && Array.isArray(m.resting_loops) && m.resting_loops.length)
        ? m.resting_loops[0] : null;

      var inv = new Invitation({
        video:       document.getElementById('fs-invite-video'),
        root:        root,
        choices:     document.getElementById('fs-invite-choices'),
        hint:        document.getElementById('fs-invite-hint'),
        frame:       document.getElementById('begin-chat'),
        origin:      window.location.origin,
        clip:        clip,
        restingClip: resting,
      });

      document.getElementById('fs-invite-tap')
        .addEventListener('click', function () { inv.tap(); });

      var stay = root.querySelector('.fs-invite-stay');
      if (stay) stay.addEventListener('click', function () { inv.dismiss(); });

      inv.start();
    })
    .catch(function () { /* manifest unavailable: leave the page as it was */ });
})();
