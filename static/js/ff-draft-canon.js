// static/js/ff-draft-canon.js
// The canonical-record signal on the console FF-match review card: the client's
// canonical attributes (conditions, terrain, systems, challenges, goals) shown
// as a distinct, labeled block SEPARATE from the scan-driven match items. It is
// NOT an .item, so collectItems()/publish never touch it, and it never reaches
// the client. Display-only, review-only.
var _FF_CANON_LABELS = {
  conditions: 'Conditions',
  terrain_concerns: 'Terrain concerns',
  body_systems: 'Body systems',
  challenges: 'Challenges',
  goals: 'Goals'
};
var _FF_CANON_ORDER = ['conditions', 'terrain_concerns', 'body_systems',
                       'challenges', 'goals'];

function _ffCanonEsc(s) {
  return (s == null ? '' : String(s)).replace(/[&<>"]/g, function (c) {
    return { '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;' }[c];
  });
}

function renderCanonBlock(canonical) {
  if (!canonical) return '';
  var lines = [];
  _FF_CANON_ORDER.forEach(function (f) {
    var v = canonical[f];
    var text = Array.isArray(v)
      ? v.map(function (x) { return String(x).trim(); }).filter(Boolean).join(', ')
      : String(v == null ? '' : v).trim();
    if (text) {
      lines.push('<div class="canon-line"><span class="canon-k">'
        + _ffCanonEsc(_FF_CANON_LABELS[f]) + '</span> '
        + _ffCanonEsc(text) + '</div>');
    }
  });
  if (!lines.length) return '';
  return '<div class="canon"><div class="canon-h">From the client’s records '
    + '(context for your review — not part of the match)</div>'
    + lines.join('') + '</div>';
}

if (typeof module !== 'undefined' && module.exports) {
  module.exports = { renderCanonBlock: renderCanonBlock };
}
