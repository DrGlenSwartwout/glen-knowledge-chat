/* Shared affiliate-referral capture for the funnel.
   Runs on every funnel page. Captures the referring affiliate slug on arrival
   and persists it in the rm_ref cookie (90 days) + localStorage, so it carries
   through the whole journey and any later conversion. Defines a global getRef()
   that the page scripts use when threading the slug onto outbound links and the
   /begin/unlock POST body. */
(function captureRef() {
  try {
    var q = new URLSearchParams(location.search);
    var VALID = /^[A-Za-z0-9_-]{1,64}$/;
    // utm_source is a generic campaign field, so only trust it as an affiliate
    // slug when utm_medium explicitly marks the link as a referral. This keeps
    // utm_source=facebook / utm_source=newsletter from being misread as an affiliate.
    var AFF_MEDIUMS = { affiliate: 1, referral: 1, concierge: 1 };
    // Precedence: ?ref= (canonical) → ?aff=/?a= → ?utm_source= (medium-gated).
    var ref = (q.get('ref') || '').trim();
    if (!ref) ref = (q.get('aff') || q.get('a') || '').trim();
    if (!ref) {
      var src = (q.get('utm_source') || '').trim();
      var med = (q.get('utm_medium') || '').trim().toLowerCase();
      if (src && AFF_MEDIUMS[med]) ref = src;
    }
    if (ref && VALID.test(ref)) {
      document.cookie = 'rm_ref=' + encodeURIComponent(ref) +
                        ';path=/;max-age=' + (90 * 24 * 3600) + ';SameSite=Lax';
      localStorage.setItem('rm_ref', ref);
    }
  } catch (_) {}
})();

function getRef() {
  try {
    var m = document.cookie.match(/(?:^|;\s*)rm_ref=([^;]+)/);
    if (m) return decodeURIComponent(m[1]);
    return localStorage.getItem('rm_ref') || '';
  } catch (_) { return ''; }
}
