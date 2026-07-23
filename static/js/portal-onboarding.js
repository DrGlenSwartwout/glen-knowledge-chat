// static/js/portal-onboarding.js
// My Onboarding tile: 3-phase progress (Be read / Match remedies / Accelerate healing).
// Consumes GET /api/portal/<token>/onboarding -> {enabled, status:{phases:[{key,title,steps:[{key,label,done,href,soon?}]}], member}}.
function renderOnboarding(status) {
  if (!status || !status.phases || !status.phases.length) return '';
  const phases = status.phases.map(function (ph) {
    const steps = (ph.steps || []).map(function (st) {
      const mark = st.done === true ? '✓' : (st.done === false ? '○' : '•');
      var label = escapeHtml(st.label);
      if (st.soon) label += ' (coming soon)';
      var text = st.href
        ? '<a href="' + st.href + '">' + label + '</a>'
        : label;
      return '<li class="ob-step"><span class="ob-mark">' + mark + '</span> ' + text + '</li>';
    }).join('');
    return '<div class="ob-phase"><h3>' + escapeHtml(ph.title) + '</h3><ul class="ob-steps">' + steps + '</ul></div>';
  }).join('');
  return '<section class="portal-onboarding">' + phases + '</section>';
}
function escapeHtml(s) {
  return String(s == null ? '' : s).replace(/[&<>"']/g, function (c) {
    return {'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[c];
  });
}
if (typeof module !== 'undefined' && module.exports) { module.exports = { renderOnboarding: renderOnboarding }; }

// Browser: fetch + mount. Token is the last path segment of /portal/<token>.
// Hides (empties) the mount when the flag is off.
if (typeof window !== 'undefined' && typeof document !== 'undefined') {
  document.addEventListener('DOMContentLoaded', function () {
    var mount = document.getElementById('portal-onboarding-mount');
    if (!mount) return;
    var m = location.pathname.match(/\/portal\/([^\/]+)/);
    if (!m) return;
    fetch('/api/portal/' + m[1] + '/onboarding')
      .then(function (r) { return r.ok ? r.json() : {enabled:false, status:null}; })
      .then(function (d) { mount.innerHTML = d.enabled ? renderOnboarding(d.status) : ''; })
      .catch(function () {});
  });
}
