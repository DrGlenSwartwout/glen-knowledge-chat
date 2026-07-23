// static/js/portal-library.js
// My Library tile: read (PDF) or listen (audio) the client's granted Starters.
// Consumes GET /api/portal/<token>/library -> {enabled, items:[{slug,title,pdf_url,audio_url,granted_at}]}.
function renderLibrary(items) {
  if (!items || !items.length) return '';
  const rows = items.map(function (it) {
    return '<li class="lib-item">' +
      '<span class="lib-title">' + escapeHtml(it.title) + '</span>' +
      '<a class="lib-read" href="' + it.pdf_url + '" target="_blank" rel="noopener">Read</a>' +
      '<audio class="lib-listen" controls preload="none" src="' + it.audio_url + '"></audio>' +
    '</li>';
  }).join('');
  return '<section class="portal-library"><h2>My Library</h2><ul class="lib-list">' + rows + '</ul></section>';
}
function escapeHtml(s) {
  return String(s == null ? '' : s).replace(/[&<>"']/g, function (c) {
    return {'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[c];
  });
}
if (typeof module !== 'undefined' && module.exports) { module.exports = { renderLibrary: renderLibrary }; }

// Browser: fetch + mount. Token is the last path segment of /portal/<token>.
// Hides (empties) the mount when the flag is off or there are no granted items.
if (typeof window !== 'undefined' && typeof document !== 'undefined') {
  document.addEventListener('DOMContentLoaded', function () {
    var mount = document.getElementById('portal-library-mount');
    if (!mount) return;
    var m = location.pathname.match(/\/portal\/([^\/]+)/);
    if (!m) return;
    fetch('/api/portal/' + m[1] + '/library')
      .then(function (r) { return r.ok ? r.json() : {enabled:false, items:[]}; })
      .then(function (d) { mount.innerHTML = d.enabled ? renderLibrary(d.items) : ''; })
      .catch(function () {});
  });
}
