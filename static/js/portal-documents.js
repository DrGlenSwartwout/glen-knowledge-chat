// static/js/portal-documents.js
// My Records tile: the client's own uploaded medical records, plus the
// plain-language narrative once Glen has reviewed it.
// Consumes GET /api/portal/<token>/documents ->
//   {enabled, items:[{id,filename,uploaded_at,status,file_url,narrative_md}]}
// The payload deliberately carries no extracted attributes, facts, or labs.
function escapeHtmlDoc(s) {
  return String(s == null ? '' : s).replace(/[&<>"']/g, function (c) {
    return {'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[c];
  });
}

function renderDocuments(items) {
  if (!items || !items.length) return '';
  const rows = items.map(function (it) {
    const body = it.status === 'ready'
      ? '<p class="doc-narrative">' + escapeHtmlDoc(it.narrative_md) + '</p>'
      : '<p class="doc-pending">Received — under review</p>';
    // file_url is server-built (token + integer document id, see
    // api_portal_documents in app.py) and never carries client-supplied
    // content, so entity-escaping it is enough to stop attribute breakout.
    // escapeHtmlDoc does NOT validate the URL scheme — it would NOT stop a
    // `javascript:` href. If this value ever starts coming from user input,
    // add scheme validation before trusting it here.
    return '<li class="doc-item">' +
      '<a class="doc-file" href="' + escapeHtmlDoc(it.file_url) +
        '" target="_blank" rel="noopener">' + escapeHtmlDoc(it.filename) + '</a>' +
      body +
    '</li>';
  }).join('');
  return '<section class="portal-documents"><h2>My Records</h2>' +
         '<ul class="doc-list">' + rows + '</ul></section>';
}

if (typeof module !== 'undefined' && module.exports) {
  module.exports = { renderDocuments: renderDocuments };
}

// Browser: fetch + mount. Token is the last path segment of /portal/<token>.
if (typeof window !== 'undefined' && typeof document !== 'undefined') {
  document.addEventListener('DOMContentLoaded', function () {
    var mount = document.getElementById('portal-documents-mount');
    if (!mount) return;
    var m = location.pathname.match(/\/portal\/([^\/]+)/);
    if (!m) return;
    fetch('/api/portal/' + m[1] + '/documents')
      .then(function (r) { return r.ok ? r.json() : {enabled: false, items: []}; })
      .then(function (d) { mount.innerHTML = d.enabled ? renderDocuments(d.items) : ''; })
      .catch(function () {});
  });
}
