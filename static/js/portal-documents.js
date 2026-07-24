// static/js/portal-documents.js
// My Records tile: the client's own uploaded medical records, plus the
// plain-language narrative once Glen has reviewed it, plus the upload
// control that lets the client add a new one.
// Consumes GET /api/portal/<token>/documents ->
//   {enabled, items:[{id,filename,uploaded_at,status,file_url,narrative_md}]}
// The payload deliberately carries no extracted attributes, facts, or labs.
// Upload POSTs to /api/portal/<token>/documents, multipart, field name "file".
function escapeHtmlDoc(s) {
  return String(s == null ? '' : s).replace(/[&<>"']/g, function (c) {
    return {'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[c];
  });
}

// The upload control. Rendered inside the tile even when there are no
// documents yet -- otherwise a client can never make a first upload.
function renderDocUploadHtml() {
  return '<div class="doc-upload">' +
    '<label for="doc-upload-input">Upload a record</label> ' +
    '<input type="file" id="doc-upload-input" class="doc-upload-input">' +
    '<button type="button" id="doc-upload-btn" class="doc-upload-btn btn">Upload</button>' +
    '<span id="doc-upload-status" class="doc-upload-status" role="status"></span>' +
  '</div>';
}

function renderDocuments(items) {
  const rows = (items || []).map(function (it) {
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
  return '<section class="portal-documents card"><h2>My Records</h2>' +
         renderDocUploadHtml() +
         (rows ? '<ul class="doc-list">' + rows + '</ul>'
               : '<p class="doc-empty">No records yet.</p>') +
         '</section>';
}

if (typeof module !== 'undefined' && module.exports) {
  module.exports = { renderDocuments: renderDocuments, renderDocUploadHtml: renderDocUploadHtml };
}

// Browser: fetch + mount, plus wire the upload control. Token is the last
// path segment of /portal/<token>. The whole tile stays hidden (mount
// emptied) whenever the API's `enabled` is false -- unchanged from before.
if (typeof window !== 'undefined' && typeof document !== 'undefined') {
  document.addEventListener('DOMContentLoaded', function () {
    var mount = document.getElementById('portal-documents-mount');
    if (!mount) return;
    var m = location.pathname.match(/\/portal\/([^\/]+)/);
    if (!m) return;
    var tok = m[1];

    function wireUpload() {
      var btn = document.getElementById('doc-upload-btn');
      var input = document.getElementById('doc-upload-input');
      var status = document.getElementById('doc-upload-status');
      if (!btn || !input) return;
      btn.addEventListener('click', function () {
        var f = input.files && input.files[0];
        if (!f) { if (status) status.textContent = 'Choose a file first.'; return; }
        btn.disabled = true;
        if (status) status.textContent = 'Uploading…';
        var fd = new FormData();
        fd.append('file', f);
        fetch('/api/portal/' + tok + '/documents',
              {method: 'POST', body: fd, credentials: 'same-origin'})
          .then(function (r) {
            return r.json().catch(function () { return {}; })
              .then(function (j) { return {ok: r.ok, body: j}; });
          })
          .then(function (res) {
            if (res.ok && res.body && res.body.ok) {
              return load();
            }
            if (status) {
              status.textContent = (res.body && res.body.error) ||
                'Upload failed. Please try again.';
            }
            btn.disabled = false;
          })
          .catch(function () {
            if (status) status.textContent = 'Upload failed. Please try again.';
            btn.disabled = false;
          });
      });
    }

    function load() {
      return fetch('/api/portal/' + tok + '/documents')
        .then(function (r) { return r.ok ? r.json() : {enabled: false, items: []}; })
        .then(function (d) {
          mount.innerHTML = d.enabled ? renderDocuments(d.items) : '';
          if (d.enabled) wireUpload();
        })
        .catch(function () {});
    }

    load();
  });
}
