// static/js/console-documents.js
// Console Documents review section: the ONE screen where Glen turns an AI draft
// into live clinical data. Renders the raw file, every proposal beside the
// verbatim quote it came from, and the editable narrative.
//
// Loaded as a plain script on the console page (it defines globals) and also
// exported for the node render test.
function cdEsc(s) {
  return String(s == null ? '' : s).replace(/[&<>"']/g, function (c) {
    return {'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[c];
  });
}

function cdProposalRow(kind, idx, label, quote) {
  return '<li class="cd-prop">' +
    '<label><input type="checkbox" checked data-kind="' + kind + '" ' +
      'data-idx="' + idx + '"> ' + cdEsc(label) + '</label>' +
    '<blockquote class="cd-quote">' + cdEsc(quote) + '</blockquote>' +
  '</li>';
}

function renderDocumentsHtml(items, consoleKey) {
  if (!items || !items.length) return '<p class="muted">No documents.</p>';
  return items.map(function (it) {
    var head = '<h3>' + cdEsc(it.filename) + ' ' +
      '<span class="pill">' + cdEsc(it.source) + '</span> ' +
      '<a href="' + cdEsc(it.file_url) + '&key=' +
        encodeURIComponent(consoleKey || '') +
        '" target="_blank" rel="noopener">open file</a></h3>';

    var d = it.draft;
    if (!d) {
      return '<section class="cd-doc" data-doc="' + it.id + '">' + head +
        '<p class="muted">Awaiting extraction (' +
        cdEsc(it.extract_status) + ').</p></section>';
    }
    if (d.status !== 'ai_draft') {
      return '<section class="cd-doc" data-doc="' + it.id + '">' + head +
        '<p class="muted">' +
        (d.status === 'confirmed' ? 'Approved' : 'Rejected') +
        (d.reviewed_by ? ' by ' + cdEsc(d.reviewed_by) : '') +
        '.</p></section>';
    }

    var attrs = (d.attributes || []).map(function (a, i) {
      return cdProposalRow('attributes', i, a.field + ': ' + a.value,
                           a.source_quote);
    }).join('');
    var facts = (d.facts || []).map(function (f, i) {
      return cdProposalRow('facts', i,
                           f.fact_key + ' = ' + (f.value ? 'yes' : 'no'),
                           f.source_quote);
    }).join('');
    var labs = (d.unstructured || []).map(function (u) {
      return '<li>' + cdEsc(u.label) + ': ' + cdEsc(u.value) +
        '<blockquote class="cd-quote">' + cdEsc(u.source_quote) +
        '</blockquote></li>';
    }).join('');

    return '<section class="cd-doc" data-doc="' + it.id + '">' + head +
      (attrs ? '<h4>Proposed attributes</h4><ul class="cd-props">' + attrs + '</ul>' : '') +
      (facts ? '<h4>Proposed facts</h4><ul class="cd-props">' + facts + '</ul>' : '') +
      (labs ? '<h4>Labs and medications <span class="muted">(not stored ' +
              'structurally — reference only)</span></h4><ul class="cd-labs">' +
              labs + '</ul>' : '') +
      '<h4>Client narrative</h4>' +
      '<textarea class="cd-narrative" rows="8">' + cdEsc(d.narrative_md) +
        '</textarea>' +
      '<p class="cd-actions">' +
        '<button class="cd-approve" data-doc="' + it.id + '">Approve</button> ' +
        '<button class="cd-reject" data-doc="' + it.id + '">Reject</button>' +
      '</p>' +
    '</section>';
  }).join('');
}

if (typeof module !== 'undefined' && module.exports) {
  module.exports = { renderDocumentsHtml: renderDocumentsHtml };
}
