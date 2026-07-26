// tests/test_console_documents_render.js
// Run: node tests/test_console_documents_render.js
const assert = require('assert');
const { renderDocumentsHtml, renderDocUploadHtml } = require('../static/js/console-documents.js');

// the upload control renders unconditionally (called once above the list by
// console-client.html) and exposes a file input
const uploadHtml = renderDocUploadHtml();
assert.ok(/<input type=["']file["']/.test(uploadHtml));
assert.ok(/id=["']cd-upload-btn["']/.test(uploadHtml));

const ITEM = {
  id: 5, filename: 'labs.pdf', uploaded_at: '2026-07-23T00:00:00Z',
  source: 'console', extract_status: 'drafted', client_visible: false,
  file_url: '/admin/client-document?id=5',
  draft: {
    id: 9, status: 'ai_draft', narrative_md: 'Draft narrative.',
    attributes: [{ field: 'conditions', value: 'Glaucoma', source_quote: 'Assessment: glaucoma' }],
    facts: [{ fact_key: 'on_areds2', value: true, source_quote: 'taking AREDS2' }],
    unstructured: [{ label: 'HbA1c', value: '6.4', source_quote: 'HbA1c 6.4' }]
  }
};

const html = renderDocumentsHtml([ITEM], 'k');

// per-document visibility control: labelled so the current state (staff-only
// here, since client_visible is false) is obvious
assert.ok(/Staff only/.test(html));
assert.ok(html.includes('cd-toggle-visibility'));
assert.ok(html.includes('data-doc="5"'));
assert.ok(/Make visible to client/.test(html));

const visibleItem = Object.assign({}, ITEM, { client_visible: true });
const visibleHtml = renderDocumentsHtml([visibleItem], 'k');
assert.ok(/Visible to client/.test(visibleHtml));
assert.ok(/Make staff-only/.test(visibleHtml));

// the raw file is reachable, with the console key attached
assert.ok(html.includes('/admin/client-document?id=5'));
// proposals render as PRE-CHECKED boxes carrying their index
assert.ok(/type=['"]checkbox['"][^>]*checked/.test(html));
assert.ok(html.includes('data-kind="attributes"'));
assert.ok(html.includes('data-idx="0"'));
// every proposal shows its source quote so an invention is visible at a glance
assert.ok(html.includes('Assessment: glaucoma'));
assert.ok(html.includes('taking AREDS2'));
// labs are shown but marked as not stored structurally
assert.ok(html.includes('HbA1c'));
assert.ok(/not stored/i.test(html));
// the narrative is editable
assert.ok(html.includes('<textarea'));
assert.ok(html.includes('Draft narrative.'));
// both actions are present
assert.ok(/Approve/.test(html) && /Reject/.test(html));

// a confirmed draft shows as reviewed, with no approve button
const done = renderDocumentsHtml([Object.assign({}, ITEM, {
  draft: Object.assign({}, ITEM.draft, { status: 'confirmed' })
})], 'k');
assert.ok(/Approved/i.test(done));
assert.ok(!/>Approve</.test(done));

// a document with no draft yet says so and offers no checkboxes
const raw = renderDocumentsHtml([{
  id: 6, filename: 'raw.pdf', uploaded_at: '', source: 'console',
  extract_status: 'pending', file_url: '/admin/client-document?id=6', draft: null
}], 'k');
assert.ok(/awaiting extraction/i.test(raw));
assert.ok(!raw.includes('type="checkbox"'));

// empty state
assert.strictEqual(renderDocumentsHtml([], 'k'), '<p class="empty">No documents.</p>');

// filenames and quotes are escaped, never injected
const evil = renderDocumentsHtml([{
  id: 7, filename: '<img src=x onerror=alert(1)>', uploaded_at: '', source: 'console',
  extract_status: 'drafted', file_url: '/f', draft: {
    id: 1, status: 'ai_draft', narrative_md: '', attributes: [], facts: [],
    unstructured: []
  }
}], 'k');
assert.ok(!evil.includes('<img src=x'));
assert.ok(evil.includes('&lt;img'));

console.log('ok - console documents review render');
