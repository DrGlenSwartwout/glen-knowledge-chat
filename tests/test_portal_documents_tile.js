// tests/test_portal_documents_tile.js
// Run: node tests/test_portal_documents_tile.js
const assert = require('assert');
const { renderDocuments } = require('../static/js/portal-documents.js');

// empty -> the tile still renders, WITH the upload control (otherwise a
// client can never make a first upload), but no doc list.
const empty = renderDocuments([]);
assert.ok(empty.includes('My Records'));
assert.ok(empty.includes('doc-upload'));
assert.ok(/<input type=["']file["']/.test(empty));
assert.ok(!empty.includes('doc-list'));

const emptyFromNull = renderDocuments(null);
assert.ok(emptyFromNull.includes('doc-upload'));
assert.ok(!emptyFromNull.includes('doc-list'));

// under review -> shows the file link and the review line, no narrative
const pending = renderDocuments([{
  id: 1, filename: 'labs.pdf', uploaded_at: '2026-07-23T00:00:00Z',
  status: 'under_review', file_url: '/api/portal/t/documents/1/file',
  narrative_md: ''
}]);
assert.ok(pending.includes('My Records'));
assert.ok(pending.includes('/api/portal/t/documents/1/file'));
assert.ok(pending.includes('Received — under review'));

// ready -> shows the narrative
const ready = renderDocuments([{
  id: 2, filename: 'panel.pdf', uploaded_at: '2026-07-23T00:00:00Z',
  status: 'ready', file_url: '/api/portal/t/documents/2/file',
  narrative_md: 'Your panel looked at three things.'
}]);
assert.ok(ready.includes('Your panel looked at three things.'));
assert.ok(!ready.includes('Received — under review'));

// filenames are escaped, never injected
const evil = renderDocuments([{
  id: 3, filename: '<img src=x onerror=alert(1)>', uploaded_at: '',
  status: 'under_review', file_url: '/f', narrative_md: ''
}]);
assert.ok(!evil.includes('<img src=x'));
assert.ok(evil.includes('&lt;img'));

// narrative_md is escaped, never injected, for a ready document
const evilNarrative = renderDocuments([{
  id: 4, filename: 'panel.pdf', uploaded_at: '',
  status: 'ready', file_url: '/f',
  narrative_md: '<script>alert(1)</script><img src=x onerror=alert(2)>'
}]);
assert.ok(!evilNarrative.includes('<script>alert(1)</script>'));
assert.ok(!evilNarrative.includes('<img src=x'));
assert.ok(evilNarrative.includes('&lt;script&gt;alert(1)&lt;/script&gt;'));
assert.ok(evilNarrative.includes('&lt;img src=x onerror=alert(2)&gt;'));

// file_url is escaped so it cannot break out of the href attribute
const evilUrl = renderDocuments([{
  id: 5, filename: 'panel.pdf', uploaded_at: '',
  status: 'under_review', file_url: '/f" onmouseover="alert(1)',
  narrative_md: ''
}]);
assert.ok(!evilUrl.includes('href="/f" onmouseover="alert(1)"'));
assert.ok(evilUrl.includes('href="/f&quot; onmouseover=&quot;alert(1)"'));

console.log('ok - portal documents tile');
