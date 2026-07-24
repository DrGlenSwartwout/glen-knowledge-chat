// tests/test_portal_documents_tile.js
// Run: node tests/test_portal_documents_tile.js
const assert = require('assert');
const { renderDocuments } = require('../static/js/portal-documents.js');

// empty -> no tile at all
assert.strictEqual(renderDocuments([]), '');
assert.strictEqual(renderDocuments(null), '');

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

console.log('ok - portal documents tile');
