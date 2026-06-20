/* chat-attachments.js — shared document + image upload for the chat surfaces.
 *
 * Usage:
 *   <div id="chat-attach"></div>
 *   <script src="/static/chat-attachments.js"></script>
 *   <script>
 *     const attach = ChatAttach.mount({ host: '#chat-attach', dropZone: '#input-bar' });
 *     // when sending: const { images, documents, consented } = attach.getPayload();
 *     // after a successful send: attach.clear();
 *   </script>
 *
 * Files are held in memory only (never localStorage). Consent is the single
 * boolean persisted under 'amg_images_consented'. The backend re-validates
 * everything; these caps are UX guardrails.
 */
(function (global) {
  'use strict';

  var MAX_IMAGES = 3;
  var MAX_DOCS = 2;
  var MAX_IMAGE_BYTES = 5 * 1024 * 1024;
  var MAX_DOC_BYTES = 10 * 1024 * 1024;
  var IMG_TYPES = ['image/png', 'image/jpeg', 'image/webp', 'image/gif'];
  var CONSENT_KEY = 'amg_images_consented';

  var STYLE = [
    '.ca-wrap{font:inherit;margin:6px 0;}',
    '.ca-consent{display:flex;align-items:flex-start;gap:7px;font-size:12px;opacity:.85;line-height:1.35;}',
    '.ca-consent input{margin-top:2px;}',
    '.ca-row{display:flex;align-items:center;flex-wrap:wrap;gap:8px;margin-top:6px;}',
    '.ca-btn{cursor:pointer;border:1px solid currentColor;background:transparent;color:inherit;',
    'border-radius:6px;padding:4px 10px;font-size:13px;opacity:.9;}',
    '.ca-btn:hover{opacity:1;}',
    '.ca-help{font-size:12px;opacity:.6;}',
    '.ca-err{font-size:12px;color:#c0392b;}',
    '.ca-chip{display:inline-flex;align-items:center;gap:6px;border:1px solid rgba(128,128,128,.4);',
    'border-radius:6px;padding:2px 6px;font-size:12px;max-width:180px;}',
    '.ca-chip img{width:26px;height:26px;object-fit:cover;border-radius:3px;}',
    '.ca-chip .ca-name{overflow:hidden;text-overflow:ellipsis;white-space:nowrap;}',
    '.ca-chip .ca-x{cursor:pointer;border:none;background:transparent;color:inherit;font-size:15px;line-height:1;}'
  ].join('');

  function injectStyleOnce() {
    if (document.getElementById('ca-style')) return;
    var s = document.createElement('style');
    s.id = 'ca-style';
    s.textContent = STYLE;
    document.head.appendChild(s);
  }

  function readAsDataURL(file) {
    return new Promise(function (resolve, reject) {
      var r = new FileReader();
      r.onload = function () { resolve(r.result); };
      r.onerror = function () { reject(r.error); };
      r.readAsDataURL(file);
    });
  }

  function mount(opts) {
    opts = opts || {};
    var host = typeof opts.host === 'string'
      ? document.querySelector(opts.host) : opts.host;
    if (!host) throw new Error('ChatAttach.mount: host not found');
    injectStyleOnce();

    var pending = []; // { data_url, name, kind: 'image'|'document' }

    host.classList.add('ca-wrap');
    host.innerHTML =
      '<label class="ca-consent">' +
        '<input type="checkbox" class="ca-consent-cb">' +
        '<span>Allow attaching documents and images (lab results, supplement ' +
        'labels, scan PDFs, photos). Content is extracted as text to answer ' +
        'your question; the original file is not saved.</span>' +
      '</label>' +
      '<div class="ca-row">' +
        '<button type="button" class="ca-btn ca-add">+ Add file</button>' +
        '<span class="ca-help">images or PDF · or drag-drop</span>' +
        '<span class="ca-err"></span>' +
        '<input type="file" class="ca-input" multiple style="display:none" ' +
          'accept="image/png,image/jpeg,image/webp,image/gif,application/pdf">' +
      '</div>';

    var cb = host.querySelector('.ca-consent-cb');
    var input = host.querySelector('.ca-input');
    var addBtn = host.querySelector('.ca-add');
    var errEl = host.querySelector('.ca-err');
    var row = host.querySelector('.ca-row');

    try { cb.checked = localStorage.getItem(CONSENT_KEY) === 'true'; } catch (e) {}
    cb.addEventListener('change', function () {
      try { localStorage.setItem(CONSENT_KEY, cb.checked ? 'true' : 'false'); } catch (e) {}
    });

    function consented() { return !!cb.checked; }

    function setError(msg) {
      errEl.textContent = msg || '';
      if (msg) setTimeout(function () {
        if (errEl.textContent === msg) errEl.textContent = '';
      }, 4000);
    }

    function refresh() {
      Array.prototype.slice.call(host.querySelectorAll('.ca-chip'))
        .forEach(function (el) { el.remove(); });
      pending.forEach(function (f, idx) {
        var chip = document.createElement('span');
        chip.className = 'ca-chip';
        var thumb = f.kind === 'image'
          ? '<img src="' + f.data_url + '" alt="">'
          : '<span aria-hidden="true">📄</span>';
        chip.innerHTML = thumb +
          '<span class="ca-name"></span>' +
          '<button type="button" class="ca-x" title="Remove">×</button>';
        chip.querySelector('.ca-name').textContent = f.name || 'file';
        chip.querySelector('.ca-x').addEventListener('click', function () {
          pending.splice(idx, 1); refresh();
        });
        row.insertBefore(chip, addBtn);
      });
    }

    function countKind(kind) {
      return pending.filter(function (f) { return f.kind === kind; }).length;
    }

    async function addFiles(fileList) {
      if (!fileList || !fileList.length) return;
      if (!consented()) { setError('Check the consent box first.'); return; }
      var files = Array.prototype.slice.call(fileList);
      for (var i = 0; i < files.length; i++) {
        var file = files[i];
        var isImage = file.type && IMG_TYPES.indexOf(file.type) !== -1;
        var isPdf = file.type === 'application/pdf';
        if (!isImage && !isPdf) { setError('Only images or PDF files are accepted.'); continue; }
        if (isImage && countKind('image') >= MAX_IMAGES) { setError('Max ' + MAX_IMAGES + ' images.'); continue; }
        if (isPdf && countKind('document') >= MAX_DOCS) { setError('Max ' + MAX_DOCS + ' PDFs.'); continue; }
        if (isImage && file.size > MAX_IMAGE_BYTES) { setError('"' + file.name + '" exceeds the 5 MB image limit.'); continue; }
        if (isPdf && file.size > MAX_DOC_BYTES) { setError('"' + file.name + '" exceeds the 10 MB PDF limit.'); continue; }
        try {
          var data_url = await readAsDataURL(file);
          pending.push({ data_url: data_url, name: file.name, kind: isImage ? 'image' : 'document' });
        } catch (e) { setError('Could not read "' + file.name + '".'); }
      }
      refresh();
    }

    addBtn.addEventListener('click', function () { input.click(); });
    input.addEventListener('change', function () { addFiles(input.files); input.value = ''; });

    // Paste (images only — browsers expose pasted images, not PDFs).
    document.addEventListener('paste', function (e) {
      if (!e.clipboardData || !e.clipboardData.items) return;
      var imgs = Array.prototype.slice.call(e.clipboardData.items)
        .filter(function (it) { return it.kind === 'file' && it.type.indexOf('image/') === 0; });
      if (!imgs.length) return;
      if (!consented()) { setError('Check the consent box first.'); e.preventDefault(); return; }
      e.preventDefault();
      addFiles(imgs.map(function (it) { return it.getAsFile(); }).filter(Boolean));
    });

    // Drag-drop onto an optional drop zone.
    var zone = opts.dropZone
      ? (typeof opts.dropZone === 'string' ? document.querySelector(opts.dropZone) : opts.dropZone)
      : null;
    if (zone) {
      ['dragenter', 'dragover'].forEach(function (ev) {
        zone.addEventListener(ev, function (e) { e.preventDefault(); zone.style.outline = '2px dashed currentColor'; });
      });
      ['dragleave', 'drop'].forEach(function (ev) {
        zone.addEventListener(ev, function (e) { e.preventDefault(); zone.style.outline = ''; });
      });
      zone.addEventListener('drop', function (e) {
        if (e.dataTransfer && e.dataTransfer.files) addFiles(e.dataTransfer.files);
      });
    }

    return {
      getPayload: function () {
        return {
          images: pending.filter(function (f) { return f.kind === 'image'; })
            .map(function (f) { return { data_url: f.data_url }; }),
          documents: pending.filter(function (f) { return f.kind === 'document'; })
            .map(function (f) { return { data_url: f.data_url, name: f.name }; }),
          consented: consented()
        };
      },
      clear: function () { pending = []; refresh(); }
    };
  }

  global.ChatAttach = { mount: mount };
})(window);
