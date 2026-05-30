/* mic-input.js — drop-in voice-to-text for any field marked with data-mic.
 *
 * Usage: add `data-mic` to a <textarea> or <input>, then include this script.
 *   <textarea id="notes" data-mic></textarea>
 *   <script src="/static/mic-input.js"></script>
 *
 * It injects a small mic button at the field's bottom-right. Click to record,
 * click again to stop; the audio is POSTed to /transcribe (Whisper) and the
 * returned text is inserted into the field. Dynamically-rendered fields are
 * picked up via a MutationObserver. No-op on browsers without MediaRecorder.
 *
 * Reuses the recorder pattern from embed.html.
 */
(function () {
  var SUPPORTED = !!(window.MediaRecorder && navigator.mediaDevices &&
                     navigator.mediaDevices.getUserMedia);
  if (!SUPPORTED) return;

  // ── styles (once) ──────────────────────────────────────────────────────────
  var css = ''
    + '.mic-wrap{position:relative;display:block;flex:1 1 auto;}'
    + '.mic-btn{position:absolute;bottom:6px;right:6px;z-index:5;'
    +   'width:26px;height:26px;display:flex;align-items:center;justify-content:center;'
    +   'border:1px solid #3a3a45;border-radius:6px;background:#1a1a22;color:#9aa0b4;'
    +   'font-size:13px;cursor:pointer;opacity:.75;'
    +   'transition:color .15s,border-color .15s,background .15s,opacity .15s;}'
    + '.mic-btn:hover{opacity:1;color:#e6edf3;border-color:#5a5a66;}'
    + '.mic-btn.recording{color:#f85149;border-color:#c0392b;background:#2d1010;opacity:1;'
    +   'animation:mic-pulse 1s ease-in-out infinite;}'
    + '.mic-btn.busy{opacity:.45;cursor:not-allowed;}'
    + '@keyframes mic-pulse{0%,100%{box-shadow:0 0 0 0 rgba(248,81,73,.5);}50%{box-shadow:0 0 0 5px rgba(248,81,73,0);}}';
  var st = document.createElement('style');
  st.id = 'mic-input-styles';
  st.textContent = css;
  document.head.appendChild(st);

  // ── per-button recorder ────────────────────────────────────────────────────
  function setState(btn, s) {
    btn.classList.remove('recording', 'busy');
    if (s === 'recording') { btn.classList.add('recording'); btn.textContent = '⏺'; }
    else if (s === 'busy') { btn.classList.add('busy'); btn.textContent = '…'; }
    else { btn.textContent = '🎙'; }
  }

  function insertText(field, text) {
    if (!text) return;
    var cur = field.value || '';
    if (!cur.trim()) field.value = text;
    else {
      // append at cursor if focused, else at end
      var sep = /\s$/.test(cur) ? '' : ' ';
      field.value = cur + sep + text;
    }
    field.dispatchEvent(new Event('input', { bubbles: true }));
    field.focus();
  }

  function wire(field) {
    if (field.dataset.micReady) return;
    field.dataset.micReady = '1';

    // wrap the field so the button can sit in its corner
    var wrap = document.createElement('span');
    wrap.className = 'mic-wrap';
    field.parentNode.insertBefore(wrap, field);
    wrap.appendChild(field);
    field.style.width = '100%';

    var btn = document.createElement('button');
    btn.type = 'button';
    btn.className = 'mic-btn';
    btn.title = 'Dictate (voice to text)';
    btn.textContent = '🎙';
    wrap.appendChild(btn);

    var recorder = null, chunks = [], stream = null, recording = false;

    function stopTracks() { if (stream) { stream.getTracks().forEach(function (t) { t.stop(); }); stream = null; } }

    function start() {
      navigator.mediaDevices.getUserMedia({ audio: true }).then(function (s) {
        stream = s;
        var mime = MediaRecorder.isTypeSupported('audio/webm;codecs=opus') ? 'audio/webm;codecs=opus'
                 : (MediaRecorder.isTypeSupported('audio/mp4') ? 'audio/mp4' : '');
        recorder = mime ? new MediaRecorder(s, { mimeType: mime }) : new MediaRecorder(s);
        chunks = [];
        recorder.ondataavailable = function (e) { if (e.data && e.data.size) chunks.push(e.data); };
        recorder.onstop = onStop;
        recorder.start();
        recording = true;
        setState(btn, 'recording');
      }).catch(function (e) {
        console.warn('[mic] getUserMedia failed:', e);
        setState(btn, 'idle');
      });
    }

    function onStop() {
      stopTracks();
      setState(btn, 'busy');
      var type = (recorder && recorder.mimeType) || 'audio/webm';
      var blob = new Blob(chunks, { type: type });
      chunks = []; recorder = null; recording = false;
      var ext = type.indexOf('mp4') !== -1 ? 'clip.mp4' : 'clip.webm';
      var fd = new FormData();
      fd.append('audio', blob, ext);
      fetch('/transcribe', { method: 'POST', body: fd, credentials: 'same-origin' })
        .then(function (r) { return r.ok ? r.json() : null; })
        .then(function (d) { insertText(field, d && d.text ? d.text.trim() : ''); })
        .catch(function (e) { console.warn('[mic] /transcribe failed:', e); })
        .finally(function () { setState(btn, 'idle'); });
    }

    btn.addEventListener('click', function (ev) {
      ev.preventDefault(); ev.stopPropagation();
      if (btn.classList.contains('busy')) return;
      if (recording && recorder) { try { recorder.stop(); } catch (e) {} }
      else { start(); }
    });
  }

  function scan(root) {
    var nodes = (root.querySelectorAll ? root.querySelectorAll('[data-mic]') : []);
    for (var i = 0; i < nodes.length; i++) wire(nodes[i]);
    if (root.matches && root.matches('[data-mic]')) wire(root);
  }

  function init() {
    scan(document);
    new MutationObserver(function (muts) {
      for (var i = 0; i < muts.length; i++) {
        var added = muts[i].addedNodes;
        for (var j = 0; j < added.length; j++) {
          if (added[j].nodeType === 1) scan(added[j]);
        }
      }
    }).observe(document.body, { childList: true, subtree: true });
  }

  if (document.readyState === 'loading') document.addEventListener('DOMContentLoaded', init);
  else init();
})();
