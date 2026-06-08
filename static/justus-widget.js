/* Justus Widget — a self-contained "Ask Justus" side-column for any console page.
 *
 * Drop-in: <script src="/static/justus-widget.js" defer></script>
 * Optional: data-reflow="<selector>" on the tag to reflow a specific container
 *           instead of <body> (defaults to body padding-right).
 *
 * Injects its own CSS + DOM, talks to /api/console-ask (SSE streaming), and
 * docks as a right-side column that pushes page content left so it never
 * overlaps. One source of truth — every console page includes this same file.
 */
(function () {
  'use strict';
  if (window.__justusWidgetLoaded) return;
  window.__justusWidgetLoaded = true;

  var SCRIPT   = document.currentScript;
  var REFLOW   = SCRIPT && SCRIPT.getAttribute('data-reflow');  // optional selector
  var PANEL_W  = 400;

  function consoleKey() {
    return localStorage.getItem('console_key') ||
           localStorage.getItem('consoleKey') || '';
  }

  // ── CSS (self-contained — does not depend on the host page's variables) ──────
  var css =
    '#jw-sidebar{position:fixed;top:0;right:-' + (PANEL_W + 24) + 'px;width:' + PANEL_W + 'px;' +
      'height:100vh;background:#111f16;border-left:1px solid #21472d;z-index:9000;' +
      'display:flex;flex-direction:column;transition:right .25s ease;' +
      'box-shadow:-4px 0 24px rgba(0,0,0,.45);font-family:-apple-system,BlinkMacSystemFont,"Segoe UI",sans-serif;}' +
    '#jw-sidebar.open{right:0;}' +
    '#jw-head{display:flex;align-items:center;justify-content:space-between;padding:14px 16px;' +
      'border-bottom:1px solid #21472d;background:#0a150d;}' +
    '#jw-head h3{font-size:14px;font-weight:700;color:#b8a3d4;margin:0;}' +
    '#jw-close{background:transparent;border:1px solid #21472d;color:#a89870;border-radius:6px;' +
      'padding:3px 9px;cursor:pointer;font-size:13px;}' +
    '#jw-close:hover{color:#fdf4d8;border-color:#fdf4d8;}' +
    '#jw-messages{flex:1;overflow-y:auto;padding:14px 16px;display:flex;flex-direction:column;gap:12px;}' +
    '.jw-msg{max-width:100%;font-size:13px;line-height:1.5;}' +
    '.jw-bubble{padding:9px 12px;border-radius:10px;white-space:pre-wrap;word-wrap:break-word;}' +
    '.jw-msg.user .jw-bubble{background:#3d8a52;color:#fff;border-radius:10px 10px 2px 10px;' +
      'margin-left:auto;max-width:85%;}' +
    '.jw-msg.assistant .jw-bubble{background:#0a150d;border:1px solid #21472d;color:#fdf4d8;' +
      'border-radius:10px 10px 10px 2px;max-width:96%;}' +
    '.jw-msg.assistant .jw-bubble strong{color:#fff;}' +
    /* Verbatim tool result (e.g. drafted email) — shown above the summary bubble */
    '.jw-tool-result{background:#0a150d;border:1px solid #3d8a52;color:#fdf4d8;' +
      'border-radius:10px;padding:10px 12px;margin-bottom:6px;max-width:96%;' +
      'font-family:ui-monospace,SFMono-Regular,Menlo,monospace;font-size:12px;' +
      'line-height:1.5;white-space:pre-wrap;word-wrap:break-word;}' +
    '.jw-tool-result-head{font-family:-apple-system,BlinkMacSystemFont,"Segoe UI",sans-serif;' +
      'font-size:10px;font-weight:700;letter-spacing:.12em;text-transform:uppercase;' +
      'color:#b8a3d4;margin-bottom:6px;}' +
    '.jw-tool-result-copy{float:right;background:transparent;border:1px solid #21472d;' +
      'color:#a89870;border-radius:4px;padding:1px 7px;cursor:pointer;font-size:10px;' +
      'font-family:inherit;letter-spacing:.04em;}' +
    '.jw-tool-result-copy:hover{color:#fdf4d8;border-color:#fdf4d8;}' +
    '#jw-input-row{padding:12px 14px;border-top:1px solid #21472d;display:flex;gap:8px;background:#0a150d;}' +
    '#jw-input{flex:1;padding:9px 11px;background:#111f16;border:1px solid #21472d;border-radius:6px;' +
      'color:#fdf4d8;font-size:13px;font-family:inherit;resize:none;height:38px;max-height:120px;overflow-y:auto;}' +
    '#jw-input:focus{outline:none;border-color:#b8a3d4;}' +
    '#jw-send{background:#b8a3d4;border:none;color:#0a150d;border-radius:6px;padding:0 16px;' +
      'font-weight:700;cursor:pointer;font-size:13px;}' +
    '#jw-send:disabled{opacity:.4;cursor:not-allowed;}' +
    '#jw-open-btn{position:fixed;bottom:20px;right:20px;z-index:8999;background:#b8a3d4;color:#0a150d;' +
      'border:none;border-radius:22px;padding:10px 18px;font-size:13px;font-weight:700;cursor:pointer;' +
      'box-shadow:0 4px 16px rgba(0,0,0,.4);transition:right .25s ease;}' +
    '#jw-open-btn.shifted{right:' + (PANEL_W + 20) + 'px;}' +
    '.jw-empty{color:#a89870;font-size:12px;font-style:italic;text-align:center;padding:18px 6px;}' +
    'body.jw-open{transition:padding-right .25s ease;}';
  var styleEl = document.createElement('style');
  styleEl.textContent = css;
  document.head.appendChild(styleEl);

  // ── DOM ──────────────────────────────────────────────────────────────────────
  var sidebar = document.createElement('div');
  sidebar.id = 'jw-sidebar';
  sidebar.innerHTML =
    '<div id="jw-head"><h3>✦ Ask Justus</h3>' +
    '<button id="jw-close" aria-label="Close">✕</button></div>' +
    '<div id="jw-messages"><div class="jw-empty">Ask me anything about this page, ' +
      'the business, clients, or operations.</div></div>' +
    '<div id="jw-input-row">' +
    '<textarea id="jw-input" placeholder="Ask Justus…" rows="1"></textarea>' +
    '<button id="jw-send">Send</button></div>';
  document.body.appendChild(sidebar);

  var openBtn = document.createElement('button');
  openBtn.id = 'jw-open-btn';
  openBtn.textContent = '✦ Ask Justus';
  document.body.appendChild(openBtn);

  var messagesEl = sidebar.querySelector('#jw-messages');
  var inputEl    = sidebar.querySelector('#jw-input');
  var sendBtn    = sidebar.querySelector('#jw-send');

  // ── Reflow target ────────────────────────────────────────────────────────────
  var reflowEl = REFLOW ? document.querySelector(REFLOW) : null;

  function applyReflow(open) {
    if (reflowEl) {
      reflowEl.style.transition = 'margin-right .25s ease';
      reflowEl.style.marginRight = open ? PANEL_W + 'px' : '';
    } else {
      document.body.classList.toggle('jw-open', open);
      document.body.style.paddingRight = open ? PANEL_W + 'px' : '';
    }
  }

  // ── State ────────────────────────────────────────────────────────────────────
  var history = [], streaming = false;

  function esc(s) {
    return String(s == null ? '' : s)
      .replace(/&/g, '&amp;').replace(/</g, '&lt;').replace(/>/g, '&gt;');
  }
  // Minimal markdown: bold, bullets, line breaks.
  function md(s) {
    var h = esc(s);
    h = h.replace(/\*\*([^*]+)\*\*/g, '<strong>$1</strong>');
    h = h.replace(/^[ \t]*[-•]\s+(.*)$/gm, '• $1');
    h = h.replace(/\n/g, '<br>');
    return h;
  }

  function open() {
    sidebar.classList.add('open');
    openBtn.classList.add('shifted');
    openBtn.textContent = '✕ Close';
    applyReflow(true);
    setTimeout(function () { inputEl.focus(); }, 280);
  }
  function close() {
    sidebar.classList.remove('open');
    openBtn.classList.remove('shifted');
    openBtn.textContent = '✦ Ask Justus';
    applyReflow(false);
  }
  function toggle() {
    sidebar.classList.contains('open') ? close() : open();
  }

  // Public hook: open the widget pre-filled with text (caller adds directions, then Send).
  window.openJustusWith = function (text) {
    open();
    inputEl.value = text || '';
    inputEl.style.height = 'auto';
    inputEl.style.height = Math.min(inputEl.scrollHeight, 120) + 'px';
    setTimeout(function () { inputEl.focus(); }, 300);
  };

  function pageContext() {
    return 'Page: ' + (document.title || 'console') + ' (' + location.pathname + ')';
  }

  async function send() {
    if (streaming) return;
    var query = inputEl.value.trim();
    if (!query) return;
    var key = consoleKey();
    if (!key) { alert('No console key found — open the main console once to unlock.'); return; }

    inputEl.value = ''; inputEl.style.height = '38px';
    var firstMsg = messagesEl.querySelector('.jw-empty');
    if (firstMsg) firstMsg.remove();

    history.push({ role: 'user', content: query });
    messagesEl.insertAdjacentHTML('beforeend',
      '<div class="jw-msg user"><div class="jw-bubble">' + esc(query) + '</div></div>');
    var resp = document.createElement('div');
    resp.className = 'jw-msg assistant';
    var bubble = document.createElement('div');
    bubble.className = 'jw-bubble';
    bubble.textContent = '…';
    resp.appendChild(bubble);
    messagesEl.appendChild(resp);
    messagesEl.scrollTop = messagesEl.scrollHeight;

    streaming = true; sendBtn.disabled = true;
    var acc = '';
    try {
      var res = await fetch('/api/console-ask', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json', 'X-Console-Key': key },
        body: JSON.stringify({
          query: query, owner: 'glen',
          context: pageContext(), history: history.slice(-6),
          page: window.location.pathname,
        }),
      });
      if (res.status === 401) { bubble.textContent = 'Unauthorized — console key rejected.'; }
      else {
        var reader = res.body.getReader(), dec = new TextDecoder(), buf = '';
        while (true) {
          var chunk = await reader.read();
          if (chunk.done) break;
          buf += dec.decode(chunk.value, { stream: true });
          var lines = buf.split('\n');
          buf = lines.pop();
          for (var i = 0; i < lines.length; i++) {
            var ln = lines[i];
            if (ln.indexOf('data:') !== 0) continue;
            try {
              var evt = JSON.parse(ln.slice(5).trim());
              if (evt.text) {
                acc += evt.text;
                bubble.innerHTML = md(acc);
                messagesEl.scrollTop = messagesEl.scrollHeight;
              } else if (evt.tool_result) {
                // Verbatim tool output (e.g. drafted email body) — inserted
                // above the summary bubble so the user sees the actual content,
                // not just Justus's "draft ready" recap.
                var tr = document.createElement('div');
                tr.className = 'jw-tool-result';
                var head = document.createElement('div');
                head.className = 'jw-tool-result-head';
                head.textContent = (evt.tool_result.name || 'Tool result').replace(/_/g, ' ');
                var copyBtn = document.createElement('button');
                copyBtn.className = 'jw-tool-result-copy';
                copyBtn.textContent = 'Copy';
                copyBtn.addEventListener('click', function () {
                  try {
                    navigator.clipboard.writeText(evt.tool_result.content || '');
                    copyBtn.textContent = 'Copied';
                    setTimeout(function () { copyBtn.textContent = 'Copy'; }, 1500);
                  } catch (e) { /* clipboard blocked */ }
                });
                head.appendChild(copyBtn);
                tr.appendChild(head);
                var body = document.createElement('div');
                body.textContent = evt.tool_result.content || '';
                tr.appendChild(body);
                resp.insertBefore(tr, bubble);
                messagesEl.scrollTop = messagesEl.scrollHeight;
              }
            } catch (e) { /* ignore partial */ }
          }
        }
      }
    } catch (e) {
      bubble.textContent = 'Error: ' + e.message;
    }
    if (acc) history.push({ role: 'assistant', content: acc });
    streaming = false; sendBtn.disabled = false;
    messagesEl.scrollTop = messagesEl.scrollHeight;
  }

  // ── Wire ─────────────────────────────────────────────────────────────────────
  openBtn.addEventListener('click', toggle);
  sidebar.querySelector('#jw-close').addEventListener('click', close);
  sendBtn.addEventListener('click', send);
  inputEl.addEventListener('keydown', function (e) {
    if (e.key === 'Enter' && !e.shiftKey) { e.preventDefault(); send(); }
  });
  inputEl.addEventListener('input', function () {
    inputEl.style.height = '38px';
    inputEl.style.height = Math.min(inputEl.scrollHeight, 120) + 'px';
  });
  document.addEventListener('keydown', function (e) {
    if (e.key === 'Escape' && sidebar.classList.contains('open')) close();
  });
})();
