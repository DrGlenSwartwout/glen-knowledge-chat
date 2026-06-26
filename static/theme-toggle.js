/* Shared Light/Dark theme toggle for the funnel pages and the chat widget.
   Include in <head> (not deferred) so the saved theme applies before paint.
   - Applies the saved theme from localStorage('rm-theme').
   - Injects a light-palette override covering the funnel var names, the
     tone-analyzer/journal alt names, AND the chat-widget alt names.
   - Adds a fixed toggle button, EXCEPT when loaded as the funnel chat iframe
     (?mode=funnel) where the parent funnel already owns the toggle.
   - Live-syncs across same-origin documents via the 'storage' event, so the
     chat iframe follows the funnel toggle instantly. */
(function(){
  function applyTheme(t){
    if (t === 'light' || t === 'dark') document.documentElement.setAttribute('data-theme', t);
    else document.documentElement.removeAttribute('data-theme');
  }
  try { applyTheme(localStorage.getItem('rm-theme')); } catch (e) {}

  if (!document.getElementById('rm-theme-style')) {
    var s = document.createElement('style');
    s.id = 'rm-theme-style';
    s.textContent = ':root[data-theme="light"]{' +
      '--bg:#FBF8F3;--surface:#FFFFFF;--surface-2:#F4ECDE;--border:#E2D9C9;' +
      '--cream:#1E2A2A;--muted:#5F6B6B;--gold:#B08A3E;--green:#2D7A6A;' +
      '--panel:#FFFFFF;--panel-2:#F4ECDE;--ink:#1E2A2A;--dim:#5F6B6B;' +
      '--hair:#E2D9C9;--accent:#B08A3E;' +
      /* chat-widget (embed.html) alternate names */
      '--text:#1E2A2A;--text-muted:#5F6B6B;--surface2:#F4ECDE;--accent2:#2D7A6A;}';
    (document.head || document.documentElement).appendChild(s);
  }

  // Inside the funnel chat iframe (?mode=funnel) the parent funnel owns the
  // toggle, so suppress our own button there — but still follow theme changes.
  var inFunnelIframe = false;
  try { inFunnelIframe = new URLSearchParams(location.search).get('mode') === 'funnel'; } catch (e) {}

  var relabel = function(){};
  function addBtn(){
    if (inFunnelIframe || document.getElementById('rm-theme-toggle') || !document.body) return;
    // Don't add a second control when another owner already provides the toggle:
    // the GLEN·OPS bar (op-nav.js) on internal pages, or the journey ribbon
    // (window.__SHELL__) on public pages.
    if (document.querySelector('.op-nav-bar') || window.__SHELL__) return;
    var b = document.createElement('button');
    b.id = 'rm-theme-toggle';
    b.style.cssText = 'position:fixed;top:12px;right:12px;z-index:99999;' +
      'background:transparent;color:var(--gold,var(--accent,#d4a843));' +
      'border:1px solid var(--border,var(--hair,#21472d));border-radius:8px;' +
      'padding:6px 12px;font:600 13px/1 system-ui,sans-serif;cursor:pointer';
    relabel = function(){ b.textContent = document.documentElement.getAttribute('data-theme') === 'light' ? 'Dark' : 'Light'; };
    relabel();
    b.addEventListener('click', function(){
      var next = document.documentElement.getAttribute('data-theme') === 'light' ? 'dark' : 'light';
      applyTheme(next);
      try { localStorage.setItem('rm-theme', next); } catch (e) {}
      relabel();
    });
    document.body.appendChild(b);
  }
  if (document.readyState === 'loading') document.addEventListener('DOMContentLoaded', addBtn);
  else addBtn();

  // Live-sync: another same-origin document (e.g. the funnel parent) changed the
  // theme. The 'storage' event fires only in OTHER documents, so this is how the
  // chat iframe follows the parent toggle without any postMessage wiring.
  window.addEventListener('storage', function(e){
    if (e.key !== 'rm-theme') return;
    applyTheme(e.newValue);
    relabel();
  });
})();
