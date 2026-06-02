/* Shared Light/Dark theme toggle for the funnel pages.
   Include in <head> (not deferred) so the saved theme applies before paint.
   Applies the saved theme, injects a light-palette override (covering both the
   funnel variable names and the tone-analyzer's), and adds a fixed toggle button. */
(function(){
  try { var t = localStorage.getItem('rm-theme'); if (t) document.documentElement.setAttribute('data-theme', t); } catch (e) {}

  if (!document.getElementById('rm-theme-style')) {
    var s = document.createElement('style');
    s.id = 'rm-theme-style';
    s.textContent = ':root[data-theme="light"]{' +
      '--bg:#FBF8F3;--surface:#FFFFFF;--surface-2:#F4ECDE;--border:#E2D9C9;' +
      '--cream:#1E2A2A;--muted:#5F6B6B;--gold:#B08A3E;--green:#2D7A6A;' +
      '--panel:#FFFFFF;--panel-2:#F4ECDE;--ink:#1E2A2A;--dim:#5F6B6B;' +
      '--hair:#E2D9C9;--accent:#B08A3E;}';
    (document.head || document.documentElement).appendChild(s);
  }

  function addBtn(){
    if (document.getElementById('rm-theme-toggle') || !document.body) return;
    var b = document.createElement('button');
    b.id = 'rm-theme-toggle';
    b.style.cssText = 'position:fixed;top:12px;right:12px;z-index:99999;' +
      'background:transparent;color:var(--gold,#d4a843);' +
      'border:1px solid var(--border,var(--hair,#21472d));border-radius:8px;' +
      'padding:6px 12px;font:600 13px/1 system-ui,sans-serif;cursor:pointer';
    function label(){ b.textContent = document.documentElement.getAttribute('data-theme') === 'light' ? 'Dark' : 'Light'; }
    label();
    b.addEventListener('click', function(){
      var next = document.documentElement.getAttribute('data-theme') === 'light' ? 'dark' : 'light';
      document.documentElement.setAttribute('data-theme', next);
      try { localStorage.setItem('rm-theme', next); } catch (e) {}
      label();
    });
    document.body.appendChild(b);
  }
  if (document.readyState === 'loading') document.addEventListener('DOMContentLoaded', addBtn);
  else addBtn();
})();
