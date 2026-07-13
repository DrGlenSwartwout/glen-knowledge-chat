/* Funnel light-palette override. The theme itself (data-theme) and the toggle
   are owned by theme-mode.js. This file only guarantees the light palette CSS
   is present on funnel/chat surfaces. */
(function(){
  if (document.getElementById('rm-theme-style')) return;
  var s = document.createElement('style');
  s.id = 'rm-theme-style';
  s.textContent = ':root[data-theme="light"]{' +
    '--bg:#FBF8F3;--surface:#FFFFFF;--surface-2:#F4ECDE;--border:#E2D9C9;' +
    '--cream:#1E2A2A;--muted:#5F6B6B;--gold:#B08A3E;--green:#2D7A6A;' +
    '--panel:#FFFFFF;--panel-2:#F4ECDE;--ink:#1E2A2A;--dim:#5F6B6B;' +
    '--hair:#E2D9C9;--accent:#B08A3E;' +
    '--text:#1E2A2A;--text-muted:#5F6B6B;--surface2:#F4ECDE;--accent2:#2D7A6A;}';
  (document.head || document.documentElement).appendChild(s);
})();
