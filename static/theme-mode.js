/* Single theme controller. Owns mode (rm-theme-mode), migrates legacy keys,
   resolves to data-theme, caches geolocation once (rm-geo), and renders the
   shared 3-state toggle. Sets data-theme ONLY; never injects palette CSS.
   Load in <head>, not deferred, before paint. */
(function () {
  var MODE_KEY = 'rm-theme-mode', GEO_KEY = 'rm-geo', MIRROR_KEY = 'rm-theme';
  var VALID = { light: 1, dark: 1, auto: 1 };
  var geoDenied = false;

  function lsGet(k){ try { return localStorage.getItem(k); } catch (e) { return null; } }
  function lsSet(k,v){ try { localStorage.setItem(k,v); } catch (e) {} }

  function getMode(){
    var m = lsGet(MODE_KEY);
    return VALID[m] ? m : 'auto';
  }

  // Pure: mode + context -> concrete theme. Unit tested.
  function _resolve(mode, ctx){
    if (mode === 'light' || mode === 'dark') return mode;
    var sr = ctx.sunrise, ss = ctx.sunset, h = ctx.nowH;
    if (sr == null || ss == null) return (h >= 7 && h < 19) ? 'light' : 'dark';
    return (h >= sr && h < ss) ? 'light' : 'dark';
  }

  function geo(){
    var raw = lsGet(GEO_KEY);
    if (raw) { try { return JSON.parse(raw); } catch (e) {} }
    // Longitude from the time-zone offset; latitude unknown until asked.
    return { lat: null, lng: -(new Date().getTimezoneOffset()) / 60 * 15 };
  }

  function currentContext(){
    var now = new Date();
    var g = geo();
    var st = (g.lat == null)
      ? { sunrise: null, sunset: null }
      : window.RMSun.sunTimes(now, g.lat, g.lng);
    return { nowH: now.getHours() + now.getMinutes() / 60, sunrise: st.sunrise, sunset: st.sunset };
  }

  function apply(){
    var theme = _resolve(getMode(), currentContext());
    document.documentElement.setAttribute('data-theme', theme);
    lsSet(MIRROR_KEY, theme); // legacy cross-document sync only
    try { document.dispatchEvent(new CustomEvent('rm-theme-change', { detail: { theme: theme, mode: getMode() } })); } catch (e) {}
    return theme;
  }

  function requestLocation(){
    if (geoDenied || !navigator.geolocation) return;
    navigator.geolocation.getCurrentPosition(function(p){
      lsSet(GEO_KEY, JSON.stringify({ lat: p.coords.latitude, lng: p.coords.longitude }));
      apply();
    }, function(){ geoDenied = true; }, { timeout: 8000, maximumAge: 86400000 });
  }

  function setMode(mode){
    if (!VALID[mode]) mode = 'auto';
    lsSet(MODE_KEY, mode);
    if (mode === 'auto' && geo().lat == null) requestLocation();
    apply();
  }

  function migrate(){
    if (VALID[lsGet(MODE_KEY)]) return;               // already on the new key
    var legacy = lsGet('rm-theme') || lsGet('rm_portal_theme');
    lsSet(MODE_KEY, (legacy === 'light' || legacy === 'dark') ? legacy : 'auto');
  }

  var ICONS = {
    light: '<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round"><circle cx="12" cy="12" r="4.5"/><path d="M12 2v2M12 20v2M4 12H2M22 12h-2M5 5l1.5 1.5M17.5 17.5 19 19M19 5l-1.5 1.5M6.5 17.5 5 19"/></svg>',
    dark:  '<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><path d="M21 12.8A9 9 0 1 1 11.2 3 7 7 0 0 0 21 12.8z"/></svg>',
    auto:  '<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round"><path d="M3 18h18"/><path d="M7 18a5 5 0 0 1 10 0"/><path d="M12 3v3M4.5 9l1.6 1.6M19.5 9l-1.6 1.6"/></svg>'
  };
  var LABELS = { light: 'Light', dark: 'Dark', auto: 'Auto' };

  function mountToggle(container){
    if (!container) return;
    var seg = document.createElement('div');
    seg.className = 'rm-theme-seg';
    seg.setAttribute('role', 'group');
    seg.setAttribute('aria-label', 'Theme mode');
    ['light', 'dark', 'auto'].forEach(function(m){
      var b = document.createElement('button');
      b.type = 'button';
      b.className = 'rm-theme-seg-btn';
      b.dataset.mode = m;
      b.title = LABELS[m];
      b.setAttribute('aria-label', LABELS[m]);
      b.innerHTML = ICONS[m];
      b.onclick = function(){ setMode(m); };
      seg.appendChild(b);
    });
    container.appendChild(seg);
    function refresh(){
      var mode = getMode();
      seg.querySelectorAll('.rm-theme-seg-btn').forEach(function(b){
        b.setAttribute('aria-pressed', b.dataset.mode === mode ? 'true' : 'false');
      });
    }
    document.addEventListener('rm-theme-change', refresh);
    refresh();
  }

  function init(){
    migrate();
    apply();
    if (getMode() === 'auto' && geo().lat == null) requestLocation();
    // Re-resolve periodically so Auto flips across sunrise/sunset without a reload.
    setInterval(function(){ if (getMode() === 'auto') apply(); }, 600000);
    // Follow theme changes made in other same-origin documents.
    window.addEventListener('storage', function(e){
      if (e.key === MODE_KEY || e.key === MIRROR_KEY) apply();
    });
  }

  window.RMTheme = {
    _resolve: _resolve, getMode: getMode, setMode: setMode,
    resolvedTheme: function(){ return _resolve(getMode(), currentContext()); },
    sunTimes: function(d, lat, lng){ return window.RMSun.sunTimes(d, lat, lng); },
    requestLocation: requestLocation, mountToggle: mountToggle, init: init
  };

  // Auto-run only in a real browser (guarded so Node tests can require this file).
  if (typeof document !== 'undefined' && document.documentElement && typeof localStorage !== 'undefined') {
    init();
  }
})();
