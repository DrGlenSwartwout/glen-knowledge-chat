/* op-nav.js — shared sticky tab bar across Glen's internal control panels.
 *
 * Usage: drop one synchronous script tag immediately after <body> on each page:
 *   <script src="/static/op-nav.js" data-active="dashboard"></script>
 *
 * Valid data-active values: "dashboard" | "console" | "bos" | "projects" | "inbox" | "settings" | "funnel"
 *
 * Business OS sub-tabs: when data-active="bos", a secondary row of the BOS module
 * boards renders under the main bar. Mark the active board with data-sub, e.g.
 *   <script src="/static/op-nav.js" data-active="bos" data-sub="finance"></script>
 * Valid data-sub values: "orders" | "client-orders" | "payments" | "finance" | "crm" | "products" | "biofield" | "sales" | "ingredients" | "approvals" | "shipping" | "neworder"
 * (Shipping = /admin/shipping, New Order = /orders/new — folded in from the old
 *  standalone top tabs. The old "home" board is retired; its signals now live on
 *  the /console front door.)
 *
 * The bar:
 *   - Renders synchronously via document.write so there is no flash
 *   - Resolves the console key from ?key= OR localStorage('console_key'), persists
 *     a URL key to localStorage, and carries the resolved key across ALL internal
 *     links — so pages that gate on ?key= keep working once you've unlocked once
 *   - Highlights the active tab from the data-active attribute
 *   - Sticks to the top of the viewport while scrolling
 *   - Uses neutral dark styling that sits on top of all page palettes
 */
(function () {
  var script = document.currentScript;
  var active = (script && script.dataset && script.dataset.active) || "";
  var sub = (script && script.dataset && script.dataset.sub) || "";

  // --- Light/Dark theme: shared mechanism (localStorage 'rm-theme' + <html data-theme>),
  //     same as static/theme-toggle.js. op-nav owns the toggle on every internal page so
  //     Dashboard + Business OS get a light/dark control without per-page edits. ---
  function rmApplyTheme(t) {
    if (t === 'light' || t === 'dark') document.documentElement.setAttribute('data-theme', t);
    else document.documentElement.removeAttribute('data-theme');
  }
  try { rmApplyTheme(localStorage.getItem('rm-theme')); } catch (e) {}

  // Resolve the console key: a URL ?key= wins and is persisted; otherwise fall
  // back to a previously-stored key. This lets every internal link carry the key
  // even when the current page's URL doesn't have it (e.g. unlocked via a gate).
  var urlKey = new URLSearchParams(location.search).get("key") || "";
  if (urlKey) { try { localStorage.setItem("console_key", urlKey); } catch (e) {} }

  // On a PUBLIC page (e.g. the chatbot at /), only render the nav when a key is in
  // the URL — so public visitors aren't shown an internal "GLEN · OPS" bar.
  var isPublic = script && script.dataset && script.dataset.publicPage === "true";
  if (isPublic && !urlKey) return;

  // Brand favicon — the phoenix logo. Injected here so every page that loads the
  // ops nav gets the tab icon without per-page <head> edits. Only added if the
  // page doesn't already declare its own icon.
  try {
    if (!document.querySelector('link[rel~="icon"]')) {
      var fav = document.createElement("link");
      fav.rel = "icon";
      fav.type = "image/png";
      fav.href = "/static/favicon.png";
      document.head.appendChild(fav);
    }
  } catch (e) {}

  var storedKey = "";
  try { storedKey = localStorage.getItem("console_key") || ""; } catch (e) {}
  var effKey = urlKey || storedKey;
  var qs = effKey ? ("?key=" + encodeURIComponent(effKey)) : "";

  var tabs = [
    { id: "dashboard", label: "Dashboard", href: "/dashboard" + qs },
    { id: "console",   label: "Console",   href: "/console"   + qs },
    { id: "bos",       label: "Business OS", href: "/console/orders" + qs },
    { id: "projects",  label: "Projects",  href: "/console/projects" + qs },
    { id: "inbox",     label: "Inbox",     href: "/console/inbox" + qs },
    { id: "settings",  label: "Settings",  href: "/console/settings" + qs },
    { id: "funnel",    label: "Funnel",    href: "/funnel" + qs },
  ];

  // Business OS module boards — rendered as a secondary sub-tab row under the main
  // bar whenever the BOS section is active. Shipping + New Order were folded in
  // from the old standalone top tabs.
  var bosMods = [
    { id: "orders",   label: "Orders",    href: "/console/orders" + qs },
    { id: "client-orders", label: "Client Orders", href: "/console/client-orders" + qs },
    { id: "money",    label: "Money",     href: "/console/money" + qs },
    { id: "crm",      label: "CRM",       href: "/console/crm" + qs },
    { id: "products", label: "Products",  href: "/console/products" + qs },
    { id: "biofield", label: "Biofield",  href: "/console/biofield-portal" + qs },
    { id: "pages",    label: "Pages",      href: "/console/pages" + qs },
    { id: "biofield-reveals", label: "Biofield Reveals", href: "/console/biofield-reveals" + qs },
    { id: "biofield-intake", label: "Biofield Intake", href: "/console/biofield-intake" + qs },
    { id: "clinical-tags", label: "Tags", href: "/console/clinical-tags" + qs },
    { id: "approvals", label: "Approvals", href: "/console/approvals" + qs },
    { id: "testimonial-invites", label: "Invite Queue", href: "/console/testimonial-invites" + qs },
    { id: "shipping", label: "Shipping",  href: "/admin/shipping" + qs },
    { id: "neworder", label: "New Order", href: "/orders/new" + qs },
    { id: "practitioners", label: "Practitioners", href: "/console/practitioners" + qs },
    { id: "top-products",  label: "Top Products",  href: "/console/top-products" + qs },
    { id: "remedy-meanings", label: "Remedy Meanings", href: "/console/remedy-meanings" + qs },
    { id: "ingredients-ops", label: "Ingredients (Ops)", href: "/admin/ingredients" + qs },
    { id: "coaching",      label: "Coaching",      href: "/console/coaching-cohort" + qs },
    { id: "membership",    label: "Membership",    href: "/admin/membership" + qs },
    { id: "members",       label: "Members",       href: "/console/members" + qs },
  ];

  var styles = ''
    + '<style id="op-nav-styles">'
    + '.op-nav-bar{'
    +   'position:sticky;top:0;z-index:9999;'
    +   'display:flex;align-items:center;gap:0;'
    +   'background:#0a0a0f;border-bottom:1px solid #2a2a35;'
    +   'padding:0 14px;height:40px;'
    +   'font-family:-apple-system,BlinkMacSystemFont,"Segoe UI",sans-serif;'
    +   'font-size:13px;'
    +   'box-shadow:0 1px 0 rgba(0,0,0,.4),0 4px 12px rgba(0,0,0,.25);'
    + '}'
    + '.op-nav-bar .op-nav-brand{'
    +   'color:#9a9384;letter-spacing:.18em;text-transform:uppercase;'
    +   'font-size:10px;font-weight:600;margin-right:18px;'
    +   'font-family:ui-monospace,"SF Mono",Menlo,Consolas,monospace;'
    + '}'
    + '.op-nav-bar .op-nav-brand b{color:#e6b800;font-weight:700}'
    + '.op-nav-bar a.op-nav-tab{'
    +   'display:inline-flex;align-items:center;'
    +   'height:100%;padding:0 14px;'
    +   'color:#9aa0b4;text-decoration:none;'
    +   'border-bottom:2px solid transparent;'
    +   'transition:color .15s ease,border-color .15s ease,background .15s ease;'
    + '}'
    + '.op-nav-bar a.op-nav-tab:hover{color:#e6edf3;background:rgba(255,255,255,.03)}'
    + '.op-nav-bar a.op-nav-tab.active{'
    +   'color:#fff;border-bottom-color:#7c5cbf;'
    + '}'
    + '.op-nav-bar .op-nav-spacer{flex:1}'
    + '.op-nav-bar .op-nav-key-warn{'
    +   'color:#f85149;font-size:11px;margin-right:10px;'
    + '}'
    // Business OS secondary sub-tab row
    + '.op-nav-sub{'
    +   'position:sticky;top:40px;z-index:9998;'
    +   'display:flex;align-items:center;gap:0;'
    +   'background:#0d1c12;border-bottom:1px solid #21472d;'
    +   'padding:0 14px;height:36px;'
    +   'font-family:-apple-system,BlinkMacSystemFont,"Segoe UI",sans-serif;font-size:13px;'
    + '}'
    + '.op-nav-sub .op-nav-sub-brand{'
    +   'color:#a89870;letter-spacing:.16em;text-transform:uppercase;'
    +   'font-size:10px;font-weight:700;margin-right:16px;'
    +   'font-family:ui-monospace,"SF Mono",Menlo,Consolas,monospace;'
    + '}'
    + '.op-nav-sub a.op-nav-subtab{'
    +   'display:inline-flex;align-items:center;height:100%;padding:0 13px;'
    +   'color:#a89870;text-decoration:none;border-bottom:2px solid transparent;'
    +   'transition:color .15s ease,border-color .15s ease,background .15s ease;'
    + '}'
    + '.op-nav-sub a.op-nav-subtab:hover{color:#fdf4d8;background:rgba(255,255,255,.04)}'
    + '.op-nav-sub a.op-nav-subtab.active{color:#fdf4d8;border-bottom-color:#d4a843}'
    // Site-wide search box + mode toggle + results dropdown
    + '.op-nav-search-wrap{position:relative;display:flex;align-items:center;gap:6px;margin-left:10px}'
    + '.op-nav-mode-toggle{display:inline-flex;border:1px solid #2a2a35;border-radius:6px;overflow:hidden}'
    + '.op-nav-mode-toggle button{background:transparent;border:0;color:#9aa0b4;font:inherit;font-size:11px;padding:3px 8px;cursor:pointer;line-height:1.4}'
    + '.op-nav-mode-toggle button.active{background:#7c5cbf;color:#fff}'
    + '.op-nav-search{background:#15151d;border:1px solid #2a2a35;border-radius:6px;color:#e6edf3;font:inherit;font-size:12px;padding:5px 9px;width:190px;outline:none;transition:width .15s ease,border-color .15s ease}'
    + '.op-nav-search:focus{border-color:#7c5cbf;width:240px}'
    + '.op-nav-search::placeholder{color:#6b7180}'
    + '.op-nav-dropdown{position:absolute;top:34px;right:0;width:340px;max-height:60vh;overflow-y:auto;background:#0d0d14;border:1px solid #2a2a35;border-radius:8px;box-shadow:0 10px 30px rgba(0,0,0,.5);display:none;z-index:10000;padding:4px 0}'
    + '.op-nav-dropdown.open{display:block}'
    + '.op-nav-dd-group{padding:4px 0}'
    + '.op-nav-dd-grouphdr{color:#6b7180;font-size:9px;letter-spacing:.14em;text-transform:uppercase;padding:4px 12px 2px;font-weight:700}'
    + '.op-nav-dd-row{display:flex;flex-direction:column;gap:1px;padding:6px 12px;cursor:pointer;border-left:2px solid transparent}'
    + '.op-nav-dd-row:hover,.op-nav-dd-row.active{background:rgba(124,92,191,.16);border-left-color:#7c5cbf}'
    + '.op-nav-dd-title{color:#e6edf3;font-size:13px}'
    + '.op-nav-dd-sub{color:#8a90a0;font-size:11px}'
    + '.op-nav-dd-ext{color:#d4a843;font-size:10px;margin-left:4px}'
    + '.op-nav-dd-empty{color:#6b7180;font-size:12px;padding:10px 12px}'
    + '@keyframes opGotoFlash{0%,100%{box-shadow:0 0 0 0 rgba(124,92,191,0)}15%{box-shadow:0 0 0 3px rgba(124,92,191,.7)}}'
    + '.op-goto-flash{border-radius:6px;animation:opGotoFlash 1.6s ease}'
    + '@media(max-width:520px){'
    +   '.op-nav-bar{padding:0 8px;font-size:12px;height:38px}'
    +   '.op-nav-bar .op-nav-brand{display:none}'
    +   '.op-nav-bar a.op-nav-tab{padding:0 10px}'
    +   '.op-nav-sub{padding:0 8px;height:34px}'
    +   '.op-nav-sub .op-nav-sub-brand{display:none}'
    +   '.op-nav-sub a.op-nav-subtab{padding:0 10px}'
    +   '.op-nav-search{width:120px}'
    +   '.op-nav-search:focus{width:150px}'
    +   '.op-nav-dropdown{width:86vw;right:-8px}'
    + '}'
    // "More ▾" overflow dropdown
    + '.op-nav-more{position:relative;display:inline-flex;align-items:center}'
    + '.op-nav-more-btn{cursor:pointer;background:transparent;border:0;font:inherit}'
    + '.op-nav-more-menu{position:absolute;top:100%;left:0;min-width:180px;'
    +   'background:#0d0d14;border:1px solid #2a2a35;border-radius:8px;'
    +   'box-shadow:0 10px 30px rgba(0,0,0,.5);padding:4px 0;display:none;z-index:10000}'
    + '.op-nav-more.open .op-nav-more-menu{display:block}'
    + '.op-nav-more-menu a{display:block;padding:7px 14px;color:#9aa0b4;text-decoration:none;border-bottom:0}'
    + '.op-nav-more-menu a:hover{background:rgba(124,92,191,.16);color:#e6edf3}'
    // light/dark toggle button in the bar
    + '.op-nav-theme{background:transparent;border:1px solid #2a2a35;border-radius:6px;'
    +   'color:#9aa0b4;font:inherit;font-size:13px;line-height:1;padding:4px 8px;margin-right:8px;'
    +   'cursor:pointer;transition:color .15s ease,border-color .15s ease}'
    + '.op-nav-theme:hover{color:#e6edf3;border-color:#7c5cbf}'
    // Light-palette override (superset of the funnel/console/dashboard var names).
    // Distinct id from theme-toggle.js's 'rm-theme-style' so dual-load pages stay valid;
    // op-nav's lives in <body> (after any head copy) so its superset wins the cascade.
    + '</style>'
    + '<style id="op-nav-theme-style">'
    + ':root[data-theme="light"]{'
    +   '--bg:#FBF8F3;--bg-2:#F4ECDE;--surface:#FFFFFF;--surface-2:#F4ECDE;--surface2:#F4ECDE;'
    +   '--border:#E2D9C9;--rule:#E2D9C9;--hair:#E2D9C9;--cream:#1E2A2A;'
    +   '--muted:#5F6B6B;--dim:#5F6B6B;--gray:#5F6B6B;--ink:#1E2A2A;--ink-2:#5F6B6B;--ink-3:#7A8585;'
    +   '--gold:#B08A3E;--gold-soft:#EFE4C8;--green:#2D7A6A;--panel:#FFFFFF;--panel-2:#F4ECDE;'
    +   '--text:#1E2A2A;--text-muted:#5F6B6B;--accent:#B08A3E;--accent2:#2D7A6A;'
    + '}'
    + '</style>';

  var bar = '<nav class="op-nav-bar" role="navigation" aria-label="Glen ops">'
    + '<span class="op-nav-brand">GLEN <b>·</b> OPS</span>';
  for (var i = 0; i < tabs.length; i++) {
    var t = tabs[i];
    var cls = (t.id === active) ? "op-nav-tab active" : "op-nav-tab";
    bar += '<a class="' + cls + '" data-id="' + t.id + '" href="' + t.href + '">' + t.label + '</a>';
  }
  bar += '<span class="op-nav-more" id="op-nav-more-top" style="display:none">'
    + '<button type="button" class="op-nav-tab op-nav-more-btn">More ▾</button>'
    + '<span class="op-nav-more-menu"></span></span>';
  bar += '<span class="op-nav-spacer"></span>';
  if (!effKey) {
    bar += '<span class="op-nav-key-warn">no ?key — paste &amp; reload</span>';
  }
  bar += '<button type="button" class="op-nav-theme" id="op-nav-theme" title="Light / Dark" aria-label="Toggle light or dark theme">☀</button>';
  bar += '<span class="op-nav-search-wrap">'
    + '<span class="op-nav-mode-toggle" id="op-nav-mode">'
    +   '<button type="button" data-mode="pages">Pages</button>'
    +   '<button type="button" data-mode="records">Records</button>'
    + '</span>'
    + '<input class="op-nav-search" id="op-nav-search" type="text" autocomplete="off" spellcheck="false" placeholder="Search…" aria-label="Search console">'
    + '<div class="op-nav-dropdown" id="op-nav-dropdown" role="listbox"></div>'
    + '</span>';
  bar += '</nav>';

  if (active === "bos") {
    bar += '<nav class="op-nav-sub" role="navigation" aria-label="Business OS modules">'
      + '<span class="op-nav-sub-brand">Business OS</span>';
    for (var j = 0; j < bosMods.length; j++) {
      var m = bosMods[j];
      var scls = (m.id === sub) ? "op-nav-subtab active" : "op-nav-subtab";
      bar += '<a class="' + scls + '" data-id="' + m.id + '" href="' + m.href + '">' + m.label + '</a>';
    }
    bar += '<span class="op-nav-more" id="op-nav-more-bos" style="display:none">'
      + '<button type="button" class="op-nav-subtab op-nav-more-btn">More ▾</button>'
      + '<span class="op-nav-more-menu"></span></span>';
    bar += '</nav>';
  }

  if (active === "settings") {
    var setMods = [
      { id: "pricing",  label: "Pricing",         href: "/console/pricing-settings" + qs },
      { id: "shipping", label: "Shipping-config",  href: "/admin/shipping" + qs },
      { id: "tax",      label: "Tax",              href: "/admin/tax" + qs },
      { id: "writemac", label: "Write-Mac",        href: "/console/settings" + qs }
    ];
    bar += '<nav class="op-nav-sub" role="navigation" aria-label="Settings">'
      + '<span class="op-nav-sub-brand">Settings</span>';
    for (var s = 0; s < setMods.length; s++) {
      var sm = setMods[s];
      var smcls = (sm.id === sub) ? "op-nav-subtab active" : "op-nav-subtab";
      bar += '<a class="' + smcls + '" data-id="' + sm.id + '" href="' + sm.href + '">' + sm.label + '</a>';
    }
    bar += '</nav>';
  }

  document.write(styles + bar);

  // ── Role-aware nav: reorganize the rendered bar per the caller's nav profile ──
  // primary = stays on the bar; everything else moves into the matching "More ▾".
  var NAV_PROFILES = {
    glen: {
      tabs: ["dashboard","console","bos","projects","inbox","settings","funnel"],
      bos:  ["orders","client-orders","money","crm","products","biofield","pages",
             "biofield-reveals","biofield-intake",
             "approvals","shipping","neworder"]
    },
    rae: {
      tabs: ["dashboard","console","bos","inbox"],
      bos:  ["orders","client-orders","money","crm","approvals","shipping","neworder"]
    }
  };

  function reorg(wrapId, linkSelector, primaryIds) {
    var wrap = document.getElementById(wrapId);
    if (!wrap) return;
    var menu = wrap.querySelector(".op-nav-more-menu");
    var bar = wrap.parentNode;
    var moved = 0;
    bar.querySelectorAll(linkSelector).forEach(function (a) {
      var id = a.getAttribute("data-id");
      if (!id) return;
      if (primaryIds.indexOf(id) === -1) {
        a.classList.remove("op-nav-tab", "op-nav-subtab");
        menu.appendChild(a);
        moved++;
      }
    });
    wrap.style.display = (moved || menu.querySelectorAll("a").length) ? "inline-flex" : "none";
  }

  function applyNavProfile(navName) {
    var prof = NAV_PROFILES[navName] || NAV_PROFILES.glen;
    reorg("op-nav-more-top", ".op-nav-bar > a.op-nav-tab", prof.tabs);
    reorg("op-nav-more-bos", ".op-nav-sub > a.op-nav-subtab", prof.bos);
  }

  // Toggle "More" dropdowns (event delegation; survives reorg).
  document.addEventListener("click", function (e) {
    var btn = e.target.closest && e.target.closest(".op-nav-more-btn");
    document.querySelectorAll(".op-nav-more.open").forEach(function (w) {
      if (!btn || w !== btn.parentNode) w.classList.remove("open");
    });
    if (btn) { btn.parentNode.classList.toggle("open"); e.preventDefault(); }
  });

  // Render instantly from the cached profile (no flash on repeat visits),
  // then revalidate via /api/me. Owner-safe: any failure leaves the full bar.
  var NAV_CACHE_KEY = "op_nav_profile";
  try {
    var cached = localStorage.getItem(NAV_CACHE_KEY);
    if (cached === "rae" || cached === "glen") applyNavProfile(cached);
  } catch (e) {}

  fetch("/api/me" + (effKey ? "?key=" + encodeURIComponent(effKey) : ""),
        { headers: effKey ? { "X-Console-Key": effKey } : {} })
    .then(function (r) { return r.json(); })
    .then(function (me) {
      var navName = (me && me.nav === "rae") ? "rae" : "glen";  // only 'rae' streamlines
      try { localStorage.setItem(NAV_CACHE_KEY, navName); } catch (e) {}
      applyNavProfile(navName);
    })
    .catch(function () { /* leave full bar */ });

  // Wire the light/dark toggle. Reflects the active theme, persists to localStorage,
  // and live-syncs across same-origin documents via the 'storage' event.
  (function () {
    var themeBtn = document.getElementById('op-nav-theme');
    if (!themeBtn) return;
    function relabelTheme() {
      themeBtn.textContent = document.documentElement.getAttribute('data-theme') === 'light' ? '☾' : '☀';
    }
    relabelTheme();
    themeBtn.addEventListener('click', function () {
      var next = document.documentElement.getAttribute('data-theme') === 'light' ? 'dark' : 'light';
      rmApplyTheme(next);
      try { localStorage.setItem('rm-theme', next); } catch (e) {}
      relabelTheme();
    });
    window.addEventListener('storage', function (e) {
      if (e.key !== 'rm-theme') return;
      rmApplyTheme(e.newValue);
      relabelTheme();
    });
  })();

  // ---------------------------------------------------------------------------
  // Site-wide search — one box, two modes.
  //   Pages   : navigate the console's own surface (static destination index).
  //   Records : look up a specific person / product / order (/api/console/search).
  // The choice is remembered in localStorage; both render into the same dropdown.
  // ---------------------------------------------------------------------------
  var SEARCH_MODE_KEY = "console_search_mode";
  var searchMode = "pages";
  try { searchMode = localStorage.getItem(SEARCH_MODE_KEY) || "pages"; } catch (e) {}
  if (searchMode !== "records") searchMode = "pages";

  var pagesIndex = null;          // cached destination catalog
  var dbTimer = null;             // input debounce
  var activeIdx = -1;             // keyboard-highlighted row
  var flatResults = [];           // flat list backing keyboard nav + clicks

  function withKey(u) {
    if (!effKey) return u;
    if (/^https?:\/\//i.test(u)) return u;   // external link — leave untouched
    return u + (u.indexOf("?") >= 0 ? "&" : "?") + "key=" + encodeURIComponent(effKey);
  }
  function esc(s) {
    return String(s == null ? "" : s).replace(/[&<>"]/g, function (c) {
      return { "&": "&amp;", "<": "&lt;", ">": "&gt;", "\"": "&quot;" }[c];
    });
  }

  function loadPagesIndex(cb) {
    if (pagesIndex) { cb(pagesIndex); return; }
    fetch("/static/console-search-index.json", { cache: "no-cache" })
      .then(function (r) { if (!r.ok) throw 0; return r.json(); })
      .then(function (d) { pagesIndex = Array.isArray(d) ? d : []; cb(pagesIndex); })
      .catch(function () { cb(null); });
  }

  function rankPages(q) {
    q = q.toLowerCase();
    var out = [];
    for (var i = 0; i < pagesIndex.length; i++) {
      var e = pagesIndex[i];
      var title = (e.title || "").toLowerCase();
      var page = (e.page || "").toLowerCase();
      var kw = (e.keywords || []).join(" ").toLowerCase();
      var score = -1;
      if (title.indexOf(q) === 0) score = 0;
      else if (title.indexOf(q) >= 0) score = 1;
      else if (page.indexOf(q) >= 0) score = 2;
      else if (kw.indexOf(q) >= 0) score = 3;
      if (score >= 0) out.push({ e: e, s: score });
    }
    out.sort(function (a, b) { return a.s - b.s; });
    return out.slice(0, 12).map(function (x) { return x.e; });
  }

  // Both engines resolve to the same shape: [{label, items:[{title,subtitle,url,goto,external}]}]
  function pagesGroups(q, cb) {
    loadPagesIndex(function (idx) {
      if (!idx) { cb(null); return; }
      var byPage = {}, order = [];
      rankPages(q).forEach(function (e) {
        var g = e.page || "Other";
        if (!byPage[g]) { byPage[g] = []; order.push(g); }
        byPage[g].push({
          title: e.title, subtitle: e.page, url: e.url,
          goto: e.goto || "", external: !!e.external
        });
      });
      cb(order.map(function (g) { return { label: g, items: byPage[g] }; }));
    });
  }

  function recordsGroups(q, cb) {
    fetch("/api/console/search?q=" + encodeURIComponent(q), {
      headers: effKey ? { "X-Console-Key": effKey } : {}
    })
      .then(function (r) {
        if (r.status === 401) { cb("AUTH"); return null; }
        if (!r.ok) throw 0;
        return r.json();
      })
      .then(function (d) {
        if (!d) return;
        var groups = [];
        [["people", "People"], ["products", "Products"], ["orders", "Orders"]].forEach(function (p) {
          var arr = d[p[0]] || [];
          if (arr.length) {
            groups.push({ label: p[1], items: arr.map(function (it) {
              return { title: it.title, subtitle: it.subtitle || "", url: it.url, goto: "", external: false };
            }) });
          }
        });
        cb(groups);
      })
      .catch(function () { cb(null); });
  }

  function renderDropdown(groups, dd) {
    flatResults = []; activeIdx = -1;
    if (groups === "AUTH") { dd.innerHTML = '<div class="op-nav-dd-empty">Enter your console key to search records.</div>'; dd.classList.add("open"); return; }
    if (groups === null) { dd.innerHTML = '<div class="op-nav-dd-empty">Search unavailable.</div>'; dd.classList.add("open"); return; }
    if (!groups.length) { dd.innerHTML = '<div class="op-nav-dd-empty">No matches.</div>'; dd.classList.add("open"); return; }
    var html = "";
    groups.forEach(function (g) {
      html += '<div class="op-nav-dd-group"><div class="op-nav-dd-grouphdr">' + esc(g.label) + '</div>';
      g.items.forEach(function (it) {
        var idx = flatResults.length;
        flatResults.push(it);
        html += '<div class="op-nav-dd-row" data-idx="' + idx + '">'
          + '<span class="op-nav-dd-title">' + esc(it.title) + (it.external ? '<span class="op-nav-dd-ext">↗</span>' : '') + '</span>'
          + (it.subtitle ? '<span class="op-nav-dd-sub">' + esc(it.subtitle) + '</span>' : '')
          + '</div>';
      });
      html += '</div>';
    });
    dd.innerHTML = html;
    dd.classList.add("open");
  }

  function navigateItem(it) {
    if (!it) return;
    if (it.external) { window.open(it.url, "_blank", "noopener"); return; }
    var target = withKey(it.url);
    if (it.goto) target += "#goto-" + encodeURIComponent(it.goto);
    window.location.href = target;
  }

  function setupSearch() {
    var input = document.getElementById("op-nav-search");
    var dd = document.getElementById("op-nav-dropdown");
    var modeWrap = document.getElementById("op-nav-mode");
    if (!input || !dd || !modeWrap) return;

    function paintMode() {
      var btns = modeWrap.getElementsByTagName("button");
      for (var i = 0; i < btns.length; i++) {
        btns[i].className = (btns[i].getAttribute("data-mode") === searchMode) ? "active" : "";
      }
      input.placeholder = searchMode === "records" ? "Find a person, product, order…" : "Search pages…";
    }
    paintMode();

    function runSearch() {
      var q = input.value.trim();
      if (!q) { dd.classList.remove("open"); dd.innerHTML = ""; flatResults = []; activeIdx = -1; return; }
      if (searchMode === "records") recordsGroups(q, function (g) { renderDropdown(g, dd); });
      else pagesGroups(q, function (g) { renderDropdown(g, dd); });
    }

    function setActive(n) {
      var rows = dd.getElementsByClassName("op-nav-dd-row");
      if (!rows.length) return;
      if (n < 0) n = rows.length - 1;
      if (n >= rows.length) n = 0;
      activeIdx = n;
      for (var i = 0; i < rows.length; i++) rows[i].className = (i === n) ? "op-nav-dd-row active" : "op-nav-dd-row";
      rows[n].scrollIntoView({ block: "nearest" });
    }

    input.addEventListener("input", function () {
      if (dbTimer) clearTimeout(dbTimer);
      dbTimer = setTimeout(runSearch, 120);
    });
    input.addEventListener("focus", function () { if (input.value.trim()) runSearch(); });
    input.addEventListener("keydown", function (ev) {
      if (ev.key === "ArrowDown") { ev.preventDefault(); setActive(activeIdx + 1); }
      else if (ev.key === "ArrowUp") { ev.preventDefault(); setActive(activeIdx - 1); }
      else if (ev.key === "Enter") {
        ev.preventDefault();
        navigateItem(activeIdx >= 0 ? flatResults[activeIdx] : flatResults[0]);
      } else if (ev.key === "Escape") {
        dd.classList.remove("open"); input.blur();
      }
    });

    dd.addEventListener("mousedown", function (ev) {
      var row = ev.target.closest ? ev.target.closest(".op-nav-dd-row") : null;
      if (!row) return;
      ev.preventDefault();
      navigateItem(flatResults[parseInt(row.getAttribute("data-idx"), 10)]);
    });

    modeWrap.addEventListener("click", function (ev) {
      var b = ev.target;
      while (b && b.tagName !== "BUTTON") b = b.parentNode;
      if (!b || !b.getAttribute("data-mode")) return;
      searchMode = b.getAttribute("data-mode");
      try { localStorage.setItem(SEARCH_MODE_KEY, searchMode); } catch (e) {}
      paintMode();
      input.focus();
      runSearch();
    });

    document.addEventListener("click", function (ev) {
      var wrap = document.querySelector(".op-nav-search-wrap");
      if (wrap && !wrap.contains(ev.target)) dd.classList.remove("open");
    });

    // Global shortcuts: Cmd/Ctrl-K or "/" focuses the box (when not already typing).
    document.addEventListener("keydown", function (ev) {
      var k = (ev.metaKey || ev.ctrlKey) && (ev.key === "k" || ev.key === "K");
      var tag = ((document.activeElement || {}).tagName || "");
      var editing = /^(INPUT|TEXTAREA|SELECT)$/.test(tag) || (document.activeElement && document.activeElement.isContentEditable);
      var slash = ev.key === "/" && !editing;
      if (k || slash) { ev.preventDefault(); input.focus(); input.select(); }
    });
  }

  // Section targeting: arriving with #goto-<id> activates/scrolls that section.
  // Pages opt in by adding data-goto="<id>" to a tab button or section anchor.
  function runHashRouter() {
    var m = (location.hash || "").match(/^#goto-(.+)$/);
    if (!m) return;
    var id = decodeURIComponent(m[1]).replace(/"/g, "");
    var el = document.querySelector('[data-goto="' + id + '"]');
    if (!el) return;
    var clickable = el.tagName === "BUTTON" || el.tagName === "A" || el.hasAttribute("onclick") || el.getAttribute("role") === "tab";
    if (clickable) { try { el.click(); } catch (e) {} }
    setTimeout(function () {
      try { el.scrollIntoView({ behavior: "smooth", block: "center" }); }
      catch (e) { try { el.scrollIntoView(); } catch (e2) {} }
      el.classList.add("op-goto-flash");
      setTimeout(function () { el.classList.remove("op-goto-flash"); }, 1600);
    }, 80);
  }

  // Elements written above exist synchronously; wire now, else on DOMContentLoaded.
  if (document.getElementById("op-nav-search")) setupSearch();
  else document.addEventListener("DOMContentLoaded", setupSearch);

  if (document.readyState === "loading") document.addEventListener("DOMContentLoaded", runHashRouter);
  else runHashRouter();
})();
