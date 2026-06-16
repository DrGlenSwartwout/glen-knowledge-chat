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
 * Valid data-sub values: "orders" | "finance" | "crm" | "products" | "shipping" | "neworder"
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
    { id: "finance",  label: "Finance",   href: "/console/finance" + qs },
    { id: "crm",      label: "CRM",       href: "/console/crm" + qs },
    { id: "products", label: "Products",  href: "/console/products" + qs },
    { id: "biofield", label: "Biofield",  href: "/console/biofield-portal" + qs },
    { id: "shipping", label: "Shipping",  href: "/admin/shipping" + qs },
    { id: "neworder", label: "New Order", href: "/orders/new" + qs },
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
    + '@media(max-width:520px){'
    +   '.op-nav-bar{padding:0 8px;font-size:12px;height:38px}'
    +   '.op-nav-bar .op-nav-brand{display:none}'
    +   '.op-nav-bar a.op-nav-tab{padding:0 10px}'
    +   '.op-nav-sub{padding:0 8px;height:34px}'
    +   '.op-nav-sub .op-nav-sub-brand{display:none}'
    +   '.op-nav-sub a.op-nav-subtab{padding:0 10px}'
    + '}'
    + '</style>';

  var bar = '<nav class="op-nav-bar" role="navigation" aria-label="Glen ops">'
    + '<span class="op-nav-brand">GLEN <b>·</b> OPS</span>';
  for (var i = 0; i < tabs.length; i++) {
    var t = tabs[i];
    var cls = (t.id === active) ? "op-nav-tab active" : "op-nav-tab";
    bar += '<a class="' + cls + '" href="' + t.href + '">' + t.label + '</a>';
  }
  bar += '<span class="op-nav-spacer"></span>';
  if (!effKey) {
    bar += '<span class="op-nav-key-warn">no ?key — paste &amp; reload</span>';
  }
  bar += '</nav>';

  if (active === "bos") {
    bar += '<nav class="op-nav-sub" role="navigation" aria-label="Business OS modules">'
      + '<span class="op-nav-sub-brand">Business OS</span>';
    for (var j = 0; j < bosMods.length; j++) {
      var m = bosMods[j];
      var scls = (m.id === sub) ? "op-nav-subtab active" : "op-nav-subtab";
      bar += '<a class="' + scls + '" href="' + m.href + '">' + m.label + '</a>';
    }
    bar += '</nav>';
  }

  document.write(styles + bar);
})();
