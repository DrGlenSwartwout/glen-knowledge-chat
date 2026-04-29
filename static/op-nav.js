/* op-nav.js — shared sticky tab bar across Glen's internal control panels.
 *
 * Usage: drop one synchronous script tag immediately after <body> on each page:
 *   <script src="/static/op-nav.js" data-active="dashboard"></script>
 *
 * Valid data-active values: "dashboard" | "console" | "shipping" | "orders"
 *
 * The bar:
 *   - Renders synchronously via document.write so there is no flash
 *   - Reads the current URL's ?key= and preserves it across all internal links
 *   - Highlights the active tab from the data-active attribute
 *   - Sticks to the top of the viewport while scrolling
 *   - Uses neutral dark styling that sits on top of all three page palettes
 *     (editorial-gold dashboard, ops-blue console, purple shipping/orders)
 */
(function () {
  var script = document.currentScript;
  var active = (script && script.dataset && script.dataset.active) || "";
  var key = new URLSearchParams(location.search).get("key") || "";
  var qs = key ? ("?key=" + encodeURIComponent(key)) : "";

  var tabs = [
    { id: "dashboard", label: "Dashboard", href: "/dashboard" + qs },
    { id: "console",   label: "Console",   href: "/console"   + qs },
    { id: "shipping",  label: "Shipping",  href: "/admin/shipping" + qs },
    { id: "orders",    label: "Orders",    href: "/orders/new"     + qs },
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
    + '@media(max-width:520px){'
    +   '.op-nav-bar{padding:0 8px;font-size:12px;height:38px}'
    +   '.op-nav-bar .op-nav-brand{display:none}'
    +   '.op-nav-bar a.op-nav-tab{padding:0 10px}'
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
  if (!key) {
    bar += '<span class="op-nav-key-warn">no ?key — paste &amp; reload</span>';
  }
  bar += '</nav>';

  document.write(styles + bar);
})();
