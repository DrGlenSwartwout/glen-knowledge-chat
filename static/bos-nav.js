/* bos-nav.js — secondary module bar for the Business OS console area.
 *
 * Renders directly UNDER the shared op-nav bar (op-nav.js) on every Business OS
 * page and lets Glen hop between the module boards without going back to the hub.
 *
 * Usage: drop one synchronous script tag immediately after the op-nav tag, e.g.
 *   <script src="/static/op-nav.js" data-active="bos"></script>
 *   <script src="/static/bos-nav.js" data-bos="finance"></script>
 *
 * Valid data-bos values: "home" | "orders" | "finance" | "crm" | "products"
 *
 * Like op-nav it renders synchronously via document.write (no flash) and
 * preserves the current ?key= across links. The BOS boards themselves gate on a
 * console key held in localStorage, so the ?key= is only carried for parity with
 * the rest of the ops nav — the boards ignore it.
 */
(function () {
  var script = document.currentScript;
  var active = (script && script.dataset && script.dataset.bos) || "";
  var key = new URLSearchParams(location.search).get("key") || "";
  var qs = key ? ("?key=" + encodeURIComponent(key)) : "";

  var mods = [
    { id: "home",     label: "Home",     href: "/console/home" + qs },
    { id: "orders",   label: "Orders",   href: "/console/orders" + qs },
    { id: "finance",  label: "Finance",  href: "/console/finance" + qs },
    { id: "crm",      label: "CRM",      href: "/console/crm" + qs },
    { id: "products", label: "Products", href: "/console/products" + qs },
  ];

  var styles = ''
    + '<style id="bos-nav-styles">'
    + '.bos-nav-bar{'
    +   'position:sticky;top:40px;z-index:9998;'
    +   'display:flex;align-items:center;gap:0;'
    +   'background:#0d1c12;border-bottom:1px solid #21472d;'
    +   'padding:0 14px;height:36px;'
    +   'font-family:-apple-system,BlinkMacSystemFont,"Segoe UI",sans-serif;'
    +   'font-size:13px;'
    + '}'
    + '.bos-nav-bar .bos-nav-brand{'
    +   'color:#a89870;letter-spacing:.16em;text-transform:uppercase;'
    +   'font-size:10px;font-weight:700;margin-right:16px;'
    +   'font-family:ui-monospace,"SF Mono",Menlo,Consolas,monospace;'
    + '}'
    + '.bos-nav-bar a.bos-nav-tab{'
    +   'display:inline-flex;align-items:center;'
    +   'height:100%;padding:0 13px;'
    +   'color:#a89870;text-decoration:none;'
    +   'border-bottom:2px solid transparent;'
    +   'transition:color .15s ease,border-color .15s ease,background .15s ease;'
    + '}'
    + '.bos-nav-bar a.bos-nav-tab:hover{color:#fdf4d8;background:rgba(255,255,255,.04)}'
    + '.bos-nav-bar a.bos-nav-tab.active{color:#fdf4d8;border-bottom-color:#d4a843}'
    + '@media(max-width:520px){'
    +   '.bos-nav-bar{padding:0 8px;height:34px}'
    +   '.bos-nav-bar .bos-nav-brand{display:none}'
    +   '.bos-nav-bar a.bos-nav-tab{padding:0 10px}'
    + '}'
    + '</style>';

  var bar = '<nav class="bos-nav-bar" role="navigation" aria-label="Business OS modules">'
    + '<span class="bos-nav-brand">Business OS</span>';
  for (var i = 0; i < mods.length; i++) {
    var m = mods[i];
    var cls = (m.id === active) ? "bos-nav-tab active" : "bos-nav-tab";
    bar += '<a class="' + cls + '" href="' + m.href + '">' + m.label + '</a>';
  }
  bar += '</nav>';

  document.write(styles + bar);
})();
