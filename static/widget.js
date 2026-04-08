/* Glen Swartwout Knowledge Assistant — Embeddable Widget
 * Add to any webpage with one line:
 *   <script src="https://YOUR-DOMAIN/widget.js"></script>
 *
 * Optional data attributes on the script tag:
 *   data-position="bottom-right"  (or bottom-left)
 *   data-color="#58a6ff"          (button color)
 *   data-label="Ask Glen"         (button tooltip)
 *   data-title="Glen's Assistant" (panel header title)
 */
(function () {
  'use strict';

  // ── Determine server origin from this script's src ───────────────────────
  var scriptEl = document.currentScript ||
    (function () {
      var scripts = document.getElementsByTagName('script');
      return scripts[scripts.length - 1];
    })();

  var scriptSrc = scriptEl ? scriptEl.src : '';
  var serverOrigin = scriptSrc
    ? scriptSrc.replace(/\/widget\.js.*$/, '')
    : window.location.origin;

  // ── Config from data attributes ──────────────────────────────────────────
  var position = (scriptEl && scriptEl.dataset.position) || 'bottom-right';
  var color    = (scriptEl && scriptEl.dataset.color)    || '#58a6ff';
  var label    = (scriptEl && scriptEl.dataset.label)    || 'Ask Dr. Glen';
  var title    = (scriptEl && scriptEl.dataset.title)    || "Glen's Knowledge Assistant";

  var isRight  = position !== 'bottom-left';
  var side     = isRight ? 'right' : 'left';
  var panelW   = 380;
  var panelH   = 560;
  var gap      = 20;

  // ── Inject styles ────────────────────────────────────────────────────────
  var css = `
    #gsw-widget-btn {
      position: fixed;
      ${side}: ${gap}px;
      bottom: ${gap}px;
      width: 56px; height: 56px;
      border-radius: 50%;
      background: ${color};
      border: none;
      cursor: pointer;
      z-index: 2147483646;
      display: flex; align-items: center; justify-content: center;
      box-shadow: 0 4px 20px rgba(0,0,0,0.35);
      transition: transform 0.2s, box-shadow 0.2s;
    }
    #gsw-widget-btn:hover {
      transform: scale(1.08);
      box-shadow: 0 6px 28px rgba(0,0,0,0.45);
    }
    #gsw-widget-btn svg { width: 26px; height: 26px; pointer-events: none; }
    #gsw-widget-btn .gsw-close-icon { display: none; }
    #gsw-widget-btn.open .gsw-chat-icon { display: none; }
    #gsw-widget-btn.open .gsw-close-icon { display: block; }

    #gsw-widget-tooltip {
      position: fixed;
      ${side}: ${gap + 64}px;
      bottom: ${gap + 16}px;
      background: rgba(13,17,23,0.92);
      color: #e6edf3;
      font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
      font-size: 13px;
      padding: 5px 10px;
      border-radius: 6px;
      pointer-events: none;
      opacity: 0;
      transition: opacity 0.2s;
      z-index: 2147483645;
      white-space: nowrap;
    }
    #gsw-widget-btn:hover + #gsw-widget-tooltip { opacity: 1; }

    #gsw-widget-panel {
      position: fixed;
      ${side}: ${gap}px;
      bottom: ${gap + 56 + 12}px;
      width: ${panelW}px;
      height: ${panelH}px;
      border-radius: 14px;
      overflow: hidden;
      box-shadow: 0 12px 48px rgba(0,0,0,0.5);
      z-index: 2147483645;
      border: 1px solid rgba(255,255,255,0.1);
      transform: scale(0.92) translateY(20px);
      transform-origin: ${isRight ? 'bottom right' : 'bottom left'};
      opacity: 0;
      pointer-events: none;
      transition: transform 0.25s cubic-bezier(0.34,1.56,0.64,1), opacity 0.2s;
    }
    #gsw-widget-panel.open {
      transform: scale(1) translateY(0);
      opacity: 1;
      pointer-events: all;
    }
    #gsw-widget-panel iframe {
      width: 100%; height: 100%;
      border: none; display: block;
    }

    @media (max-width: 450px) {
      #gsw-widget-panel {
        ${side}: 0 !important;
        bottom: 0 !important;
        width: 100vw !important;
        height: 100vh !important;
        border-radius: 0 !important;
      }
    }
  `;

  var styleEl = document.createElement('style');
  styleEl.textContent = css;
  document.head.appendChild(styleEl);

  // ── Chat icon SVG ────────────────────────────────────────────────────────
  var chatIconSVG = `<svg class="gsw-chat-icon" viewBox="0 0 24 24" fill="none"
      stroke="#000" stroke-width="2.2" stroke-linecap="round" stroke-linejoin="round">
    <path d="M21 15a2 2 0 0 1-2 2H7l-4 4V5a2 2 0 0 1 2-2h14a2 2 0 0 1 2 2z"/>
  </svg>`;

  var closeIconSVG = `<svg class="gsw-close-icon" viewBox="0 0 24 24" fill="none"
      stroke="#000" stroke-width="2.5" stroke-linecap="round">
    <line x1="18" y1="6" x2="6" y2="18"/><line x1="6" y1="6" x2="18" y2="18"/>
  </svg>`;

  // ── Build DOM ────────────────────────────────────────────────────────────
  var btn = document.createElement('button');
  btn.id = 'gsw-widget-btn';
  btn.setAttribute('aria-label', label);
  btn.innerHTML = chatIconSVG + closeIconSVG;

  var tooltip = document.createElement('div');
  tooltip.id = 'gsw-widget-tooltip';
  tooltip.textContent = label;

  var panel = document.createElement('div');
  panel.id = 'gsw-widget-panel';

  document.body.appendChild(btn);
  document.body.appendChild(tooltip);
  document.body.appendChild(panel);

  // ── Iframe (lazy — created only on first open) ───────────────────────────
  var iframeLoaded = false;

  function loadIframe() {
    if (iframeLoaded) return;
    iframeLoaded = true;
    var iframe = document.createElement('iframe');
    iframe.src = serverOrigin + '/embed';
    iframe.title = title;
    iframe.allow = 'clipboard-write';
    panel.appendChild(iframe);
  }

  // ── Toggle ───────────────────────────────────────────────────────────────
  var isOpen = false;

  btn.addEventListener('click', function () {
    isOpen = !isOpen;
    if (isOpen) {
      loadIframe();
      panel.classList.add('open');
      btn.classList.add('open');
      btn.setAttribute('aria-label', 'Close');
      tooltip.textContent = 'Close';
    } else {
      panel.classList.remove('open');
      btn.classList.remove('open');
      btn.setAttribute('aria-label', label);
      tooltip.textContent = label;
    }
  });

  // Close on Escape
  document.addEventListener('keydown', function (e) {
    if (e.key === 'Escape' && isOpen) btn.click();
  });

})();
