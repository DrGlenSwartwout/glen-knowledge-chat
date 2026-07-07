/* ask-guide.js — floating "Ask & Guide" panel, injected on every console page by op-nav.js.
   Slice 1: Guide only (logs a page-tagged todo via POST /api/guide). Ask tab is disabled. */
(function () {
  if (typeof document === "undefined") return;
  if (window.__askGuideLoaded) return; window.__askGuideLoaded = true;

  var nav = document.querySelector('script[src*="op-nav.js"]');
  var active = (nav && nav.dataset && nav.dataset.active) || "unknown";
  var sub = (nav && nav.dataset && nav.dataset.sub) || "";
  var key = "";
  try { key = localStorage.getItem("console_key") || ""; } catch (e) {}

  var reduce = window.matchMedia && window.matchMedia("(prefers-reduced-motion: reduce)").matches;

  var css = document.createElement("style");
  css.textContent =
    ".ag-pill{position:fixed;right:16px;bottom:16px;z-index:99998;background:#0f1f16;color:#fdf4d8;" +
      "border:1px solid #d4a843;border-radius:20px;padding:9px 15px;font:600 13px/1 -apple-system,BlinkMacSystemFont,'Segoe UI',sans-serif;" +
      "cursor:pointer;box-shadow:0 4px 14px rgba(0,0,0,.4)}" +
    ".ag-pill:focus-visible{outline:2px solid #d4a843;outline-offset:2px}" +
    ".ag-drawer{position:fixed;right:16px;bottom:60px;z-index:99999;width:360px;max-width:calc(100vw - 32px);" +
      "background:#0f1f16;color:#fdf4d8;border:1px solid #21472d;border-radius:12px;box-shadow:0 12px 40px rgba(0,0,0,.55);" +
      "font-family:-apple-system,BlinkMacSystemFont,'Segoe UI',sans-serif;display:none;overflow:hidden}" +
    ".ag-drawer.open{display:block}" + (reduce ? "" : ".ag-drawer{transition:opacity .12s ease}") +
    ".ag-hd{display:flex;align-items:center;justify-content:space-between;padding:11px 13px;border-bottom:1px solid #21472d}" +
    ".ag-hd b{font-size:13px}.ag-ctx{font-size:11px;color:#9db29f}" +
    ".ag-x{background:none;border:0;color:#9db29f;font-size:16px;cursor:pointer;line-height:1}" +
    ".ag-tabs{display:flex;gap:4px;padding:8px 13px 0}.ag-tab{font-size:12px;font-weight:600;color:#9db29f;background:none;" +
      "border:0;border-bottom:2px solid transparent;padding:5px 8px;cursor:pointer}.ag-tab.on{color:#e7c877;border-color:#d4a843}" +
    ".ag-tab[disabled]{opacity:.5;cursor:not-allowed}" +
    ".ag-body{padding:11px 13px 13px}.ag-body textarea{width:100%;min-height:90px;box-sizing:border-box;background:#0a150d;" +
      "color:#fdf4d8;border:1px solid #21472d;border-radius:8px;padding:8px;font:inherit;font-size:13px;resize:vertical}" +
    ".ag-send{margin-top:9px;background:#d4a843;color:#0a150d;border:0;border-radius:7px;padding:8px 14px;font:600 13px/1 inherit;cursor:pointer}" +
    ".ag-send[disabled]{opacity:.5;cursor:not-allowed}" +
    ".ag-msg{margin-top:8px;font-size:12px;min-height:16px}.ag-msg.ok{color:#7fd6a0}.ag-msg.err{color:#f0857d}" +
    ".ag-q{width:100%;box-sizing:border-box;background:#0a150d;color:#fdf4d8;border:1px solid #21472d;border-radius:8px;padding:8px;font:inherit;font-size:13px}" +
    ".ag-ask-send{margin-top:9px;background:#d4a843;color:#0a150d;border:0;border-radius:7px;padding:8px 14px;font:600 13px/1 inherit;cursor:pointer}" +
    ".ag-answer{margin-top:10px;font-size:13px;white-space:pre-wrap;line-height:1.45}" +
    ".ag-sources{margin-top:8px;font-size:11px;color:#9db29f}";
  document.head.appendChild(css);

  var ctx = sub ? (active + " › " + sub) : active;
  var wrap = document.createElement("div");
  wrap.innerHTML =
    '<button class="ag-pill" type="button" aria-haspopup="dialog" aria-expanded="false">✦ Ask &amp; Guide</button>' +
    '<div class="ag-drawer" role="dialog" aria-label="Ask and Guide">' +
      '<div class="ag-hd"><div><b>Ask &amp; Guide</b><div class="ag-ctx">Guidance for: ' + ctx + '</div></div>' +
        '<button class="ag-x" type="button" aria-label="Close">✕</button></div>' +
      '<div class="ag-tabs"><button class="ag-tab on" type="button" data-t="guide">Guide</button>' +
        '<button class="ag-tab" type="button" data-t="ask">Ask</button></div>' +
      '<div class="ag-body"><textarea placeholder="Describe a change you want on this page…" aria-label="Guidance"></textarea>' +
        '<button class="ag-send" type="button">Send to Projects</button><div class="ag-msg" role="status"></div></div>' +
      '<div class="ag-body ag-ask" style="display:none">' +
        '<input class="ag-q" type="text" placeholder="Ask how a system works…" aria-label="Ask a question">' +
        '<button class="ag-ask-send" type="button">Ask</button>' +
        '<div class="ag-answer" role="status"></div><div class="ag-sources"></div></div>' +
    '</div>';
  document.body.appendChild(wrap);

  var pill = wrap.querySelector(".ag-pill"), drawer = wrap.querySelector(".ag-drawer");
  var ta = wrap.querySelector("textarea"), send = wrap.querySelector(".ag-send"), msg = wrap.querySelector(".ag-msg");
  function setOpen(o) { drawer.classList.toggle("open", o); pill.setAttribute("aria-expanded", o ? "true" : "false"); if (o) ta.focus(); }
  pill.addEventListener("click", function () { setOpen(!drawer.classList.contains("open")); });
  wrap.querySelector(".ag-x").addEventListener("click", function () { setOpen(false); });
  document.addEventListener("keydown", function (e) { if (e.key === "Escape") setOpen(false); });

  send.addEventListener("click", function () {
    var text = (ta.value || "").trim();
    msg.className = "ag-msg";
    if (!text) { msg.textContent = "Type something first."; msg.className = "ag-msg err"; return; }
    send.disabled = true; msg.textContent = "Sending…";
    fetch("/api/guide" + (key ? "?key=" + encodeURIComponent(key) : ""), {
      method: "POST",
      headers: { "Content-Type": "application/json", "X-Console-Key": key },
      body: JSON.stringify({ text: text, active: active, sub: sub, url: location.pathname })
    }).then(function (r) { return r.json().catch(function () { return { ok: false }; }); })
      .then(function (d) {
        send.disabled = false;
        if (d && d.ok) { msg.textContent = "Sent to Projects ✓"; msg.className = "ag-msg ok"; ta.value = ""; setTimeout(function () { setOpen(false); msg.textContent = ""; }, 1100); }
        else { msg.textContent = (d && d.error === "unauthorized") ? "Not authorized — reload with your key." : "Couldn't send — try again."; msg.className = "ag-msg err"; }
      })
      .catch(function () { send.disabled = false; msg.textContent = "Couldn't send — try again."; msg.className = "ag-msg err"; });
  });

  // Ask tab: switch views + query /api/ask (Slice 2).
  var guideBody = wrap.querySelector(".ag-body:not(.ag-ask)"), askBody = wrap.querySelector(".ag-ask");
  wrap.querySelectorAll(".ag-tab").forEach(function (t) {
    t.addEventListener("click", function () {
      wrap.querySelectorAll(".ag-tab").forEach(function (x) { x.classList.remove("on"); });
      t.classList.add("on");
      var ask = t.getAttribute("data-t") === "ask";
      guideBody.style.display = ask ? "none" : "block";
      askBody.style.display = ask ? "block" : "none";
    });
  });
  var q = wrap.querySelector(".ag-q"), askSend = wrap.querySelector(".ag-ask-send");
  var ans = wrap.querySelector(".ag-answer"), srcs = wrap.querySelector(".ag-sources");
  function doAsk() {
    var text = (q.value || "").trim();
    if (!text) { ans.textContent = "Type a question first."; return; }
    askSend.disabled = true; ans.textContent = "Thinking…"; srcs.textContent = "";
    fetch("/api/ask" + (key ? "?key=" + encodeURIComponent(key) : ""), {
      method: "POST", headers: { "Content-Type": "application/json", "X-Console-Key": key },
      body: JSON.stringify({ question: text, active: active, sub: sub })
    }).then(function (r) { return r.json().catch(function () { return { ok: false }; }); })
      .then(function (d) {
        askSend.disabled = false;
        if (d && d.ok) {
          ans.textContent = d.answer || "";
          srcs.textContent = (d.sources && d.sources.length) ? ("Sources: " + d.sources.map(function (s) { return s.title || s.source; }).join(", ")) : "";
        } else { ans.textContent = "Ask is unavailable right now."; }
      })
      .catch(function () { askSend.disabled = false; ans.textContent = "Ask is unavailable right now."; });
  }
  askSend.addEventListener("click", doAsk);
  q.addEventListener("keydown", function (e) { if (e.key === "Enter") doAsk(); });
})();
