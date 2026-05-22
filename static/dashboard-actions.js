// dashboard-actions.js — per-item actions for the Command Center cards.
//
// Loaded after dashboard.html's main script. Reuses:
//   • localStorage["consoleKey"] for auth
//   • the global `fetchCard(card)` defined in dashboard.html
//
// Affordance: tap an `<li.item>` to expand a sibling `.action-panel`.
// One panel open per card at a time. Panels are lazy-built on first open.

(function(){
  "use strict";

  // ── helpers ───────────────────────────────────────────────────────────────
  const KEY  = () => localStorage.getItem("consoleKey") || "";
  const auth = () => ({"X-Console-Key": KEY()});
  const esc  = s  => String(s||"").replace(/[&<>"']/g,
    c => ({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[c]));

  async function call(method, url, body){
    const opts = {method, headers: {...auth()}, cache: "no-store"};
    if(body !== undefined){
      opts.headers["Content-Type"] = "application/json";
      opts.body = JSON.stringify(body);
    }
    const res = await fetch(url, opts);
    if(!res.ok){
      let detail = "HTTP " + res.status;
      try { const j = await res.json(); detail = j.error || j.message || detail; } catch(_){}
      throw new Error(detail);
    }
    const j = await res.json();
    return (j && typeof j === "object" && "ok" in j && "data" in j) ? j.data : j;
  }

  function dropItem(li){
    li.style.transition = "opacity .25s, transform .25s";
    li.style.opacity = "0";
    li.style.transform = "translateX(-12px)";
    setTimeout(() => {
      const card = li.closest(".card");
      li.remove();
      if(card && typeof window.fetchCard === "function") window.fetchCard(card);
    }, 320);
  }

  function fade(li){
    li.style.opacity = "0.5";
  }

  function toast(panel, msg, isErr){
    let t = panel.querySelector(".msg");
    if(!t){
      t = document.createElement("div");
      t.className = "msg";
      panel.appendChild(t);
    }
    t.textContent = msg;
    t.classList.toggle("err", !!isErr);
    t.hidden = false;
    clearTimeout(t._h);
    t._h = setTimeout(() => { t.hidden = true; }, isErr ? 4000 : 2400);
  }

  function busy(btn, on, label){
    btn.disabled = !!on;
    if(on){
      btn.dataset.orig = btn.textContent;
      btn.textContent = label || "…";
    } else if(btn.dataset.orig){
      btn.textContent = btn.dataset.orig;
      delete btn.dataset.orig;
    }
  }

  async function copyToClipboard(text){
    try { await navigator.clipboard.writeText(text); return true; }
    catch(_) { return false; }
  }

  // Async-action wrapper: runs fn, surfaces error as a toast, busy-locks btn.
  function bind(panel, sel, label, fn){
    panel.querySelectorAll(sel).forEach(btn => {
      btn.addEventListener("click", async () => {
        busy(btn, true, label || "Working…");
        try { await fn(btn); }
        catch(err) { toast(panel, "Error: " + err.message, true); }
        finally { busy(btn, false); }
      });
    });
  }

  // ── per-kind panel builders ───────────────────────────────────────────────

  function todoPanel(li){
    const id = li.dataset.id;
    const panel = li.querySelector(".action-panel");
    panel.innerHTML = `
      <div class="btn-row">
        <button data-act="done"     class="primary">✓ Done</button>
        <button data-act="draft">✉ Draft reply</button>
        <button data-act="focus">◎ Focus</button>
        <button data-act="rae">→ Rae</button>
        <button data-act="shaira">→ Shaira</button>
        <button data-act="blocked">⊘ Blocked</button>
        <button data-act="dismiss" class="danger">× Dismiss</button>
      </div>
      <div class="draft-wrap" hidden>
        <textarea placeholder="Optional guidance for Justus…" rows="2"></textarea>
        <div class="btn-row">
          <button data-act="draft-go" class="primary">Generate ✦</button>
          <button data-act="draft-cancel">Close</button>
        </div>
        <pre class="draft-out" hidden></pre>
      </div>
    `;
    bind(panel, '[data-act="done"]',    "Marking done…",
         async () => { await call("PATCH", `/api/todos/${id}`, {action:"done"}); dropItem(li); });
    bind(panel, '[data-act="dismiss"]', "Dismissing…",
         async () => { await call("DELETE", `/api/todos/${id}`);                  dropItem(li); });
    bind(panel, '[data-act="focus"]',   "Focusing…",
         async () => { await call("POST",  `/api/todos/${id}/focus`); toast(panel, "Focus session started"); });
    bind(panel, '[data-act="blocked"]', "Marking blocked…",
         async () => { await call("POST", `/api/todos/${id}/mark-blocked`,
                              {reason: "(flagged from dashboard)"}); fade(li); toast(panel, "Marked blocked"); });
    bind(panel, '[data-act="rae"]',     "→ Rae…",
         async () => { await call("PATCH", `/api/todos/${id}`, {action:"delegate", to:"rae"}); dropItem(li); });
    bind(panel, '[data-act="shaira"]',  "→ Shaira…",
         async () => { await call("PATCH", `/api/todos/${id}`, {action:"delegate", to:"shaira"}); dropItem(li); });

    const draftWrap = panel.querySelector(".draft-wrap");
    panel.querySelector('[data-act="draft"]').addEventListener("click", () => { draftWrap.hidden = false; });
    panel.querySelector('[data-act="draft-cancel"]').addEventListener("click", () => { draftWrap.hidden = true; });
    bind(panel, '[data-act="draft-go"]', "Generating…", async () => {
      const guidance = draftWrap.querySelector("textarea").value;
      const res = await call("POST", `/api/todos/${id}/draft-reply`, {guidance});
      const out = draftWrap.querySelector(".draft-out");
      out.textContent = res.draft || "(no draft returned)";
      out.hidden = false;
    });
  }

  function eventPanel(li){
    const id = li.dataset.id;
    const panel = li.querySelector(".action-panel");
    panel.innerHTML = `
      <div class="btn-row">
        <button data-act="hide">○ Hide</button>
        <button data-act="suppress">∞ Suppress series</button>
        <button data-act="alert">▲ Toggle alert</button>
        <button data-act="delete" class="danger">× Request delete</button>
      </div>
    `;
    bind(panel, '[data-act="hide"]',     "Hiding…",
         async () => { await call("PATCH", `/api/calendar/${id}`, {action:"hide"}); dropItem(li); });
    bind(panel, '[data-act="suppress"]', "Suppressing…",
         async () => { await call("POST",  `/api/calendar/${id}/suppress`); dropItem(li); });
    bind(panel, '[data-act="delete"]',   "Requesting delete…",
         async () => { await call("PATCH", `/api/calendar/${id}`, {action:"delete"}); dropItem(li); });
    bind(panel, '[data-act="alert"]',    "Toggling…",
         async () => { await call("PATCH", `/api/calendar/${id}/alert`); toast(panel, "Alert toggled"); });
  }

  // Leads + ScoreApp signups share endpoints — both stored in inbound_leads.
  function leadPanel(li){
    const id    = li.dataset.id;
    const email = li.dataset.email || "";
    const panel = li.querySelector(".action-panel");
    panel.innerHTML = `
      <div class="meta">${esc(email)}</div>
      <div class="btn-row">
        <button data-act="draft" class="primary">✉ Draft reply</button>
        <button data-act="copy">⎘ Copy email</button>
        <button data-act="tag-warm">★ Warm</button>
        <button data-act="tag-cold">◌ Cold</button>
        <button data-act="dismiss" class="danger">× Dismiss</button>
      </div>
      <div class="draft-wrap" hidden>
        <input class="subject" placeholder="Subject" />
        <div class="btn-row" style="margin-top:6px">
          <button data-act="gen" class="primary">Generate draft ✦</button>
          <button data-act="cancel">Close</button>
        </div>
        <textarea class="body" placeholder="Draft will appear here. Edit before sending." rows="6"></textarea>
        <div class="btn-row">
          <button data-act="send" class="primary">→ Send via Gmail</button>
        </div>
      </div>
    `;
    bind(panel, '[data-act="copy"]', "Copying…", async () => {
      const ok = await copyToClipboard(email);
      toast(panel, ok ? "Copied " + email : "Copy failed", !ok);
    });
    bind(panel, '[data-act="tag-warm"]', "Tagging…",
         async () => { await call("POST", `/api/leads/${id}/tag`, {tag:"warm"}); toast(panel, "Tagged warm"); });
    bind(panel, '[data-act="tag-cold"]', "Tagging…",
         async () => { await call("POST", `/api/leads/${id}/tag`, {tag:"cold"}); toast(panel, "Tagged cold"); });
    bind(panel, '[data-act="dismiss"]', "Dismissing…",
         async () => { await call("POST", `/api/leads/${id}/dismiss`); dropItem(li); });

    const wrap = panel.querySelector(".draft-wrap");
    panel.querySelector('[data-act="draft"]').addEventListener("click", () => { wrap.hidden = false; });
    panel.querySelector('[data-act="cancel"]').addEventListener("click", () => { wrap.hidden = true; });
    bind(panel, '[data-act="gen"]', "Generating…", async () => {
      const res = await call("POST", `/api/leads/${id}/draft-reply`);
      wrap.querySelector(".subject").value = res.subject || "";
      wrap.querySelector(".body").value    = res.draft   || "";
    });
    bind(panel, '[data-act="send"]', "Sending…", async () => {
      const subject = wrap.querySelector(".subject").value.trim();
      const body    = wrap.querySelector(".body").value.trim();
      if(!subject || !body){ toast(panel, "Subject and body required", true); return; }
      const r = await call("POST", `/api/leads/${id}/send-reply`, {subject, body});
      toast(panel, "Sent to " + (r.to || email));
      fade(li);
    });
  }

  function feedbackPanel(li){
    const id = li.dataset.id;
    const transcript = li.dataset.transcript || "";
    const panel = li.querySelector(".action-panel");
    panel.innerHTML = `
      ${transcript ? `<div class="meta">“${esc(transcript)}”</div>` : ""}
      <div class="btn-row">
        <button data-act="helpful"     class="primary">★ Helpful</button>
        <button data-act="not_helpful">⊘ Not helpful</button>
        <button data-act="noise">✕ Noise</button>
        <button data-act="reviewed">✓ Mark reviewed</button>
      </div>
    `;
    ["helpful","not_helpful","noise"].forEach(tag => {
      bind(panel, `[data-act="${tag}"]`, "Tagging…", async () => {
        await call("POST", `/api/rae-feedback/${id}/tag`, {tag});
        toast(panel, "Tagged " + tag.replace("_"," "));
        fade(li);
      });
    });
    bind(panel, '[data-act="reviewed"]', "Marking…",
         async () => { await call("POST", `/api/rae-feedback/${id}/mark-reviewed`); dropItem(li); });
  }

  function videoPanel(li){
    const id  = li.dataset.id;
    const url = li.dataset.url || "";
    const panel = li.querySelector(".action-panel");
    panel.innerHTML = `
      ${url ? `<div class="meta"><a href="${esc(url)}" target="_blank" rel="noopener">${esc(url)}</a></div>` : ""}
      <div class="btn-row">
        ${url ? `<button data-act="open" class="primary">▶ Open render</button>
                 <button data-act="copy">⎘ Copy URL</button>` : ""}
        <button data-act="reviewed">✓ Mark reviewed</button>
      </div>
    `;
    if(url){
      panel.querySelector('[data-act="open"]').addEventListener("click",
        () => window.open(url, "_blank", "noopener"));
      bind(panel, '[data-act="copy"]', "Copying…", async () => {
        const ok = await copyToClipboard(url);
        toast(panel, ok ? "Copied" : "Copy failed", !ok);
      });
    }
    bind(panel, '[data-act="reviewed"]', "Marking…",
         async () => { await call("POST", `/api/heygen/${encodeURIComponent(id)}/mark-reviewed`); fade(li); });
  }

  const BUILDERS = {
    todo:     todoPanel,
    event:    eventPanel,
    lead:     leadPanel,
    signup:   leadPanel,    // ScoreApp signups reuse the lead endpoints
    feedback: feedbackPanel,
    video:    videoPanel,
  };

  // ── click delegation ──────────────────────────────────────────────────────
  // Tap an `.item-row` to toggle that item's `.action-panel`.
  // We stop propagation so the card's refresh-on-click handler doesn't fire.

  document.addEventListener("click", (e) => {
    const row = e.target.closest("li.item .item-row");
    if(!row) return;
    e.stopPropagation();
    const li   = row.closest("li.item");
    const card = li.closest(".card");
    if(!card) return;
    const panel = li.querySelector(".action-panel");
    if(!panel) return;

    // Close any other open panel in the same card
    card.querySelectorAll("li.item.expanded").forEach(other => {
      if(other === li) return;
      other.classList.remove("expanded");
      const op = other.querySelector(".action-panel");
      if(op){ op.classList.remove("open"); op.hidden = true; }
    });

    const opening = panel.hidden;
    if(opening){
      const kind = li.dataset.kind;
      const build = BUILDERS[kind];
      if(build && !panel.dataset.built){
        try { build(li); panel.dataset.built = "1"; }
        catch(err){ panel.innerHTML = `<div class="msg err">Panel error: ${esc(err.message)}</div>`; }
      }
      panel.hidden = false;
      panel.classList.add("open");
      li.classList.add("expanded");
    } else {
      panel.classList.remove("open");
      panel.hidden = true;
      li.classList.remove("expanded");
    }
  });

  // Expose a helper dashboard.html can use to decide whether to auto-refresh.
  window.cardHasOpenPanel = (card) => !!card.querySelector("li.item.expanded");
})();
