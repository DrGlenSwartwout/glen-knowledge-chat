/* Shared entity-ref popover. No dependencies. One shared #entityPop element,
   one delegated listener per root. data-info/data-name are treated as PLAIN TEXT
   (assigned via textContent) — never injected as HTML. */
(function(){
  "use strict";
  var pop, hideT, isTouch = ("ontouchstart" in window) || navigator.maxTouchPoints > 0;

  function ensurePop(){
    if(pop) return pop;
    pop = document.createElement("div");
    pop.id = "entityPop";
    pop.setAttribute("role","tooltip");
    pop.innerHTML = '<div class="ep-name"></div><div class="ep-info"></div>'
                  + '<a class="ep-link" target="_blank" rel="noopener" hidden>Open full page ↗</a>';
    document.body.appendChild(pop);
    // Keep the popover open while the pointer is inside it (so the link is clickable).
    pop.addEventListener("mouseenter", function(){ clearTimeout(hideT); });
    pop.addEventListener("mouseleave", hide);
    return pop;
  }

  function show(el){
    var p = ensurePop();
    clearTimeout(hideT);
    p.querySelector(".ep-name").textContent = el.getAttribute("data-name") || el.textContent || "";
    p.querySelector(".ep-info").textContent = el.getAttribute("data-info") || "";
    var href = el.getAttribute("href") || el.getAttribute("data-href") || "";
    var link = p.querySelector(".ep-link");
    // The link row is for touch (no hover-click). On desktop the element itself
    // is the link, so we only surface the row when there's no hover affordance.
    if(href && isTouch){ link.href = href; link.hidden = false; }
    else { link.hidden = true; link.removeAttribute("href"); }
    p.classList.add("show");
    position(p, el);
  }

  function position(p, el){
    var r = el.getBoundingClientRect();
    var sx = window.pageXOffset, sy = window.pageYOffset;
    p.style.left = "0px"; p.style.top = "0px";  // measure at origin first
    var pw = p.offsetWidth, ph = p.offsetHeight, vw = document.documentElement.clientWidth;
    var left = Math.min(Math.max(8, r.left + sx), sx + vw - pw - 8);
    var below = r.bottom + sy + 6, above = r.top + sy - ph - 6;
    // Prefer below; flip above when it would overflow the viewport bottom.
    var top = (r.bottom + ph + 6 > document.documentElement.clientHeight && above > sy) ? above : below;
    p.style.left = left + "px"; p.style.top = top + "px";
  }

  function hide(){ hideT = setTimeout(function(){ if(pop) pop.classList.remove("show"); }, 120); }

  window.wireEntityRefs = function(root){
    root = root || document;
    if(root.__entityWired) return;
    root.__entityWired = true;
    root.addEventListener("mouseover", function(e){
      var el = e.target.closest && e.target.closest(".entity-ref[data-info]");
      if(el) show(el);
    });
    root.addEventListener("mouseout", function(e){
      if(e.target.closest && e.target.closest(".entity-ref[data-info]")) hide();
    });
    root.addEventListener("focusin", function(e){
      var el = e.target.closest && e.target.closest(".entity-ref[data-info]");
      if(el) show(el);
    });
    root.addEventListener("focusout", hide);
    // Touch: first tap shows the popover (and its link) instead of navigating.
    root.addEventListener("click", function(e){
      var el = e.target.closest && e.target.closest(".entity-ref[data-info]");
      if(!el) return;
      if(isTouch && pop && pop.classList.contains("show") &&
         pop.querySelector(".ep-name").textContent === (el.getAttribute("data-name")||el.textContent)){
        return; // second tap on the same ref: let the native link (if any) proceed
      }
      if(isTouch){ e.preventDefault(); show(el); }
      // desktop: native <a target=_blank> handles the new tab; span refs do nothing
    }, true);
    document.addEventListener("keydown", function(e){ if(e.key === "Escape") hide(); });
    // Dismiss on outside tap (touch).
    document.addEventListener("click", function(e){
      if(pop && pop.classList.contains("show") && !e.target.closest(".entity-ref") && !e.target.closest("#entityPop")) hide();
    });
  };
})();
