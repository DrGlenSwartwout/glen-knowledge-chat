"""Render the condition -> canonical pathway review queue.

Sibling of pathway_review_html (same dark shell, same card-then-actions shape) but
one axis over: there the reviewer collapses free-text pathway strings into a
canonical vocabulary; here he rules on whether a CONDITION genuinely implicates a
canonical pathway, in a stated direction, at a stated importance.

DIRECTION IS THE FIELD BEING ADJUDICATED, not the pathway or the tier. Every
serious defect found across thirteen review passes on the ingredient-side
vocabulary was a sign inversion, not a missing edge -- a missing edge merely hides
a remedy, an inverted one makes the finished score recommend the opposite of what
helps. So the proposed direction is rendered as the first, boldest, largest
control on the card (`.dirline`), not buried in a caption line, and it is also the
live `<select>` the reviewer edits -- changing it is the same one action as
picking a new option, with no separate save step. Confirm/Reject reads whatever
the select currently shows.

THE CORPUS LINE IS THE DISAGREEMENT SIGNAL. It reports how many of Glen's own
products already touch this pathway. `condition_corpus_signal` has no builder yet,
so every one of the 323 cards renders the no-coverage state today -- that must
read as "flagged for your judgment," not as a broken feature, because it is about
to be the state of the whole queue.

Rendering only. No database access, no Flask, no route logic in this module.
"""
from html import escape as _e

_DIRECTIONS = ("up", "down", "balance", "substrate", "unknown")
_TIERS = ("core", "contributing", "modifying")

# Decorative prefix only -- the <option value> stays the bare vocabulary word so
# the JS that reads `.value` needs no translation layer.
_DIR_ARROW = {"up": "▲", "down": "▼", "balance": "⇌",
              "substrate": "→", "unknown": "?"}


def _select(field, options, selected, edge_id, css_class, labels=None):
    labels = labels or {}
    opts = "".join(
        "<option value=\"%s\"%s>%s</option>"
        % (_e(o), " selected" if o == selected else "", _e(labels.get(o, o)))
        for o in options)
    return ("<select class=\"%s\" data-field=\"%s\" data-id=\"%d\">%s</select>"
            % (_e(css_class), _e(field), edge_id, opts))


def render_edge_card(row):
    """One reviewable edge. `row` is a sqlite3.Row from
    condition_pathway_review.queue -- see that function's SELECT for the exact
    column names this reads.

    queue() deliberately does NOT filter on `conditions.scored` (filtering there
    deadlocks the vault's batch gate -- see queue()'s docstring), so an edge on a
    condition marked scored=0 (no genuine biochemical mechanism) can reach this
    card. That is either a fabricated edge or a wrong `scored` flag; either way
    Glen must see it named, not silently decide a mechanism ruling on a condition
    nobody meant to score. Flagged ahead of everything else on the card, because
    it questions whether the edge should exist at all.
    """
    eid = row["id"]
    unscored = ("<div class=\"unscored\">&#9888; <b>%s</b> is marked unscored (no "
                "established biochemical mechanism) &mdash; this edge should not "
                "exist. Reject it, or flag that the condition's scored flag is "
                "wrong.</div>" % _e(row["condition_label"])) if not row["scored"] else ""
    covered = int(row["corpus_products"] or 0)
    if covered:
        n_ing = row["corpus_ingredients"] or 0
        corpus = (
            "<span class=\"corpus\">%d product%s, %d ingredient%s in the catalog "
            "already touch this pathway &middot; %s</span>"
            % (covered, "" if covered == 1 else "s",
               n_ing, "" if n_ing == 1 else "s",
               _e(row["example_skus"] or "")))
    else:
        # Not an error state -- it is the flag. An edge no product touches is
        # either a hole in the product line or a wrong edge; either way Glen
        # must see it named, not silently absent.
        corpus = ("<span class=\"corpus warn\">no product in the catalog touches "
                  "this pathway yet &mdash; a hole in the line, or a wrong edge</span>")

    dir_labels = {d: "%s %s" % (_DIR_ARROW[d], d) for d in _DIRECTIONS}
    dir_select = _select("desired_direction", _DIRECTIONS, row["desired_direction"],
                          eid, "dirsel", dir_labels)
    tier_select = _select("tier", _TIERS, row["tier"], eid, "tiersel")

    return (
        "<div class=\"card\" data-id=\"%d\">%s"
        "<div class=\"hdr\"><b>%s</b> <span class=\"arrow\">&rarr;</span> "
        "<b>%s</b> <span class=\"fam\">%s</span></div>"
        "<div class=\"dirline\"><span class=\"lbl\">proposed direction</span>%s"
        "<span class=\"lbl\">at</span>%s<span class=\"lbl\">tier</span></div>"
        "<div class=\"why\">%s</div>"
        "<div class=\"corpusline\">%s</div>"
        "<div class=\"act\">"
        "<button class=\"btn\" data-act=\"confirmed\" data-id=\"%d\">Confirm</button>"
        "<button class=\"btn ghost\" data-act=\"rejected\" data-id=\"%d\">Reject</button>"
        "<button class=\"btn ghost\" data-act=\"needs\" data-id=\"%d\">Needs a pathway "
        "that does not exist</button>"
        "</div></div>"
        % (eid, unscored, _e(row["condition_label"]), _e(row["pathway_label"]),
           _e(row["family"] or ""), dir_select, tier_select,
           _e(row["rationale"]), corpus, eid, eid, eid))


_STYLE = """<style>
:root{--bg:#0f1115;--card:#171a21;--line:#2a2f3a;--fg:#e8ebf0;--muted:#9aa3b2;
 --accent:#d4a843;--ok:#3fb968}
*{box-sizing:border-box}
body{margin:0;background:var(--bg);color:var(--fg);
 font:15px/1.5 -apple-system,Segoe UI,Roboto,sans-serif;padding:14px}
.wfnav{display:flex;gap:4px;margin:0 0 14px;padding:4px;background:var(--card);
 border:1px solid var(--line);border-radius:10px;flex-wrap:wrap}
.wfnav a{padding:6px 14px;border-radius:7px;color:var(--muted);font-size:13px;
 font-weight:600;text-decoration:none}
.wfnav a:hover{background:rgba(255,255,255,.05);color:var(--fg)}
.wfnav a.active{background:var(--accent);color:#0c0e12}
.btn{background:var(--accent);color:#0c0e12;border:0;border-radius:8px;padding:6px 12px;
 font:inherit;font-size:13px;font-weight:600;cursor:pointer}
.ghost{background:#13161c;color:var(--fg);border:1px solid var(--line)}
.cprwrap{max-width:900px;margin:0 auto;padding:0 12px}
.cprbar{display:flex;gap:14px;flex-wrap:wrap;align-items:baseline;font-size:12px;
 border:1px solid #1c2029;border-radius:10px;padding:8px 12px;margin:8px 0;background:#0c0e12}
.cprbar b{font-size:15px;color:#d4a843}
.card{border:1px solid var(--line,#222);border-radius:10px;padding:10px 12px;
 margin:8px 0;background:#0c0e12}
.card.done{opacity:.45}
.hdr{font-size:15px}
.fam{color:var(--muted);font-size:11px}
.unscored{background:#2a1414;color:#e0a0a0;border:1px solid #6b2f2f;border-radius:6px;
 padding:4px 8px;margin-bottom:6px;font-size:12px;font-weight:600}
.dirline{display:flex;gap:8px;align-items:center;margin:8px 0;padding:6px 8px;
 border:1px solid #1c2029;border-radius:8px;background:#12141a}
.dirline .lbl{color:var(--muted);font-size:11px}
.dirsel,.tiersel{font-weight:700;font-size:.95rem;padding:4px 8px;border-radius:6px;
 background:#1c2029;color:var(--fg);border:1px solid #333}
.dirsel:has(option[value=down]:checked){background:#1b2b4a;color:#8ab4f8;border-color:#2f4a7a}
.dirsel:has(option[value=up]:checked){background:#4a2a12;color:#e0a468;border-color:#7a4a2f}
.dirsel:has(option[value=balance]:checked){background:#3a1b4a;color:#c68af8;border-color:#5a2f7a}
.dirsel:has(option[value=substrate]:checked){background:#123a3a;color:#68e0d8;border-color:#2f7a76}
.dirsel:has(option[value=unknown]:checked){background:#2a2a2a;color:#999;border-color:#444}
.why{color:#cfd6df;margin:.35rem 0;font-size:13px}
.corpusline{font-size:.85rem;color:var(--muted)}
.corpus.warn{color:#d8a94a}
.act{margin-top:.5rem;display:flex;gap:6px;flex-wrap:wrap}
.food{color:var(--muted)}
</style>"""

_JS = """<script>
document.getElementById('cprq').addEventListener('click',async e=>{
 const b=e.target.closest('button'); if(!b) return;
 const card=b.closest('.card'), id=+b.dataset.id;
 if(b.dataset.act==='needs'){
   const label=prompt('Pathway label that does not exist yet:'); if(!label) return;
   await fetch('/api/condition-review/needs-canonical',{method:'POST',
     headers:{'Content-Type':'application/json'},
     body:JSON.stringify({edge_id:id,label:label})});
   return; }
 const dir=card.querySelector('[data-field=desired_direction]').value;
 const tier=card.querySelector('[data-field=tier]').value;
 const j=await fetch('/api/condition-review/decide',{method:'POST',
   headers:{'Content-Type':'application/json'},
   body:JSON.stringify({edge_id:id,decision:b.dataset.act,
     desired_direction:dir,tier:tier})}).then(r=>r.json()).catch(()=>({ok:false}));
 if(j && j.ok===false) return;
 card.classList.add('done');
 card.remove();
});
async function cprMore(btn){
 btn.disabled=true;btn.textContent='loading…';
 const off=parseInt(btn.getAttribute('data-offset'));
 const r=await fetch('/api/condition-review/queue?offset='+off);
 const j=await r.json().catch(()=>({}));
 if(j.html){document.getElementById('cprq').insertAdjacentHTML('beforeend',j.html);
  btn.setAttribute('data-offset',off+(j.count||0))}
 if(!j.count||!j.html){btn.remove();return}
 btn.disabled=false;btn.textContent='load more'}
</script>"""


def render_condition_review_page(rows, stats, nav="", offset=0):
    """Full review page: stats bar, the card queue, a load-more button (mirrors
    the sibling's `/api/pathway-review/queue?offset=` pattern; the route itself
    is a later task)."""
    cards = "".join(render_edge_card(r) for r in rows)
    empty = ("<p class=\"food\" style=\"margin:18px 0\">Queue is clear &mdash; every "
              "proposed edge has been decided.</p>" if not rows else "")
    more = ("<button type=button class=\"btn ghost\" data-offset=\"%d\" "
            "onclick=\"cprMore(this)\">load more</button>"
            % (offset + len(rows))) if rows else ""
    return (
        "<!doctype html><meta charset=utf-8><title>Condition &rarr; pathway review</title>"
        "<meta name=viewport content='width=device-width,initial-scale=1'>"
        "%s<body>"
        "%s<div class=\"cprwrap\">"
        "<h2 style=\"margin:10px 0 2px\">Condition &rarr; pathway review</h2>"
        "<p class=\"food\" style=\"margin:0 0 8px;font-size:12px\">Each card is one "
        "proposed edge: does this condition genuinely implicate this pathway, in this "
        "direction, at this importance? A wrong direction is worse than a missing edge "
        "&mdash; it recommends the opposite of what helps &mdash; so it leads the card.</p>"
        "<div class=\"cprbar\">"
        "<span><b>%d pending</b></span>"
        "<span><b>%d confirmed</b></span>"
        "<span><b>%d rejected</b></span>"
        "<span><b>%d conditions</b></span>"
        "</div>"
        "<div id=\"cprq\">%s</div>%s%s"
        "</div>%s"
        % (_STYLE, nav, stats["pending"], stats["confirmed"], stats["rejected"],
           stats["conditions"], cards, empty, more, _JS))
