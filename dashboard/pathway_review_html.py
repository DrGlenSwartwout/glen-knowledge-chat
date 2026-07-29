"""Render the canonical-pathway review page for the local Biofield tool.

One card per pending atom, highest-impact first: what the atom is, how many
ingredients use it, the raw pathway strings it came from, and the merge the
atomizer proposes. Glen confirms, reassigns, orphans, or rejects. Decisions
persist, so the queue is resumable — close the tab mid-batch and the settled
ones never come back.

A second, smaller queue settles effect direction, which is a separate axis from
pathway identity. Pure HTML; the JS calls /api/pathway-review/*.
"""
from html import escape as _e


def _options(canon, selected_id):
    """Just the current selection. The full 150-entry list is emitted once for
    the whole page and filled in on focus — inlining it per card is 6,000 DOM
    nodes for a queue where most cards are confirmed without ever opening it."""
    if selected_id:
        label = next((c["label"] for c in canon if c["id"] == selected_id), "")
        if label:
            return f"<option value=\"{selected_id}\" selected>{_e(label)}</option>"
    return "<option value=''>— pick a canonical pathway —</option>"


def _canon_json(canon):
    """The picker's data, emitted once. Grouped by family in the order given."""
    items = ",".join(
        "{i:%d,l:%s,f:%s}" % (c["id"], _js(c["label"]), _js(c["family"] or "other"))
        for c in canon)
    return f"<script>window.PWCANON=[{items}]</script>"


def _js(s):
    return '"' + (s or "").replace("\\", "\\\\").replace('"', '\\"') \
        .replace("<", "\\u003c").replace("&", "\\u0026") + '"'


def render_atom_card(row, canon):
    """One pending atom. `row` comes from pathway_review.queue()."""
    proposed_reject = row["decision"] == "proposed-reject"
    if proposed_reject:
        verdict = (f"<span class=pwrej>proposed: not a pathway</span>"
                   f"<span class=pwwhy>{_e(row.get('rationale') or '')}</span>")
    elif row.get("canonical_label"):
        verdict = f"<span class=pwmerge>merge &rarr; <b>{_e(row['canonical_label'])}</b></span>"
    else:
        verdict = "<span class=pwwhy>no merge proposed</span>"

    ex = "".join(f"<li>{_e(e)}</li>" for e in (row.get("examples") or []))
    ex = f"<ul class=pwex>{ex}</ul>" if ex else ""

    return (
        f"<div class=pwcard data-atom=\"{_e(row['atom_key'])}\">"
        f"<div class=pwhdr><b class=pwatom>{_e(row['atom'])}</b>"
        f"<span class=pwn>{row['ingredients']} ingredients &middot; "
        f"{row['mentions']} mentions</span></div>"
        f"<div class=pwverdict>{verdict}</div>"
        f"{ex}"
        "<div class=pwact>"
        "<button type=button class='btn' onclick=pwDecide(this,'confirmed')>confirm</button>"
        "<button type=button class='btn ghost' onclick=pwDecide(this,'orphan')"
        " title='A real pathway, but its own — stays unmapped, contributes no overlap'>orphan</button>"
        "<button type=button class='btn ghost' onclick=pwDecide(this,'rejected')"
        " title='Not a functional pathway at all'>not a pathway</button>"
        "<select class=pwsel onfocus=pwFill(this) onchange=pwPickChanged(this)>"
        f"{_options(canon, row.get('canonical_id'))}</select>"
        "</div><div class=pwmsg></div></div>")


def render_direction_row(row):
    opts = "".join(
        f"<option value='{d}'{' selected' if d == row['direction'] else ''}>{d}</option>"
        for d in ("up", "down", "balance", "substrate", "adverse", "neutral", "unknown"))
    return (f"<div class=pwdir data-effect=\"{_e(row['effect_key'])}\">"
            f"<span class=pwdtext>{_e(row['effect_key'])}</span>"
            f"<span class=pwn>{row['n_rows']} rows</span>"
            f"<select onchange=pwDirSet(this)>{opts}</select>"
            "<span class=pwmsg></span></div>")


_STYLE = """<style>
/* Tokens, nav strip and buttons mirror biofield_report_html so the shared
   sub-tab strip renders the same here as on the pages it links to. */
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
.pwwrap{max-width:900px;margin:0 auto;padding:0 12px}
.pwbar{display:flex;gap:14px;flex-wrap:wrap;align-items:baseline;font-size:12px;
 border:1px solid #1c2029;border-radius:10px;padding:8px 12px;margin:8px 0;background:#0c0e12}
.pwbar b{font-size:15px;color:#d4a843}
.pwcard{border:1px solid var(--line,#222);border-radius:10px;padding:10px 12px;margin:8px 0;background:#0c0e12}
.pwcard.done{opacity:.45}
.pwhdr{display:flex;justify-content:space-between;gap:10px;align-items:baseline}
.pwatom{font-size:15px}
.pwn{color:var(--muted,#8a93a0);font-size:11px;white-space:nowrap}
.pwverdict{font-size:12px;margin:4px 0 6px}
.pwmerge{color:#8fc7a8}.pwrej{color:#c98b8b}
.pwwhy{color:var(--muted,#8a93a0);margin-left:6px}
.pwex{margin:0 0 8px;padding-left:18px;color:#8a93a0;font-size:11px;line-height:1.5}
.pwex li{margin:1px 0}
.pwact{display:flex;gap:6px;flex-wrap:wrap;align-items:center}
.pwsel,.pwdir select{background:#0f1218;border:1px solid #222;border-radius:7px;
 color:#cfd6df;padding:5px 8px;font-size:12px;max-width:280px}
.pwmsg{font-size:11px;color:#8fc7a8;margin-left:6px}
.pwdir{display:flex;gap:10px;align-items:center;padding:5px 8px;border:1px solid #1c2029;
 border-radius:7px;margin:3px 0;background:#0f1218;font-size:12px}
.pwdtext{flex:1;color:#cfd6df}
.pwmore{display:block;margin:14px auto;padding:8px 18px}
.food{color:var(--muted,#8a93a0)}
h3{margin:22px 0 4px;font-size:14px}
</style>"""

_JS = """<script>
async function pwPost(p,b){const r=await fetch(p,{method:'POST',
 headers:{'Content-Type':'application/json'},body:JSON.stringify(b)});return r.json()}
function pwCard(el){return el.closest('.pwcard')}
function pwFill(sel){
 if(sel.dataset.filled)return; sel.dataset.filled='1';
 var cur=sel.value,fam=null,grp=null;
 var blank=document.createElement('option');blank.value='';
 blank.textContent='\\u2014 pick a canonical pathway \\u2014';
 sel.insertBefore(blank,sel.firstChild);
 (window.PWCANON||[]).forEach(function(c){
  if(String(c.i)===cur)return;              // already the selected option
  if(c.f!==fam){fam=c.f;grp=document.createElement('optgroup');grp.label=fam;sel.appendChild(grp)}
  var o=document.createElement('option');o.value=c.i;o.textContent=c.l;grp.appendChild(o)});
 sel.value=cur}
function pwPickChanged(sel){
 // Choosing a different canonical is itself the confirmation of that merge.
 if(sel.value)pwDecide(sel,'confirmed')}
async function pwDecide(el,decision){
 var card=pwCard(el),atom=card.getAttribute('data-atom');
 var sel=card.querySelector('.pwsel');
 var cid=(decision==='confirmed'&&sel&&sel.value)?parseInt(sel.value):null;
 if(decision==='confirmed'&&!cid){
  card.querySelector('.pwmsg').textContent='pick a canonical pathway first';return}
 var j=await pwPost('/api/pathway-review/decide',{atom_key:atom,decision:decision,canonical_id:cid});
 if(!j.ok){card.querySelector('.pwmsg').textContent='failed';return}
 card.classList.add('done');
 card.querySelector('.pwmsg').innerHTML=decision+' \\u2713 <a href=# onclick="pwUndo(this);return false">undo</a>';
 pwBump(1)}
async function pwUndo(a){
 var card=pwCard(a);await pwPost('/api/pathway-review/undo',{atom_key:card.getAttribute('data-atom')});
 card.classList.remove('done');card.querySelector('.pwmsg').textContent='back in the queue';pwBump(-1)}
function pwBump(n){
 var s=document.getElementById('pwsettled'),p=document.getElementById('pwpending');
 if(s)s.textContent=parseInt(s.textContent)+n; if(p)p.textContent=parseInt(p.textContent)-n}
async function pwDirSet(sel){
 var row=sel.closest('.pwdir');
 var j=await pwPost('/api/pathway-review/direction',{effect_key:row.getAttribute('data-effect'),
  direction:sel.value});
 row.querySelector('.pwmsg').textContent=j.ok?'\\u2713':'failed'}
async function pwMore(btn){
 btn.disabled=true;btn.textContent='loading…';
 var off=parseInt(btn.getAttribute('data-offset'));
 var r=await fetch('/api/pathway-review/queue?offset='+off);var j=await r.json();
 if(j.html){document.getElementById('pwqueue').insertAdjacentHTML('beforeend',j.html);
  btn.setAttribute('data-offset',off+j.count)}
 if(!j.count||!j.html){btn.remove();return}
 btn.disabled=false;btn.textContent='load more'}
</script>"""


def render_pathway_review_page(rows, canon, stats, directions=(), nav="", offset=0):
    cards = "".join(render_atom_card(r, canon) for r in rows)
    dirs = "".join(render_direction_row(d) for d in directions)
    dir_block = (
        "<h3>Effect direction</h3>"
        "<p class=food style='margin:0 0 6px;font-size:12px'>Direction is a separate axis "
        "from pathway identity &mdash; NF-&kappa;B inhibited and NF-&kappa;B activated are the "
        "same pathway. These are the phrasings the classifier could not place; the rest were "
        "read straight off the leading verb.</p>" + dirs) if dirs else ""
    more = (f"<button type=button class='btn ghost pwmore' data-offset='{offset + len(rows)}' "
            "onclick=pwMore(this)>load more</button>") if rows else ""
    empty = ("<p class=food style='margin:18px 0'>Queue is clear &mdash; every atom shared by "
             "two or more ingredients has been settled.</p>" if not rows else "")
    return (
        "<!doctype html><meta charset=utf-8><title>Pathway review</title>"
        "<meta name=viewport content='width=device-width,initial-scale=1'>"
        f"{_STYLE}<body>"
        f"{nav}<div class=pwwrap>"
        "<h2 style='margin:10px 0 2px'>Pathway review</h2>"
        "<p class=food style='margin:0 0 8px;font-size:12px'>Collapse the free-text pathway "
        "vocabulary into canonical pathways so coverage can rank remedies against a condition. "
        "Nothing here edits an original pathway string &mdash; a bad merge is undone by changing "
        "one mapping. Never invent a canonical pathway just to give an orphan a home: an unmapped "
        "orphan contributes no overlap, which is honest rather than lossy.</p>"
        "<div class=pwbar>"
        f"<span><b id=pwsettled>{stats['settled']}</b> settled</span>"
        f"<span><b id=pwpending>{stats['batch']}</b> in this batch "
        f"<span class=food>(atoms reaching 2+ ingredients)</span></span>"
        f"<span class=food>{stats['pending']} pending in all, "
        f"singleton tail included</span>"
        f"<span><b>{stats['canonical']}</b> canonical "
        f"({stats['canonical_confirmed']} confirmed)</span>"
        f"<span class=food>{stats['atoms_total']} distinct atoms in the corpus</span>"
        "</div>"
        f"<div id=pwqueue>{cards}</div>{empty}{more}{dir_block}"
        f"</div>{_canon_json(canon)}{_JS}")
