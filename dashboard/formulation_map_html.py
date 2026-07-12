"""Render the formulation-map curation page for the local Biofield tool. One row per
e4l code: current mappings (drag to reorder, x to remove), a manual catalog add, and a
'suggest' button that lazy-loads semantic proposals. Pure HTML; the JS calls the
/api/formulation-map/* routes. Reachable from both the Biofield and Reveals sub-tabs."""
from html import escape as _e


def render_code_row(code, name, mappings):
    """One code's editable mapping row. `mappings` = [{formulation_id, name, priority}]."""
    lis = ""
    for m in mappings or []:
        lis += (f"<li class=fmmap draggable=true data-fid=\"{m['formulation_id']}\" "
                "ondragstart=fmDragStart(event) ondragover=fmDragOver(event) "
                "ondrop=fmDrop(event) ondragend=fmDragEnd(event)>"
                "<span class=fmgrip title='Drag to reorder'>&#10303;</span>"
                f"<span class=fmname>{_e(m['name'])}</span>"
                f"<button type=button class=fmx title='Remove' onclick=fmRemove(this)>&times;</button></li>")
    empty = "" if mappings else "<li class=food style='list-style:none'>no remedies mapped yet</li>"
    return (f"<div class=fmcode data-code=\"{_e(code)}\">"
            f"<div class=fmhdr><b>{_e(code)}</b> <span class=food>{_e(name or '')}</span>"
            f"<span class=fmn>{len(mappings or [])}</span></div>"
            f"<ol class=fmlist>{lis}{empty}</ol>"
            "<div class=fmadd>"
            "<input class=fminput list=fmcatalog placeholder='add a remedy…' "
            "onkeydown='if(event.key===\"Enter\"){fmAdd(this);event.preventDefault()}'>"
            "<button type=button class='btn ghost' onclick=fmAdd(this.previousElementSibling)>+ add</button>"
            "<button type=button class='btn ghost' onclick=fmSuggest(this)>suggest &#9662;</button>"
            "<div class=fmprops></div></div></div>")


_STYLE = """<style>
.fmwrap{max-width:900px;margin:0 auto;padding:0 12px}
.fmcode{border:1px solid var(--line,#222);border-radius:10px;padding:10px 12px;margin:8px 0;background:#0c0e12}
.fmhdr{font-size:14px;margin-bottom:6px}.fmhdr .fmn{float:right;color:var(--muted,#8a93a0);font-size:11px}
.fmlist{list-style:none;margin:0 0 6px;padding:0}
.fmmap{display:flex;align-items:center;gap:8px;padding:4px 6px;border:1px solid #1c2029;border-radius:7px;margin:3px 0;background:#0f1218}
.fmmap.fmover{border-color:#d4a843}.fmgrip{cursor:grab;color:#556}.fmname{flex:1;font-size:13px}
.fmx{background:none;border:0;color:#a55;font-size:16px;cursor:pointer;line-height:1}
.fmadd{display:flex;gap:6px;flex-wrap:wrap;align-items:center}
.fminput{background:#0f1218;border:1px solid #222;border-radius:7px;color:#cfd6df;padding:5px 8px;font-size:12px;min-width:180px}
.fmprops{flex-basis:100%;margin-top:4px}
.fmchip{font-size:11px;margin:2px 4px 2px 0}.food{color:var(--muted,#8a93a0)}
#fmfilter{width:100%;max-width:300px;background:#0f1218;border:1px solid #222;border-radius:7px;color:#cfd6df;padding:6px 10px;margin:6px 0}
</style>"""

_JS = """<script>
async function fmPost(p,b){const r=await fetch(p,{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify(b)});return r.json()}
function fmRow(el){return el.closest('.fmcode')}
function fmCode(el){return fmRow(el).getAttribute('data-code')}
function fmPaint(el,mappings){
 var ol=fmRow(el).querySelector('.fmlist');ol.innerHTML='';
 (mappings||[]).forEach(function(m){
  var li=document.createElement('li');li.className='fmmap';li.draggable=true;li.setAttribute('data-fid',m.formulation_id);
  li.setAttribute('ondragstart','fmDragStart(event)');li.setAttribute('ondragover','fmDragOver(event)');
  li.setAttribute('ondrop','fmDrop(event)');li.setAttribute('ondragend','fmDragEnd(event)');
  li.innerHTML='<span class=fmgrip>\\u2937</span><span class=fmname></span><button type=button class=fmx onclick=fmRemove(this)>\\u00d7</button>';
  li.querySelector('.fmname').textContent=m.name;ol.appendChild(li)});
 if(!(mappings||[]).length){ol.innerHTML='<li class=food style="list-style:none">no remedies mapped yet</li>'}
 fmRow(el).querySelector('.fmn').textContent=(mappings||[]).length}
async function fmAdd(inp){var name=(inp.value||'').trim();if(!name)return;
 var j=await fmPost('/api/formulation-map/add',{code:fmCode(inp),remedy:name});inp.value='';if(j.ok)fmPaint(inp,j.mappings)}
async function fmRemove(btn){var fid=btn.closest('.fmmap').getAttribute('data-fid');
 var j=await fmPost('/api/formulation-map/remove',{code:fmCode(btn),formulation_id:parseInt(fid)});if(j.ok)fmPaint(btn,j.mappings)}
async function fmSuggest(btn){var box=fmRow(btn).querySelector('.fmprops');box.textContent='loading…';
 var j=await (await fetch('/api/formulation-map/propose?code='+encodeURIComponent(fmCode(btn)))).json();
 box.innerHTML='';(j.proposals||[]).forEach(function(p){
  var b=document.createElement('button');b.type='button';b.className='btn ghost fmchip';
  b.textContent='+ '+p.name+' ('+p.score+')';b.onclick=function(){var i=fmRow(btn).querySelector('.fminput');i.value=p.name;fmAdd(i);b.remove()};
  box.appendChild(b)});
 if(!(j.proposals||[]).length)box.innerHTML='<span class=food>no new suggestions</span>'}
var _fmDrag=null;
function fmDragStart(e){_fmDrag=e.target.closest('.fmmap')}
function fmDragOver(e){e.preventDefault();var li=e.target.closest('.fmmap');if(li&&li!==_fmDrag)li.classList.add('fmover')}
function fmDragEnd(e){fmRow(e.target).querySelectorAll('.fmmap').forEach(function(x){x.classList.remove('fmover')})}
async function fmDrop(e){e.preventDefault();var li=e.target.closest('.fmmap');if(!li||li===_fmDrag)return;
 var ol=li.parentNode;ol.insertBefore(_fmDrag,li);li.classList.remove('fmover');
 var order=[].slice.call(ol.querySelectorAll('.fmmap')).map(function(x){return parseInt(x.getAttribute('data-fid'))});
 var j=await fmPost('/api/formulation-map/reorder',{code:fmCode(li),order:order});if(j.ok)fmPaint(li,j.mappings)}
function fmFilter(v){v=(v||'').toLowerCase();document.querySelectorAll('.fmcode').forEach(function(c){
 c.style.display=(c.getAttribute('data-code')+' '+c.querySelector('.fmhdr').textContent).toLowerCase().indexOf(v)<0?'none':''})}
</script>"""


def render_formulation_map_page(codes, nav="", catalog_datalist=""):
    """`codes` = [(code, name, mappings)]. `nav` = the shared sub-tab strip html."""
    rows = "".join(render_code_row(c, n, m) for c, n, m in codes)
    return (
        "<!doctype html><meta charset=utf-8><title>Formulation map</title>"
        "<meta name=viewport content='width=device-width,initial-scale=1'>"
        f"{_STYLE}<body style='background:#080a0d;color:#cfd6df;font-family:system-ui,sans-serif'>"
        f"{nav}<div class=fmwrap>"
        "<h2 style='margin:10px 0 2px'>Formulation map</h2>"
        "<p class=food style='margin:0 0 8px;font-size:12px'>Curate which remedies each E4L code maps to. "
        "Priority 1 (top) is the reveal's default; the rest are the alternatives, in order. "
        "Drag to reorder, &times; to remove, or add from the suggestions / catalog.</p>"
        "<input id=fmfilter placeholder='filter codes…' oninput='fmFilter(this.value)'>"
        f"{catalog_datalist}{rows}</div>{_JS}")
