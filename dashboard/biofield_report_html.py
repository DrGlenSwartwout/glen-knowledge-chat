"""HTML rendering for the local Biofield Analysis viewer (Glen's Mac only).

Pure string builders so they're unit-testable without Flask. ALL dynamic values
(remedy names, timing, client names) come from FileMaker free-text fields and are
HTML-escaped.
"""
import os
from html import escape as _e

# Where the deployed console lives (for the "Back to Console" link in the header bar).
CONSOLE_BASE = os.environ.get("CONSOLE_BASE_URL", "https://illtowell.com").rstrip("/")

_STYLE = """
<style>
 :root{--bg:#0f1115;--card:#171a21;--line:#2a2f3a;--fg:#e8ebf0;--muted:#9aa3b2;--accent:#d4a843;--ok:#3fb968}
 *{box-sizing:border-box} body{margin:0;background:var(--bg);color:var(--fg);
   font:15px/1.5 -apple-system,Segoe UI,Roboto,sans-serif}
 .opbar{position:sticky;top:0;z-index:9999;display:flex;align-items:center;background:#0a0a0f;
   border-bottom:1px solid #2a2a35;padding:0 14px;height:40px;font:13px -apple-system,Segoe UI,sans-serif;
   box-shadow:0 1px 0 rgba(0,0,0,.4),0 4px 12px rgba(0,0,0,.25)}
 .opbrand{color:#9a9384;letter-spacing:.18em;text-transform:uppercase;font-size:10px;font-weight:600;
   margin-right:14px;font-family:ui-monospace,Menlo,Consolas,monospace}
 .opbrand b{color:#e6b800;font-weight:700}
 .opsub{color:#d4a843;letter-spacing:.14em;text-transform:uppercase;font-size:10px;font-weight:700}
 .opspacer{flex:1}
 .opbar a.optab{display:inline-flex;align-items:center;height:100%;padding:0 13px;color:#9aa0b4;
   text-decoration:none;border-bottom:2px solid transparent}
 .opbar a.optab:hover{color:#e6edf3;background:rgba(255,255,255,.03)}
 .wrap{max-width:1040px;margin:0 auto;padding:22px}
 a{color:var(--accent);text-decoration:none} a:hover{text-decoration:underline}
 h1{font-size:21px;margin:0 0 2px} h2{font-size:15px;color:var(--muted);margin:22px 0 8px;
   text-transform:uppercase;letter-spacing:.04em}
 .sub{color:var(--muted);margin:0 0 16px}
 table{width:100%;border-collapse:collapse;background:var(--card);border:1px solid var(--line);
   border-radius:10px;overflow:hidden}
 th,td{padding:8px 10px;border-bottom:1px solid var(--line);text-align:left;vertical-align:top;font-size:14px}
 th{color:var(--muted);font-weight:600;background:#13161c}
 tr:last-child td{border-bottom:0}
 .lyr{color:var(--accent);font-weight:700;white-space:nowrap}
 .slot{font-weight:600;color:var(--accent);white-space:nowrap;width:130px}
 .food{color:var(--muted);font-size:12px}
 .warn{color:#e0823a;font-size:12px}
 input[type=search]{background:#0c0e12;color:var(--fg);border:1px solid var(--line);
   border-radius:8px;padding:8px 10px;width:280px;font:inherit}
 .pill{display:inline-block;background:#0c0e12;border:1px solid var(--line);border-radius:999px;
   padding:1px 8px;font-size:12px;color:var(--muted)}
 textarea{width:100%;background:#0c0e12;color:var(--fg);border:1px solid var(--line);
   border-radius:8px;padding:9px;font:inherit;margin:4px 0 6px}
 label{display:block;margin-top:8px;color:var(--muted);font-size:13px}
 .btn{background:var(--accent);color:#0c0e12;border:0;border-radius:8px;padding:7px 13px;
   font:inherit;font-weight:600;cursor:pointer}
 .btnrow{margin:6px 0 14px;display:flex;gap:8px;align-items:center;flex-wrap:wrap}
 .chip{background:#0c0e12;border:1px solid var(--line);color:var(--accent);border-radius:999px;
   padding:2px 9px;font:inherit;font-size:12px;cursor:pointer}
 .ghost{background:#13161c;color:var(--fg);border:1px solid var(--line)}
 td input{width:100%;background:#0c0e12;color:var(--fg);border:1px solid var(--line);
   border-radius:6px;padding:5px;font:inherit;font-size:13px}
 td input.lyr{width:46px;text-align:center}
 td{white-space:nowrap}
 tr.unconf td{box-shadow:inset 4px 0 0 var(--accent);background:#1a160d}
</style>
"""

_NARR_JS = """
<script>
function stat(t){document.getElementById('stat').textContent=t}
async function post(p,b){const r=await fetch(p,{method:'POST',
 headers:{'Content-Type':'application/json'},body:JSON.stringify(b)});return r.json()}
async function saveNotes(){await post('/test/__TID__/notes',
 {notes:document.getElementById('notes').value});stat('Notes saved.')}
async function generate(){stat('Generating\\u2026');
 const r=await post('/test/__TID__/generate',{notes:document.getElementById('notes').value});
 document.getElementById('narr').value=r.narrative||('['+(r.error||'error')+']');
 stat(r.error?('Error: '+r.error):'Generated \\u2014 review, edit, then Save.')}
async function saveNarr(){await post('/test/__TID__/narrative',
 {narrative:document.getElementById('narr').value});stat('Narrative saved.')}
async function vgen(){stat('Generating script\\u2026');
 const r=await post('/test/__TID__/video-generate',{notes:document.getElementById('notes').value});
 document.getElementById('vscript').value=r.script||('['+(r.error||'error')+']');
 stat(r.error?('Error: '+r.error):'Script generated \\u2014 edit, then Save or Make audio.')}
async function vsave(){await post('/test/__TID__/video-script',
 {script:document.getElementById('vscript').value});stat('Script saved.')}
async function vaudio(){stat('Rendering audio in your voice\\u2026 (~10-30s)');await vsave();
 const r=await post('/test/__TID__/audio',{});
 if(r.error){stat('Error: '+r.error);return}
 document.getElementById('audiobox').innerHTML=
  '<audio controls src=\\''+r.url+'\\'></audio> &nbsp; <a href=\\''+r.url+'\\' download>Download mp3</a>';
 stat('Audio ready.')}
</script>"""


def _bar():
    return ("<nav class=opbar><span class=opbrand>GLEN <b>&middot;</b> OPS</span>"
            "<span class=opsub>Biofield Intake</span><span class=opspacer></span>"
            "<a class=optab href='/'>All tests</a>"
            f"<a class=optab href='{CONSOLE_BASE}/console'>&larr; Console</a></nav>")


def _page(title, body):
    return (f"<!doctype html><html lang=en><head><meta charset=utf-8>"
            f"<meta name=viewport content='width=device-width,initial-scale=1'>"
            f"<title>{_e(title)}</title>{_STYLE}</head>"
            f"<body>{_bar()}<div class=wrap>{body}</div></body></html>")


def render_report_html(report, notes="", narrative="", video_script="", stresses=None):
    c = report.get("client") or {}
    name = _e(c.get("name") or "(unknown)")
    email = _e(c.get("email") or "")
    date = _e(report.get("date") or "")
    head = (f"<p><a href='/'>&larr; All tests</a></p>"
            f"<h1>{name}</h1>"
            f"<p class=sub>{email} &nbsp;&middot;&nbsp; {date} "
            f"&nbsp;&middot;&nbsp; test {_e(report.get('test_id') or '')}</p>")
    tid_link = _e(report.get("test_id") or "")
    head += (f'<p class=sub><a href="/test/{tid_link}/report" target="_blank">Open clean report</a>'
             f' &nbsp;·&nbsp; <a href="/test/{tid_link}/report.pdf" target="_blank">Download printable PDF</a></p>')

    # Causal chain table
    rows = ""
    for l in report.get("layers") or []:
        ln = l.get("layer")
        badge = ""
        if l.get("depth_status") == "shallow":
            badge = (f"<br><span class=warn>&#9888; may not reach "
                     f"{_e(l.get('depth_need') or 'this depth')}</span>")
        rows += (
            "<tr>"
            f"<td class=lyr>{_e(str(ln)) if ln is not None else '&middot;'}</td>"
            f"<td>{_e(l.get('head') or '')}</td>"
            f"<td>{_e(l.get('most_affected') or '')}</td>"
            f"<td>{_e(l.get('remedy') or '')}{badge}</td>"
            f"<td>{_e(l.get('dosage') or '')}</td>"
            f"<td>{_e(l.get('frequency') or '')}</td>"
            f"<td>{_e(l.get('timing') or '')}</td>"
            "</tr>")
    chain = ("<h2>Causal Chain Report</h2>"
             "<table><tr><th>Layer</th><th>Head of Chain</th><th>Most Affected</th>"
             "<th>Remedy</th><th>Dosage</th><th>Frequency</th><th>Timing</th></tr>"
             f"{rows}</table>")

    # Schedule grid
    sched = report.get("schedule") or {}
    entries = sched.get("entries") or []
    placed = [e for e in entries if not e.get("as_directed")]
    srows = ""
    for slot in sched.get("slots") or []:
        here = [e for e in placed if slot in (e.get("slots") or [])]
        if not here:
            continue
        cells = "; ".join(
            f"{_e(e.get('name') or '')} <span class=food>({_e(e.get('dosage') or '')}"
            + (f", {_e(e.get('food'))}" if e.get('food') else "") + ")</span>"
            for e in here)
        srows += f"<tr><td class=slot>{_e(slot)}</td><td>{cells}</td></tr>"
    asdir = [e for e in entries if e.get("as_directed")]
    if asdir:
        cells = "; ".join(
            f"{_e(e.get('name') or '')} <span class=food>({_e(e.get('timing') or 'as directed')})</span>"
            for e in asdir)
        srows += f"<tr><td class=slot>As directed</td><td>{cells}</td></tr>"
    schedule = ("<h2>Remedy Schedule</h2>"
                "<table><tr><th>When</th><th>Take</th></tr>" + srows + "</table>")

    # Narrative + verbal notes (Increment 2)
    tid = _e(report.get("test_id") or "")
    narr = (
        "<h2>Narrative</h2>"
        "<p class=sub>Add your verbal notes, then generate the warm narrative "
        "(a draft for your review).</p>"
        "<label for=notes>Verbal notes</label>"
        f"<textarea id=notes rows=4>{_e(notes)}</textarea>"
        "<div class=btnrow>"
        "<button class=btn onclick=saveNotes()>Save notes</button>"
        "<button class=btn onclick=generate()>Generate narrative</button>"
        "<span id=stat class=food></span></div>"
        "<label for=narr>Narrative (editable draft)</label>"
        f"<textarea id=narr rows=16>{_e(narrative)}</textarea>"
        "<div class=btnrow><button class=btn onclick=saveNarr()>Save narrative</button></div>"
        + _NARR_JS.replace("__TID__", tid))

    # Walkthrough video — short spoken script + ElevenLabs audio (Increment 3)
    vid = (
        "<h2>Walkthrough video (your voice)</h2>"
        "<p class=sub>Generate a short spoken script, then render it as audio in your "
        "ElevenLabs voice.</p>"
        "<div class=btnrow>"
        "<button class=btn onclick=vgen()>Generate script</button>"
        "<button class=btn onclick=vsave()>Save script</button>"
        "<button class=btn onclick=vaudio()>Make audio</button></div>"
        f"<textarea id=vscript rows=6>{_e(video_script)}</textarea>"
        "<div id=audiobox class=btnrow></div>")

    stresses_section = ""
    if stresses is not None:
        bal = stresses.get("balanced") or []
        if bal:
            items = "".join(
                f"<li><b>{_e(s.get('code') or '')}</b> {_e(s.get('label') or '')} "
                f"<span class=food>&mdash; {_e(s.get('balanced_by') or '')}</span></li>"
                for s in bal)
            stresses_section = ("<h2>Stresses balanced</h2>"
                                f"<ul style='margin:4px 0;padding-left:20px'>{items}</ul>")
    return _page(f"{name} — Biofield Analysis", head + chain + schedule + narr + vid + stresses_section)


_AUTHOR_JS = """
<script>
function val(id){var e=document.getElementById(id);return e?e.value:''}
function set(id,v){var e=document.getElementById(id);if(e)e.value=v}
function astat(t){document.getElementById('astat').textContent=t}
function opt(v){return '<option value="'+String(v).replace(/"/g,'&quot;')+'">'}
async function post(p,b){const r=await fetch(p,{method:'POST',
 headers:{'Content-Type':'application/json'},body:JSON.stringify(b)});return r.json()}
function rowVals(p){return {layer:val(p+'_layer'),head:val(p+'_head'),most_affected:val(p+'_most'),
 remedy:val(p+'_remedy'),dosage:val(p+'_dosage'),frequency:val(p+'_frequency'),timing:val(p+'_timing')}}
function setE4L(j){if(j&&j.html!==undefined)document.getElementById('e4lpanel').innerHTML=j.html}
async function loadE4L(){try{setE4L(await (await fetch('/author/__TID__/e4l')).json())}catch(e){}}
function setStress(j){if(j&&j.html!==undefined)document.getElementById('stresspanel').innerHTML=j.html}
async function loadStress(){try{setStress(await (await fetch('/author/__TID__/stresses')).json())}catch(e){}}
async function balanceStress(sid,val){await post('/author/__TID__/stress/'+sid+'/balance',{value:val});loadStress()}
async function saveHeader(){const j=await post('/author/__TID__/header',
 {name:val('h_name'),email:val('h_email'),date:val('h_date')});astat('Header saved.');setE4L(j)}
// --- E4L client picker: name autocomplete -> email (dropdown if duplicates) -> date
var E4L_CLIENT_ID=null;
function _esc(s){var e=document.createElement('div');e.textContent=(s==null?'':s);return e.innerHTML}
function _today(){return new Date().toISOString().slice(0,10)}
function hideDD(){var d=document.getElementById('h_dd');if(d){d.style.display='none';d.innerHTML='';d._clients=null;d._emails=null}}
async function nameSearch(){
 var q=val('h_name'),d=document.getElementById('h_dd');
 if(!q||q.length<2){hideDD();return}
 try{var cs=((await (await fetch('/api/e4l/clients?q='+encodeURIComponent(q))).json()).clients)||[];
  if(!cs.length){hideDD();return}
  d.innerHTML=cs.map(function(c,i){
   var n=c.emails?c.emails.length:0;
   var sub=n>1?(' <span class=food>('+n+' emails)</span>'):(n==1?(' <span class=food>'+_esc(c.emails[0].email)+'</span>'):'');
   return '<div class=ddi data-i="'+i+'">'+_esc(c.name)+sub+'</div>'}).join('');
  d._clients=cs;d.style.display='block'}catch(e){hideDD()}
}
function showEmailPicker(emails){
 var d=document.getElementById('h_dd');
 d.innerHTML='<div class=food style="padding:5px 10px">Two clients share this name &mdash; pick the email:</div>'+
  emails.map(function(e,i){return '<div class=ddi data-ei="'+i+'">'+_esc(e.email)+
   (e.last_scan_date?(' <span class=food>(last scan '+_esc(e.last_scan_date)+')</span>'):'')+'</div>'}).join('');
 d._emails=emails;d.style.display='block';
}
function pickName(c){
 set('h_name',c.name);
 if(!val('h_date'))set('h_date',_today());
 if(!c.emails||c.emails.length<=1){var em=(c.emails&&c.emails[0])||{};set('h_email',em.email||'');
  E4L_CLIENT_ID=em.client_id!=null?em.client_id:null;hideDD();afterClientSelected()}
 else{showEmailPicker(c.emails)}
}
function pickEmail(e){set('h_email',e.email);E4L_CLIENT_ID=e.client_id!=null?e.client_id:null;hideDD();afterClientSelected()}
async function afterClientSelected(){set('h_client_id',E4L_CLIENT_ID==null?'':E4L_CLIENT_ID);await saveHeader();checkE4L()}
document.addEventListener('click',function(ev){
 var d=document.getElementById('h_dd');if(!d)return;
 var it=ev.target.closest?ev.target.closest('.ddi'):null;
 if(it&&d.contains(it)){
  if(it.dataset.ei!==undefined&&d._emails){pickEmail(d._emails[+it.dataset.ei])}
  else if(it.dataset.i!==undefined&&d._clients){pickName(d._clients[+it.dataset.i])}
 }else if(!(ev.target.id==='h_name')){hideDD()}
});
async function checkE4L(){
 var s=document.getElementById('e4lchk');if(s)s.textContent='Checking E4L for a newer scan\\u2026';
 try{var cid=val('h_client_id');
  var j=await post('/author/__TID__/e4l/refresh',{client_id:cid?Number(cid):(E4L_CLIENT_ID!=null?E4L_CLIENT_ID:null)});
  setE4L(j);var s2=document.getElementById('e4lchk');
  if(s2)s2.textContent=j.ok?(j.newer?'\\u2191 Newer scan pulled.':'\\u2713 Up to date.'):('E4L check failed: '+((j.error||'error')+'').slice(0,120));
 }catch(e){var s3=document.getElementById('e4lchk');if(s3)s3.textContent='E4L check failed.'}
}
async function addRow(){var b=rowVals('new');if(!b.head&&!b.remedy){astat('Enter a stress and a remedy.');return}
 await post('/author/__TID__/row',b);location.reload()}
async function saveRow(rid){await post('/author/__TID__/row/'+rid,rowVals('r'+rid));astat('Row saved.')}
async function delRow(rid){if(!confirm('Delete this row?'))return;
 await post('/author/__TID__/row/'+rid+'/delete',{});location.reload()}
async function fillDose(p){var n=val(p+'_remedy');if(!n)return;
 const r=await (await fetch('/api/dosing?name='+encodeURIComponent(n))).json();
 if(r.dosage)set(p+'_dosage',r.dosage);if(r.frequency)set(p+'_frequency',r.frequency);
 if(r.timing)set(p+'_timing',r.timing);astat('Dosing filled from catalog.')}
async function suggest(p){var s=val(p+'_head');var box=document.getElementById(p+'_sug');box.textContent='';
 if(!s){astat('Enter a stress first.');return}
 const r=await (await fetch('/api/suggest?stress='+encodeURIComponent(s))).json();var arr=r.suggestions||[];
 if(!arr.length){box.textContent='no history for that stress';return}
 box.appendChild(document.createTextNode('Used before: '));
 arr.forEach(function(x){var b=document.createElement('button');b.type='button';b.className='chip';
  b.textContent=x.remedy+' ('+x.count+')';b.onclick=function(){set(p+'_remedy',x.remedy);fillDose(p)};
  box.appendChild(b);box.appendChild(document.createTextNode(' '))})}
async function saveDepth(el){await post('/author/__TID__/depth',
 {rid:el.dataset.rid,side:el.dataset.side,rank:el.value});astat('Depth saved.')}
function rstat(t){document.getElementById('rstat').textContent=t}
var _mr,_dg,_sess='';
async function recStart(){
 rstat('Getting token...');
 _sess=(document.getElementById('sessText').value||'');
 var t;try{t=await (await fetch('/api/deepgram-token')).json()}catch(e){rstat('Token fetch failed: '+e);return}
 if(!t.key){rstat('No Deepgram key: '+(t.error||''));return}
 var stream;try{stream=await navigator.mediaDevices.getUserMedia({audio:true})}
 catch(e){rstat('Microphone blocked/denied: '+e.name);return}
 var mime='';['audio/webm;codecs=opus','audio/webm','audio/ogg;codecs=opus','audio/mp4'].forEach(
  function(m){if(!mime&&window.MediaRecorder&&MediaRecorder.isTypeSupported(m))mime=m});
 if(!mime){rstat('No supported audio recording format in this browser. Use Chrome.');return}
 rstat('Mic OK ('+mime+'). Connecting to Deepgram...');
 try{_dg=new WebSocket('wss://api.deepgram.com/v1/listen?model=nova-2&smart_format=true'+
  '&punctuate=true&interim_results=true',['token',t.key])}
 catch(e){rstat('WebSocket create failed: '+e);return}
 _dg.onopen=function(){
  try{_mr=new MediaRecorder(stream,{mimeType:mime})}catch(e){rstat('Recorder error: '+e);return}
  _mr.ondataavailable=function(e){if(e.data.size>0&&_dg.readyState===1)_dg.send(e.data)};
  _mr.start(250);rstat('Recording \\u2014 speak naturally. (codes: wear a lav/AirPods)');console.log('rec open, mime',mime)};
 _dg.onmessage=function(m){var d;try{d=JSON.parse(m.data)}catch(e){return}
  console.log('dg msg',d.type,d);
  if(d.type&&d.type!=='Results')return;
  var a=d.channel&&d.channel.alternatives&&d.channel.alternatives[0];if(!a)return;
  if(d.is_final&&a.transcript){_sess+=(_sess?' ':'')+a.transcript;
   document.getElementById('sessText').value=_sess;document.getElementById('interim').textContent=''}
  else if(a.transcript){document.getElementById('interim').textContent=a.transcript}};
 _dg.onerror=function(e){rstat('WebSocket error (see console).');console.log('dg error',e)};
 _dg.onclose=function(e){console.log('dg close',e.code,e.reason);
  if(_mr&&_mr.state!=='inactive')_mr.stop();
  if((_sess||'').length===0)rstat('Connection closed (code '+e.code+') '+(e.reason||'')+' \\u2014 nothing transcribed.')};
}
async function recStop(){
 if(_mr&&_mr.state!=='inactive')_mr.stop();
 if(_mr&&_mr.stream)_mr.stream.getTracks().forEach(function(t){t.stop()});
 if(_dg&&_dg.readyState===1){_dg.send(JSON.stringify({type:'CloseStream'}));_dg.close()}
 rstat('Saving\\u2026');
 await post('/author/__TID__/session',{transcript:document.getElementById('sessText').value});
 rstat('Saved to notes; it feeds the narrative.')}
async function interpret(){rstat('Interpreting transcript into chain rows\\u2026');
 var r=await post('/author/__TID__/interpret',{});
 if(r.error){rstat('Interpret: '+r.error);return}
 rstat('Filled '+r.added+' row(s) \\u2014 highlighted for review; reloading\\u2026');
 setTimeout(function(){location.reload()},800)}
async function delTest(){if(!confirm('Delete this entire test? This cannot be undone.'))return;
 await post('/author/__TID__/delete',{});location.href='/'}
async function confirmAll(){await post('/author/__TID__/confirm-all',{});location.reload()}
async function confirmRow(rid){await post('/author/__TID__/row/'+rid+'/confirm',{});location.reload()}
async function importReveal(){
try{
  var j=await post('/author/__TID__/e4l/import-reveal',{});
  if(j && j.needs_confirm){
    if(!confirm('This session already has '+j.existing+' rows — add the reveal layers anyway?')) return;
    j=await post('/author/__TID__/e4l/import-reveal',{force:true});
  }
  if(j && j.ok){ location.reload(); }
  else { astat((j&&j.reason)||'Import failed.'); }
}catch(e){ astat('Import failed.'); }
}
async function loadLists(){
 try{const v=await (await fetch('/api/vocab?limit=500')).json();
  document.getElementById('vocab').innerHTML=(v.vocab||[]).map(opt).join('')}catch(e){}
 try{const c=await (await fetch('/api/catalog?limit=800')).json();
  document.getElementById('catalog').innerHTML=(c.catalog||[]).map(function(x){return opt(x.name||'')}).join('')}catch(e){}
}
function setPhase(p){window._phase=p;
 document.getElementById('phaseCap').className=(p==1?'btn':'btn ghost');
 document.getElementById('phaseBal').className=(p==2?'btn':'btn ghost');
 document.getElementById('phaseAct').textContent=(p==1?'Capture stresses → list':'Interpret → fill fields')}
async function phaseRun(){if((window._phase||1)==1){captureStresses()}else{interpret()}}
async function captureStresses(){rstat('Capturing stresses from transcript…');
 var j=await post('/author/__TID__/capture-stresses',{});
 if(j.error){rstat('Capture: '+j.error);return}
 rstat('Added '+j.added+' stress(es).');loadStress()}
async function mineProfile(){rstat('Mining client profile for stresses…');
 var j=await post('/author/__TID__/mine-profile',{});
 if(j.error){rstat('Mine profile: '+j.error);return}
 rstat('Added '+j.added+' profile stress(es).');loadStress()}
loadLists();
loadE4L();
loadStress();
setPhase(1);
</script>"""


def _row_inputs(p, l):
    layer = "" if l.get("layer") is None else _e(str(l.get("layer")))
    g = lambda k: _e(l.get(k) or "")
    return (
        f'<td><input id="{p}_layer" class="lyr" value="{layer}"></td>'
        f'<td><input id="{p}_head" list="vocab" value="{g("head")}"></td>'
        f'<td><input id="{p}_most" value="{g("most_affected")}"></td>'
        f"<td><input id=\"{p}_remedy\" list=\"catalog\" value=\"{g('remedy')}\""
        f" onchange=\"fillDose('{p}')\"></td>"
        f'<td><input id="{p}_dosage" value="{g("dosage")}"></td>'
        f'<td><input id="{p}_frequency" value="{g("frequency")}"></td>'
        f'<td><input id="{p}_timing" value="{g("timing")}"></td>')


def render_e4l_panel(ctx):
    """Reference panel for the most recent E4L voice scan (fresh / stale / none).
    Always shows the scan's age + ranked findings when one exists; read-only —
    Glen's spoken testing still fills the causal chain. All scan free-text escaped."""
    ctx = ctx or {}
    status = ctx.get("status") or "none"
    color = {"fresh": "var(--ok)", "stale": "var(--accent)"}.get(status, "var(--muted)")
    icon = {"fresh": "&#9679;", "stale": "&#9888;&#65039;"}.get(status, "&#9675;")
    head = (f"<div style='display:flex;align-items:center;gap:8px;font-weight:600;color:{color}'>"
            f"<span>{icon}</span><span>{_e(ctx.get('message') or '')}</span></div>")
    date = _e(ctx.get("scan_date") or "")
    sub = (f"<div class=food style='margin-top:2px'>scan {date}</div>"
           if ctx.get("found") and date else "")
    def _list(findings):
        items = ""
        for f in findings or []:
            rank = _e(str(f.get("rank"))) if f.get("rank") is not None else ""
            desc = _e(f.get("description") or "")
            items += (f"<li><b>{_e(f.get('code') or '')}</b> {_e(f.get('name') or '')}"
                      + (f" &mdash; <span class=food>{desc}</span>" if desc else "")
                      + (f" <span class=pill>#{rank}</span>" if rank else "") + "</li>")
        return f"<ol style='margin:4px 0 0;padding-left:20px'>{items}</ol>" if items else ""

    def _section(label, sub, findings):
        if not findings:
            return ""
        return (f"<div style='margin-top:8px'><div class=food style='font-weight:600'>"
                f"{label}{(' &mdash; ' + sub) if sub else ''}</div>{_list(findings)}</div>")

    # Two lists: infoceuticals Glen balances vs. ER/MR "stresses" (info only). Fall
    # back to splitting `findings` by group for any caller that didn't pre-split.
    info = ctx.get("infoceuticals")
    stress = ctx.get("stresses")
    if info is None and stress is None:
        allf = ctx.get("findings") or []
        info = [f for f in allf if f.get("group") != "stress"]
        stress = [f for f in allf if f.get("group") == "stress"]
    body = (_section("Infoceuticals", "", info)
            + _section("Stresses", "information only, no balancing vial", stress))
    note = ("<div class=food style='margin-top:6px'>Reference only &mdash; your spoken "
            "testing fills the chain.</div>") if ctx.get("found") else ""
    days = ctx.get("days_ago")
    if ctx.get("found") and days is not None and days < 7:
        imp = "<button class='btn' onclick=importReveal()>Import Reveal &rarr; Causal Chain</button>"
    elif ctx.get("found"):
        imp = (f"<button class='btn' disabled title='Refresh to a scan under 7 days old'>"
               f"Import Reveal &rarr; Causal Chain</button>"
               f"<span class=food>scan is {_e(str(days))} days old</span>")
    else:
        imp = ""
    check = ("<div class=btnrow style='margin-top:8px'>"
             "<button class='btn ghost' onclick=checkE4L()>Check E4L now</button>"
             f"{imp}"
             "<span id=e4lchk class=food></span></div>")
    return (f"<div class=card style='border-left:3px solid {color}'>"
            "<div class=food style='text-transform:uppercase;font-size:11px;letter-spacing:.08em'>"
            f"Recent E4L voice scan</div>{head}{sub}{body}{note}{check}</div>")


def _depth_select(rid, side, current, depth_values):
    opts = "<option value=''>&mdash;</option>"
    for v in depth_values or []:
        sel = " selected" if (current is not None and int(current) == v["rank"]) else ""
        opts += f"<option value='{v['rank']}'{sel}>{_e(v['value'])}</option>"
    return (f"<select data-rid=\"{_e(str(rid))}\" data-side=\"{side}\" onchange=\"saveDepth(this)\" "
            f"style='font-size:12px;max-width:170px'>{opts}</select>")


def render_author_html(report, depth_values=None, transcript=""):
    tid = _e(report.get("test_id") or "")
    c = report.get("client") or {}
    head = (f"<p><a href='/'>&larr; All tests</a> &nbsp;&middot;&nbsp; "
            f"<a href='/test/{tid}'>View report &rarr;</a></p><h1>Edit Biofield Test</h1>"
            "<div class=btnrow><button class=btn onclick=confirmAll()>&#10003; Confirm all rows</button>"
            "<button class='btn ghost' onclick=delTest()>Delete test</button></div>")
    hdr = (
        "<style>.dd{position:absolute;top:100%;left:0;display:none;background:var(--card);"
        "border:1px solid var(--line);border-radius:6px;margin-top:2px;min-width:320px;"
        "max-width:520px;max-height:280px;overflow:auto;z-index:50}"
        ".ddi{padding:6px 10px;cursor:pointer;border-bottom:1px solid var(--line)}"
        ".ddi:hover{background:rgba(255,255,255,.06)}</style>"
        "<div class=card>"
        "<input type=hidden id=h_client_id value=''>"
        "<label>Client name</label>"
        "<span style='position:relative;display:inline-block'>"
        f"<input id=h_name autocomplete=off oninput=nameSearch() value=\"{_e(c.get('name') or '')}\" style='width:280px'>"
        "<div id=h_dd class=dd></div></span>"
        f"<label>Email</label><input id=h_email value=\"{_e(c.get('email') or '')}\" style='width:280px'>"
        f"<label>Date</label><input id=h_date value=\"{_e(report.get('date') or '')}\" style='width:160px'>"
        "<div class=btnrow><button class=btn onclick=saveHeader()>Save header</button>"
        "<span id=astat class=food></span></div></div>")
    rows = ""
    shown_divider = False
    for l in report.get("layers") or []:
        if not shown_divider and l.get("zone") == "bottom":
            rows += ("<tr><td colspan=10 style='text-align:center;color:var(--muted);"
                     "font-size:12px;padding:4px 0'><b>Unbalanced from scan</b></td></tr>")
            shown_divider = True
        rid_raw = l.get("rid")
        rid = _e(str(rid_raw or ""))
        p = "r" + rid
        depth_cell = ("<td><span class=food>stress</span> "
                      + _depth_select(rid_raw, "stress", l.get("stress_depth"), depth_values)
                      + "<br><span class=food>remedy</span> "
                      + _depth_select(rid_raw, "remedy", l.get("remedy_depth"), depth_values) + "</td>")
        cls = " class=unconf" if l.get("confirmed") == 0 else ""
        confirm_btn = (f"<button class=chip onclick=\"confirmRow('{rid}')\">&#10003; confirm</button> "
                       if l.get("confirmed") == 0 else "")
        rows += (f"<tr{cls}>" + _row_inputs(p, l) + depth_cell +
                 f"<td><button class=chip onclick=\"fillDose('{p}')\">dose</button> "
                 f"<button class=chip onclick=\"suggest('{p}')\">uses</button></td>"
                 f"<td>{confirm_btn}<button class=btn onclick=\"saveRow('{rid}')\">Save</button> "
                 f"<button class='btn ghost' onclick=\"delRow('{rid}')\">Del</button></td></tr>"
                 f"<tr><td colspan=10><span id={p}_sug class=food></span></td></tr>")
    addr = ("<tr>" + _row_inputs("new", {}) +
            "<td class=food>save row first</td>"
            "<td><button class=chip onclick=\"fillDose('new')\">dose</button> "
            "<button class=chip onclick=\"suggest('new')\">uses</button></td>"
            "<td><button class=btn onclick=addRow()>Add row</button></td></tr>"
            "<tr><td colspan=10><span id=new_sug class=food></span></td></tr>")
    table = ("<h2>Causal chain</h2>"
             "<p class=sub>Enter rows directly. Layer 1 = most recent/surface, higher = deeper root. "
             "Dosage / frequency / timing auto-fill from the catalog (minimum dose) the moment you pick a "
             "remedy, and stay editable; 'uses' shows what you've used for that stress before. "
             "Set depth-of-penetration on the stress and the remedy &mdash; a remedy shallower than its "
             "stress is flagged on the report.</p>"
             "<table><tr><th>Layer</th><th>Head / Stress</th><th>Most Affected</th>"
             "<th>Remedy</th><th>Dosage</th><th>Frequency</th><th>Timing</th>"
             "<th>Depth of penetration</th><th></th><th></th></tr>"
             + rows + addr + "</table>"
             "<datalist id=vocab></datalist><datalist id=catalog></datalist>")
    session = (
        "<h2>Live session (voice)</h2>"
        "<p class=sub>Record yourself narrating the test in your own voice &mdash; the live "
        "transcript saves to this test's notes and feeds the narrative. Wear a lav/AirPods for "
        "the codes.</p>"
        "<div class=btnrow style='margin-bottom:6px'>"
        "<button id=phaseCap class=btn onclick='setPhase(1)'>Phase 1 &middot; Capture stresses</button>"
        "<button id=phaseBal class='btn ghost' onclick='setPhase(2)'>Phase 2 &middot; Balance</button>"
        "</div>"
        "<div class=btnrow>"
        "<button class=btn onclick=recStart()>&#9679; Record</button>"
        "<button class='btn ghost' onclick=recStop()>&#9632; Stop &amp; save</button>"
        "<button id=phaseAct class=btn onclick=phaseRun()>Capture stresses &rarr; list</button>"
        "<span id=rstat class=food></span></div>"
        "<div class=food><em id=interim></em></div>"
        f"<textarea id=sessText rows=6 placeholder='Live transcript appears here as you speak..."
        f"'>{_e(transcript)}</textarea>")
    return _page("Edit Biofield Test",
                 head + hdr + "<div id=e4lpanel></div>"
                 "<div class=btnrow style='margin:6px 0'>"
                 "<button class='btn ghost' onclick=mineProfile()>Mine profile &rarr; stresses</button>"
                 "</div>"
                 "<div id=stresspanel></div>" + table + session
                 + _AUTHOR_JS.replace("__TID__", tid))


def render_list_html(tests, q="", authored=None):
    authored = authored or []
    arows = ""
    for t in authored:
        atid = _e(str(t.get("test_id") or ""))
        arows += (f"<tr><td><a href='/author/{atid}'>{_e(t.get('name') or '(unnamed)')}</a> "
                  f"<a class=food href='/test/{atid}'>(report)</a></td>"
                  f"<td>{_e(t.get('email') or '')}</td><td>{_e(t.get('date') or '')}</td>"
                  f"<td><span class=pill>{_e(str(t.get('layer_count') or 0))}</span></td></tr>")
    asection = ("<h2>Your authored tests</h2>"
                "<table><tr><th>Client</th><th>Email</th><th>Date</th><th>Remedies</th></tr>"
                + (arows or "<tr><td colspan=4 class=food>None yet — click New test.</td></tr>")
                + "</table>")
    rows = ""
    for t in tests or []:
        rows += (
            "<tr>"
            f"<td><a href='/test/{_e(str(t.get('test_id') or ''))}'>{_e(t.get('name') or '(unknown)')}</a></td>"
            f"<td>{_e(t.get('email') or '')}</td>"
            f"<td>{_e(t.get('date') or '')}</td>"
            f"<td><span class=pill>{_e(str(t.get('layer_count') or 0))}</span></td>"
            "</tr>")
    body = (
        "<h1>Biofield Analysis</h1>"
        "<p class=sub>Causal Chain Reports — local, from your FileMaker data and your own authored tests.</p>"
        "<form method=post action='/author/new'><button class=btn type=submit>+ New test</button></form>"
        + asection +
        "<form method=get><input type=search name=q placeholder='Search FileMaker tests' "
        f"value='{_e(q or '')}'></form>"
        "<h2>FileMaker tests</h2>"
        "<table><tr><th>Client</th><th>Email</th><th>Date</th><th>Remedies</th></tr>"
        f"{rows}</table>")
    return _page("Biofield Analysis", body)


def render_suggest_panel(data):
    return ""


def render_stress_panel(data):
    data = data or {}
    def _row(s, active):
        tag = _e(s.get("balance") or "")
        by = _e(s.get("balanced_by") or "")
        bytxt = f" <span class=food>&middot; {by}</span>" if (not active and by) else ""
        btn = (f"<button class='btn ghost' style='font-size:11px' "
               f"onclick=\"balanceStress({int(s.get('id') or 0)},{'true' if active else 'false'})\">"
               f"{'Balance' if active else 'Reactivate'}</button>")
        return (f"<li><b>{_e(s.get('code') or '')}</b> {_e(s.get('label') or '')} "
                f"<span class=pill>{tag}</span>{bytxt} {btn}</li>")
    act = "".join(_row(s, True) for s in data.get("active") or [])
    bal = "".join(_row(s, False) for s in data.get("balanced") or [])
    act_html = (f"<div class=food style='font-weight:600;margin-top:6px'>Active &mdash; to balance</div>"
                f"<ul style='margin:4px 0;padding-left:18px'>{act}</ul>") if act else (
                "<div class=food style='margin-top:6px'>No active stresses.</div>")
    bal_html = (f"<div class=food style='font-weight:600;margin-top:6px'>Balanced</div>"
                f"<ul style='margin:4px 0;padding-left:18px'>{bal}</ul>") if bal else ""
    return ("<div class=card><div class=food style='text-transform:uppercase;font-size:11px;"
            "letter-spacing:.08em'>Stress balancing</div>" + act_html + bal_html + "</div>")
