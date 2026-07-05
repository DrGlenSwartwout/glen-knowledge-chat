"""Local Community cataloging tool — runs on Glen's Mac, NOT the deployed app.

  python3 community_local_app.py --port 8012
  open http://127.0.0.1:8012

Point it at a source recording file + the Rumble unlisted URL of the full
published replay. It runs Whisper + an LLM to suggest a title, interest tags,
and out-take moments; you approve; it cuts the out-takes with ffmpeg, uploads
them, and publishes the catalog entry to prod. Mirrors biofield_local_app.py."""

import argparse
import os
import tempfile

from flask import Flask, request, jsonify

from dashboard.community_catalog import (transcribe, suggest_catalog,
                                         cut_outtakes, publish_session)

_PAGE = """<!doctype html><html><head><meta charset=utf-8>
<title>Community cataloging</title></head><body style="font-family:system-ui;max-width:820px;margin:2rem auto">
<h1>Community cataloging</h1>
<p>Point at a recording file and paste the Rumble unlisted link of the full replay.</p>
<label>Source file path <input id=path size=60></label><br>
<label>Rumble URL <input id=rumble size=60></label><br>
<label>Type
 <select id=type><option value=coaching_replay>Coaching replay</option>
 <option value=course_session>Course session</option></select></label><br>
<button onclick=analyze()>Analyze</button>
<div id=out></div>
<!-- BEGIN community local script -->
<script>
async function analyze(){
  const b={path:path.value,rumble_url:rumble.value,type:type.value};
  const r=await fetch('/analyze',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify(b)});
  const d=await r.json(); render(d);
}
function render(d){
  const s=d.suggestions||{};
  out.innerHTML='<h2>Suggested</h2>';
  const t=document.createElement('div');
  t.innerHTML='Title <input id=title size=60>';
  out.appendChild(t); document.getElementById('title').value=s.title||'';
  const tags=document.createElement('div');
  tags.innerHTML='Tags (comma) <input id=tags size=60>';
  out.appendChild(tags); document.getElementById('tags').value=(s.interest_tags||[]).join(', ');
  out.appendChild(document.createElement('hr'));
  window._outtakes=(s.outtakes||[]);
  (s.outtakes||[]).forEach((o,i)=>{
    const d2=document.createElement('div');
    const cap=document.createTextNode(' ['+o.start+'-'+o.end+'] '+(o.reason||''));
    const cb=document.createElement('input');cb.type='checkbox';cb.checked=true;cb.id='ot'+i;
    const ti=document.createElement('input');ti.id='ott'+i;ti.size=40;ti.value=o.title||'';
    d2.appendChild(cb);d2.appendChild(ti);d2.appendChild(cap);out.appendChild(d2);
  });
  const pb=document.createElement('button');pb.textContent='Publish';pb.onclick=publish;out.appendChild(pb);
  window._ctx={path:path.value,rumble:rumble.value,type:type.value,transcript:d.transcript||''};
}
async function publish(){
  const outs=(window._outtakes||[]).map((o,i)=>({start:o.start,end:o.end,
     title:(document.getElementById('ott'+i)||{}).value||o.title,
     interest_tags:[], _on:(document.getElementById('ot'+i)||{}).checked}))
     .filter(o=>o._on).map(({_on,...r})=>r);
  const tags=document.getElementById('tags').value.split(',').map(x=>x.trim()).filter(Boolean);
  const full={type:window._ctx.type,title:document.getElementById('title').value,
     description:'',video_ref:window._ctx.rumble,interest_tags:tags,transcript:window._ctx.transcript};
  const r=await fetch('/publish',{method:'POST',headers:{'Content-Type':'application/json'},
     body:JSON.stringify({path:window._ctx.path,full:full,outtakes:outs})});
  const d=await r.json();out.innerHTML='<p>Published. content_id='+(d.content_id||'?')+', out-takes='+(d.outtakes||0)+'</p>';
}
</script>
<!-- END community local script -->
</body></html>"""


def create_app():
    app = Flask(__name__)

    @app.route("/")
    def index():
        return _PAGE

    @app.route("/analyze", methods=["POST"])
    def analyze():
        b = request.get_json(force=True) or {}
        tr = transcribe(b["path"])
        sug = suggest_catalog(tr["text"])
        return jsonify({"transcript": tr["text"], "suggestions": sug})

    @app.route("/publish", methods=["POST"])
    def publish():
        b = request.get_json(force=True) or {}
        base = os.environ.get("PUBLIC_BASE_URL", "https://illtowell.com").rstrip("/")
        key = os.environ["CONSOLE_SECRET"]
        with tempfile.TemporaryDirectory() as wd:
            files = cut_outtakes(b["path"], b.get("outtakes", []), workdir=wd)
            resp = publish_session(base_url=base, console_key=key,
                                   full=b["full"], outtake_files=files)
        return jsonify(resp)

    return app


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", type=int, default=8012)
    args = ap.parse_args()
    create_app().run(host="127.0.0.1", port=args.port, debug=False)
