"""Local-only Biofield Analysis viewer (runs on Glen's Mac).

Reads the FileMaker snapshot in chat_log.db (`fmp_snap_*` tables) and serves the
layer-ordered Causal Chain Report + remedy schedule in the browser. All patient
data stays on this machine — DO NOT deploy this to a server.

Setup / refresh:
  1. Open the New App "Remedy Match.fmp12" in FileMaker Pro 18 Advanced.
  2. Extract:  ~/.venvs/fmp-ingest/bin/python3 \
       "$HOME/AI-Training/02 Skills/fmp-applescript-extract.py" \
       --database 'Remedy Match.fmp12' --source newapp \
       --tables clients,client_biofield_test,client_active_main_stress,\
client_active_stress_affects,client_causal_chain,client_causal_chain_sec_affects,\
client_nosode,client_remedy,client_foods,products,products_phases,products_systems,products_items
  3. Load + serve:  python3 biofield_local_app.py --load

Then open http://127.0.0.1:8011
"""
import argparse
import datetime
import os
import sqlite3

from flask import Flask, Response, redirect, request, send_from_directory

from dashboard.biofield_report import causal_chain_report, list_tests
from dashboard.biofield_report_html import (
    render_author_html, render_e4l_panel, render_list_html, render_report_html,
    render_stress_panel)
from dashboard.biofield_e4l import (
    fetch_live as _fetch_live, scan_context as _scan_context,
    search_clients as _search_clients)
from dashboard.biofield_narrative import (
    generate_narrative, generate_video_script, get_narrative, get_notes,
    get_video_script, save_narrative, save_notes, save_video_script)
from dashboard.biofield_authoring import (
    add_chain_row, authored_report, confirm_all, confirm_row, create_test,
    delete_chain_row, delete_test, list_authored, remedy_catalog, remedy_dosing,
    resolve_remedy_name, resolve_stress_name, stress_suggestions, stress_vocab,
    update_chain_row, update_header)
from dashboard.biofield_dimensions import (
    DEPTH_KEY, dimension_values, seed_dimensions, tag as dim_tag)
from dashboard.biofield_interpret import interpret_transcript
from dashboard.biofield_report_present import render_present
from dashboard.biofield_report_pdf import report_pdf_bytes, save_report_pdf

AUDIO_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "biofield-audio")


def _layer_int(v):
    try:
        return int(str(v).strip())
    except (TypeError, ValueError):
        return None


def _safe_int(v, default):
    try:
        return int(v)
    except (TypeError, ValueError):
        return default


def openai_complete(system, user):
    """Default narrative LLM. Needs OPENAI_API_KEY in env (run via `doppler run`)."""
    import openai
    model = os.environ.get("BIOFIELD_NARRATIVE_MODEL", "gpt-4o")
    resp = openai.OpenAI().chat.completions.create(
        model=model,
        temperature=0.4,  # grounded, less flowery
        messages=[{"role": "system", "content": system},
                  {"role": "user", "content": user}])
    return resp.choices[0].message.content


def openai_json(system, user):
    """Deterministic JSON completion for the transcript interpreter (temp 0, JSON mode)."""
    import openai
    model = os.environ.get("BIOFIELD_NARRATIVE_MODEL", "gpt-4o")
    resp = openai.OpenAI().chat.completions.create(
        model=model, temperature=0, response_format={"type": "json_object"},
        messages=[{"role": "system", "content": system},
                  {"role": "user", "content": user}])
    return resp.choices[0].message.content


def elevenlabs_tts(text):
    """Render text to mp3 bytes in Glen's cloned voice. Needs ELEVENLABS_API_KEY."""
    import requests
    voice = os.environ.get("ELEVENLABS_VOICE_ID", "jFxSqMckq2I4mET3C5QC")
    r = requests.post(
        f"https://api.elevenlabs.io/v1/text-to-speech/{voice}",
        headers={"xi-api-key": os.environ["ELEVENLABS_API_KEY"],
                 "Content-Type": "application/json", "Accept": "audio/mpeg"},
        json={"text": text, "model_id": "eleven_multilingual_v2",
              "voice_settings": {"stability": 0.5, "similarity_boost": 0.75}},
        timeout=180)
    r.raise_for_status()
    return r.content


def deepgram_temp_key():
    """Mint a short-lived (1h) Deepgram key so the long-lived key never reaches the
    browser. Needs DEEPGRAM_API_KEY in env (run via `doppler run`)."""
    import requests
    h = {"Authorization": "Token " + os.environ["DEEPGRAM_API_KEY"]}
    pid = requests.get("https://api.deepgram.com/v1/projects", headers=h,
                       timeout=20).json()["projects"][0]["project_id"]
    r = requests.post(f"https://api.deepgram.com/v1/projects/{pid}/keys", headers=h, timeout=20,
                      json={"comment": "biofield-live-session", "scopes": ["usage:write"],
                            "time_to_live_in_seconds": 3600})
    r.raise_for_status()
    return r.json()["key"]


def deepgram_browser_token():
    """Token for the browser's Deepgram socket. Prefer a short-lived key; fall back to
    the env key if this key lacks key-management scope (fine: localhost-only tool, the
    token only ever reaches Glen's own browser)."""
    try:
        return deepgram_temp_key()
    except Exception:
        return os.environ["DEEPGRAM_API_KEY"]

DEFAULT_DB = os.environ.get(
    "BIOFIELD_DB", os.path.join(os.path.dirname(os.path.abspath(__file__)), "chat_log.db"))


def create_app(db_path=DEFAULT_DB, complete=None, tts=None, deepgram_token=None,
               interpret_complete=None, scan_lookup=None, client_search=None,
               fetch_runner=None):
    app = Flask(__name__)
    complete = complete or openai_complete
    tts = tts or elevenlabs_tts
    deepgram_token = deepgram_token or deepgram_browser_token
    interpret_complete = interpret_complete or openai_json
    # E4L scan pull: as soon as the client (email) is known, surface their most recent
    # voice scan. Default reads ~/AI-Training/e4l.db read-only as of today's date;
    # injectable for tests. Never raises -> intake is never blocked.
    scan_lookup = scan_lookup or (
        lambda email: _scan_context(email, datetime.date.today().isoformat()))
    # Name picker + on-demand live fetch (the local mirror lags, so selecting a client
    # can pull their newest scan straight from the live E4L portal). Both injectable.
    client_search = client_search or (lambda q: _search_clients(q))
    fetch_runner = fetch_runner  # None -> fetch_live uses the real scraper+parser

    def _report_for(cx, test_id):
        return (authored_report(cx, test_id) if str(test_id).startswith("a")
                else causal_chain_report(cx, test_id))

    def _e4l(cx, test_id):
        """Scan context + rendered panel for a test's stored client email."""
        rep = _report_for(cx, test_id)
        ctx = scan_lookup((rep.get("client") or {}).get("email") or "")
        return ctx, rep

    def _seed_stresses(cx, test_id, *, force=False, layers=None):
        """Synthesize reveal layers + seed the stress coverage map for this test.
        Skips silently when no email is set, no scan is found, or (unless force)
        the test already has scan stresses. Synthesis errors are swallowed so the
        intake flow is never blocked by a live-pipeline failure.
        If *layers* is supplied they are used directly, skipping the synthesis
        network call — pass them from a route that already ran synthesize_reveal_layers
        so the pipeline runs only once per import."""
        from dashboard import biofield_reveal_import as _ri
        from dashboard import biofield_stress as _st
        import datetime as _dt
        rep = _report_for(cx, test_id)
        email = ((rep.get("client") or {}).get("email") or "").strip()
        if not email:
            return
        _st.init_stress_tables(cx)
        if not force and cx.execute(
                "SELECT 1 FROM biofield_auth_stress WHERE test_id=? AND source='scan' LIMIT 1",
                (int(str(test_id).lstrip("a") or 0),)).fetchone():
            return
        ctx = scan_lookup(email)
        if not ctx.get("found"):
            return
        if layers is not None:
            coverage = _ri.build_coverage(layers)
        else:
            try:
                res = _ri.synthesize_reveal_layers(email, today=_dt.date.today().isoformat())
            except Exception:
                return
            coverage = _ri.build_coverage(res.get("layers") or [])
        _st.seed_from_scan(cx, test_id, ctx.get("findings") or [], coverage)

    with sqlite3.connect(db_path) as _cx:
        seed_dimensions(_cx)

    # Same console key as the rest of Glen's console (when CONSOLE_SECRET is set, e.g.
    # under `doppler run`). The launcher passes ?key=; we cookie it so same-origin
    # fetches stay authed. No CONSOLE_SECRET -> open (e.g. bare local dev / tests).
    _secret = os.environ.get("CONSOLE_SECRET", "")

    @app.before_request
    def _console_gate():
        if not _secret:
            return None
        key = (request.args.get("key", "") or request.cookies.get("rm_biofield_key", "")
               or request.headers.get("X-Console-Key", ""))
        if key != _secret:
            return Response("Unauthorized — open this from the console 'Biofield Intake' link.",
                            status=401, mimetype="text/plain")
        return None

    @app.after_request
    def _console_cookie(resp):
        if _secret and request.args.get("key", "") == _secret:
            resp.set_cookie("rm_biofield_key", _secret, httponly=True, samesite="Lax")
        return resp

    @app.route("/")
    def index():
        q = request.args.get("q", "")
        with sqlite3.connect(db_path) as cx:
            return Response(render_list_html(list_tests(cx, q), q, list_authored(cx)),
                            mimetype="text/html")

    @app.route("/test/<test_id>")
    def report(test_id):
        with sqlite3.connect(db_path) as cx:
            rep = (authored_report(cx, test_id) if str(test_id).startswith("a")
                   else causal_chain_report(cx, test_id))
            notes, narrative = get_notes(cx, test_id), get_narrative(cx, test_id)
            vscript = get_video_script(cx, test_id)
        return Response(render_report_html(rep, notes, narrative, vscript),
                        mimetype="text/html")

    @app.route("/test/<test_id>/report")
    def report_present(test_id):
        with sqlite3.connect(db_path) as cx:
            rep = _report_for(cx, test_id)
            narrative = get_narrative(cx, test_id)
        return Response(render_present(rep, narrative), mimetype="text/html")

    @app.route("/test/<test_id>/report.pdf")
    def report_present_pdf(test_id):
        import os
        with sqlite3.connect(db_path) as cx:
            rep = _report_for(cx, test_id)
            narrative = get_narrative(cx, test_id)
        html = render_present(rep, narrative)
        reports_dir = os.environ.get("BIOFIELD_REPORTS_DIR",
                                     os.path.join(os.path.expanduser("~"), "biofield-reports"))
        date = (rep.get("date") or "").replace("/", "-") or "undated"
        out = os.path.join(reports_dir, f"report_{test_id}_{date}.pdf")
        try:
            save_report_pdf(html, out)          # keep a local copy to print/ship
            data = open(out, "rb").read()
        except Exception as e:
            return Response(f"PDF generation failed: {e}", status=500)
        return Response(data, mimetype="application/pdf", headers={
            "Content-Disposition": f'inline; filename="biofield-{test_id}.pdf"'})

    # --- Authoring (Increment 4a) ---
    @app.route("/author/new", methods=["POST"])
    def author_new():
        with sqlite3.connect(db_path) as cx:
            tid = create_test(cx, "", "", "")
        return redirect(f"/author/{tid}")

    @app.route("/author/<test_id>")
    def author_edit(test_id):
        with sqlite3.connect(db_path) as cx:
            rep = authored_report(cx, test_id)
            dv = dimension_values(cx, DEPTH_KEY)
            transcript = get_notes(cx, test_id)
        return Response(render_author_html(rep, dv, transcript), mimetype="text/html")

    @app.route("/author/<test_id>/depth", methods=["POST"])
    def author_depth(test_id):
        d = request.get_json(silent=True) or {}
        side = d.get("side")
        if side not in ("stress", "remedy") or not d.get("rid"):
            return {"error": "bad params"}
        with sqlite3.connect(db_path) as cx:
            dim_tag(cx, "auth_" + side, d.get("rid"), DEPTH_KEY, d.get("rank"))
        return {"ok": True}

    @app.route("/author/<test_id>/header", methods=["POST"])
    def author_header(test_id):
        d = request.get_json(silent=True) or {}
        with sqlite3.connect(db_path) as cx:
            update_header(cx, test_id, name=d.get("name"), email=d.get("email"),
                          date=d.get("date"))
            ctx, _ = _e4l(cx, test_id)  # client now known -> pull recent E4L scan
            _seed_stresses(cx, test_id)  # synthesize + seed stress coverage if scan found
        return {"ok": True, "e4l": ctx, "html": render_e4l_panel(ctx)}

    @app.route("/author/<test_id>/e4l")
    def author_e4l(test_id):
        with sqlite3.connect(db_path) as cx:
            ctx, _ = _e4l(cx, test_id)
        return {"e4l": ctx, "html": render_e4l_panel(ctx)}

    @app.route("/api/e4l/clients")
    def api_e4l_clients():
        return {"clients": client_search(request.args.get("q", ""))}

    @app.route("/author/<test_id>/e4l/refresh", methods=["POST"])
    def author_e4l_refresh(test_id):
        """Pull this client's newest scan from the LIVE E4L portal, then re-read the
        panel. Synchronous (localhost + threaded server -> no gateway timeout); the
        browser shows a spinner while it runs (~15-20s for the Playwright login)."""
        body = request.get_json(silent=True) or {}
        client_id = body.get("client_id")
        with sqlite3.connect(db_path) as cx:
            rep = _report_for(cx, test_id)
        client = rep.get("client") or {}
        email = client.get("email") or ""
        if client_id is None and not email:
            return {"ok": False, "error": "no client selected yet",
                    "newer": False, "e4l": scan_lookup(""),
                    "html": render_e4l_panel(scan_lookup(""))}
        before = scan_lookup(email)
        res = _fetch_live(client_id=client_id, name=client.get("name"), runner=fetch_runner)
        after = scan_lookup(email)
        newer = bool(after.get("found") and (
            not before.get("found")
            or (after.get("scan_date") or "") > (before.get("scan_date") or "")))
        return {"ok": bool(res.get("ok")), "error": res.get("error"),
                "newer": newer, "e4l": after, "html": render_e4l_panel(after)}

    @app.route("/author/<test_id>/e4l/import-reveal", methods=["POST"])
    def author_import_reveal(test_id):
        """Import the client's recent (<7d) E4L reveal layers + remedies as
        needs-review causal-chain rows. Appends only after an explicit force when the
        session already has rows. Synthesis runs in-process (PHI stays local)."""
        import datetime as _dt
        from dashboard import biofield_reveal_import as _ri
        force = bool((request.get_json(silent=True) or {}).get("force"))
        with sqlite3.connect(db_path) as cx:
            rep = _report_for(cx, test_id)
            email = ((rep.get("client") or {}).get("email") or "").strip()
            if not email:
                return {"ok": False, "reason": "No client selected yet"}
            try:
                res = _ri.synthesize_reveal_layers(email, today=_dt.date.today().isoformat())
            except Exception as e:
                return {"ok": False, "reason": f"Reveal synthesis failed: {e}"}
            if not res.get("found"):
                return {"ok": False, "reason": "No E4L scan on file"}
            if not res.get("fresh"):
                return {"ok": False,
                        "reason": f"Latest scan is {res.get('days_ago')} days old "
                                  "— refresh to import"}
            existing = len(rep.get("layers") or [])
            if existing and not force:
                return {"ok": False, "needs_confirm": True, "existing": existing}
            imported = _ri.import_layers_to_test(cx, test_id, res.get("layers") or [])
            # Pass already-synthesized layers so the pipeline runs only once per import
            _seed_stresses(cx, test_id, force=True, layers=res.get("layers") or [])
        return {"ok": True, "imported": imported}

    @app.route("/author/<test_id>/stresses")
    def author_stresses(test_id):
        from dashboard import biofield_stress as _st
        with sqlite3.connect(db_path) as cx:
            rep = _report_for(cx, test_id)
            remedies = [l.get("remedy") for l in (rep.get("layers") or []) if l.get("remedy")]
            data = _st.list_stresses(cx, test_id, remedies)
        return {"data": data, "html": render_stress_panel(data)}

    @app.route("/author/<test_id>/stress/<int:sid>/balance", methods=["POST"])
    def author_stress_balance(test_id, sid):
        from dashboard import biofield_stress as _st
        value = bool((request.get_json(silent=True) or {}).get("value"))
        with sqlite3.connect(db_path) as cx:
            _st.set_manual_balanced(cx, test_id, sid, value)
        return {"ok": True}

    @app.route("/author/<test_id>/row", methods=["POST"])
    def author_row_add(test_id):
        d = request.get_json(silent=True) or {}
        with sqlite3.connect(db_path) as cx:
            rid = add_chain_row(cx, test_id, _layer_int(d.get("layer")), d.get("head", ""),
                                d.get("most_affected", ""), d.get("remedy", ""),
                                d.get("dosage", ""), d.get("frequency", ""), d.get("timing", ""))
        return {"ok": True, "rid": rid}

    @app.route("/author/<test_id>/row/<int:rid>", methods=["POST"])
    def author_row_save(test_id, rid):
        d = request.get_json(silent=True) or {}
        fields = {}
        for k in ("layer", "head", "most_affected", "remedy", "dosage", "frequency", "timing"):
            if k in d:
                fields[k] = _layer_int(d[k]) if k == "layer" else d[k]
        if "layer" in fields:
            new_layer = fields.pop("layer")
            from dashboard.biofield_authoring import reorder_chain
            with sqlite3.connect(db_path) as cx:
                if fields:
                    update_chain_row(cx, rid, **fields)
                reorder_chain(cx, test_id, rid, new_layer)
            return {"ok": True}
        with sqlite3.connect(db_path) as cx:
            update_chain_row(cx, rid, **fields)
        return {"ok": True}

    @app.route("/author/<test_id>/row/<int:rid>/delete", methods=["POST"])
    def author_row_delete(test_id, rid):
        with sqlite3.connect(db_path) as cx:
            delete_chain_row(cx, rid)
        return {"ok": True}

    @app.route("/api/catalog")
    def api_catalog():
        with sqlite3.connect(db_path) as cx:
            return {"catalog": remedy_catalog(cx, request.args.get("q", ""),
                                              _safe_int(request.args.get("limit"), 20))}

    @app.route("/api/dosing")
    def api_dosing():
        with sqlite3.connect(db_path) as cx:
            return remedy_dosing(cx, request.args.get("name", ""))

    @app.route("/api/vocab")
    def api_vocab():
        with sqlite3.connect(db_path) as cx:
            return {"vocab": stress_vocab(cx, request.args.get("q", ""),
                                          _safe_int(request.args.get("limit"), 20))}

    @app.route("/api/suggest")
    def api_suggest():
        with sqlite3.connect(db_path) as cx:
            return {"suggestions": stress_suggestions(cx, request.args.get("stress", ""))}

    @app.route("/api/deepgram-token")
    def api_deepgram_token():
        try:
            return {"key": deepgram_token()}
        except Exception as e:  # no key / network / Deepgram error
            return {"error": str(e)[:200]}

    @app.route("/author/<test_id>/session", methods=["POST"])
    def author_session(test_id):
        txt = ((request.get_json(silent=True) or {}).get("transcript") or "").strip()
        if not txt:
            return {"ok": True, "skipped": "empty"}
        with sqlite3.connect(db_path) as cx:
            save_notes(cx, test_id, txt)  # box holds the full transcript -> replace
        return {"ok": True}

    @app.route("/author/<test_id>/interpret", methods=["POST"])
    def author_interpret(test_id):
        with sqlite3.connect(db_path) as cx:
            transcript = get_notes(cx, test_id)
            if not transcript.strip():
                return {"added": 0, "error": "no transcript yet -- record a session first"}
            try:
                result = interpret_transcript(transcript, interpret_complete)
            except Exception as e:
                return {"error": str(e)[:200]}
            added = 0
            for l in result.get("layers", []):
                remedy = resolve_remedy_name(cx, l["remedy"])  # auto-correct + title-case ASR mangles
                head = resolve_stress_name(cx, l["head"])      # capitalize/match stress names too
                most_affected = resolve_stress_name(cx, l["most_affected"])
                dosage, frequency, timing = l.get("dosage", ""), l.get("frequency", ""), l.get("timing", "")
                if not (dosage or frequency or timing):  # no spoken dose -> catalog minimum
                    d = remedy_dosing(cx, remedy)
                    dosage, frequency, timing = d["dosage"], d["frequency"], d["timing"]
                add_chain_row(cx, test_id, l.get("layer"), head, most_affected,
                              remedy, dosage, frequency, timing, confirmed=0)  # voice -> unconfirmed
                added += 1
        return {"added": added, "header": result.get("header", "")}

    @app.route("/author/<test_id>/delete", methods=["POST"])
    def author_delete(test_id):
        with sqlite3.connect(db_path) as cx:
            delete_test(cx, test_id)
        return {"ok": True}

    @app.route("/author/<test_id>/confirm-all", methods=["POST"])
    def author_confirm_all(test_id):
        with sqlite3.connect(db_path) as cx:
            confirm_all(cx, test_id)
        return {"ok": True}

    @app.route("/author/<test_id>/row/<int:rid>/confirm", methods=["POST"])
    def author_row_confirm(test_id, rid):
        with sqlite3.connect(db_path) as cx:
            confirm_row(cx, rid)
        return {"ok": True}

    @app.route("/test/<test_id>/notes", methods=["POST"])
    def notes_save(test_id):
        with sqlite3.connect(db_path) as cx:
            save_notes(cx, test_id, (request.get_json(silent=True) or {}).get("notes", ""))
        return {"ok": True}

    @app.route("/test/<test_id>/narrative", methods=["POST"])
    def narrative_save(test_id):
        with sqlite3.connect(db_path) as cx:
            save_narrative(cx, test_id, (request.get_json(silent=True) or {}).get("narrative", ""))
        return {"ok": True}

    @app.route("/test/<test_id>/generate", methods=["POST"])
    def narrative_generate(test_id):
        notes = (request.get_json(silent=True) or {}).get("notes", "")
        with sqlite3.connect(db_path) as cx:
            save_notes(cx, test_id, notes)
            ctx, rep = _e4l(cx, test_id)  # authored or FMP report + recent E4L scan
            try:
                text = generate_narrative(rep, notes, complete, scan=ctx)
            except Exception as e:  # no API key / network / model error
                return {"error": str(e)[:200]}
            save_narrative(cx, test_id, text)
        return {"narrative": text}

    @app.route("/test/<test_id>/video-generate", methods=["POST"])
    def video_generate(test_id):
        notes = (request.get_json(silent=True) or {}).get("notes", "")
        with sqlite3.connect(db_path) as cx:
            ctx, rep = _e4l(cx, test_id)
            try:
                script = generate_video_script(rep, notes, complete, scan=ctx)
            except Exception as e:
                return {"error": str(e)[:200]}
            save_video_script(cx, test_id, script)
        return {"script": script}

    @app.route("/test/<test_id>/video-script", methods=["POST"])
    def video_script_save(test_id):
        with sqlite3.connect(db_path) as cx:
            save_video_script(cx, test_id, (request.get_json(silent=True) or {}).get("script", ""))
        return {"ok": True}

    @app.route("/test/<test_id>/audio", methods=["POST"])
    def make_audio(test_id):
        with sqlite3.connect(db_path) as cx:
            script = get_video_script(cx, test_id)
        if not (script or "").strip():
            return {"error": "no script yet -- generate or write one first"}
        try:
            audio = tts(script)
        except Exception as e:
            return {"error": str(e)[:200]}
        os.makedirs(AUDIO_DIR, exist_ok=True)
        fname = f"test_{test_id}.mp3"
        with open(os.path.join(AUDIO_DIR, fname), "wb") as f:
            f.write(audio)
        return {"url": f"/audio/{fname}", "bytes": len(audio)}

    @app.route("/audio/<path:fname>")
    def serve_audio(fname):
        return send_from_directory(AUDIO_DIR, fname, mimetype="audio/mpeg")

    return app


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="Local Biofield Analysis viewer")
    ap.add_argument("--load", action="store_true",
                    help="(re)load the snapshot from --export-dir before serving")
    ap.add_argument("--export-dir", default="/tmp/fmp-export/newapp")
    ap.add_argument("--port", type=int, default=8011)
    args = ap.parse_args()
    if args.load:
        from dashboard.biofield_fmp_snapshot import snapshot_csv_dir
        counts = snapshot_csv_dir(args.export_dir, DEFAULT_DB)
        print("loaded snapshot:", {k: counts[k] for k in sorted(counts)})
    print(f"Biofield Analysis viewer -> http://127.0.0.1:{args.port}  (local only; Ctrl-C to stop)")
    create_app().run(host="127.0.0.1", port=args.port, debug=False)
