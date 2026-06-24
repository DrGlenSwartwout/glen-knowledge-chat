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
import os
import sqlite3

from flask import Flask, Response, redirect, request, send_from_directory

from dashboard.biofield_report import causal_chain_report, list_tests
from dashboard.biofield_report_html import (
    render_author_html, render_list_html, render_report_html)
from dashboard.biofield_narrative import (
    generate_narrative, generate_video_script, get_narrative, get_notes,
    get_video_script, save_narrative, save_notes, save_video_script)
from dashboard.biofield_authoring import (
    add_chain_row, authored_report, create_test, delete_chain_row, list_authored,
    remedy_catalog, remedy_dosing, stress_suggestions, stress_vocab,
    update_chain_row, update_header)
from dashboard.biofield_dimensions import (
    DEPTH_KEY, dimension_values, seed_dimensions, tag as dim_tag)

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


def create_app(db_path=DEFAULT_DB, complete=None, tts=None, deepgram_token=None):
    app = Flask(__name__)
    complete = complete or openai_complete
    tts = tts or elevenlabs_tts
    deepgram_token = deepgram_token or deepgram_browser_token
    with sqlite3.connect(db_path) as _cx:
        seed_dimensions(_cx)

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
        return Response(render_author_html(rep, dv), mimetype="text/html")

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
            existing = get_notes(cx, test_id)
            save_notes(cx, test_id, (existing + "\n\n" + txt).strip() if existing else txt)
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
            rep = causal_chain_report(cx, test_id)
            try:
                text = generate_narrative(rep, notes, complete)
            except Exception as e:  # no API key / network / model error
                return {"error": str(e)[:200]}
            save_narrative(cx, test_id, text)
        return {"narrative": text}

    @app.route("/test/<test_id>/video-generate", methods=["POST"])
    def video_generate(test_id):
        notes = (request.get_json(silent=True) or {}).get("notes", "")
        with sqlite3.connect(db_path) as cx:
            rep = causal_chain_report(cx, test_id)
            try:
                script = generate_video_script(rep, notes, complete)
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
