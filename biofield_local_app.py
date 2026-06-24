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

from flask import Flask, Response, request

from dashboard.biofield_report import causal_chain_report, list_tests
from dashboard.biofield_report_html import render_list_html, render_report_html
from dashboard.biofield_narrative import (
    generate_narrative, get_narrative, get_notes, save_narrative, save_notes)


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

DEFAULT_DB = os.environ.get(
    "BIOFIELD_DB", os.path.join(os.path.dirname(os.path.abspath(__file__)), "chat_log.db"))


def create_app(db_path=DEFAULT_DB, complete=None):
    app = Flask(__name__)
    complete = complete or openai_complete

    @app.route("/")
    def index():
        q = request.args.get("q", "")
        with sqlite3.connect(db_path) as cx:
            return Response(render_list_html(list_tests(cx, q), q), mimetype="text/html")

    @app.route("/test/<test_id>")
    def report(test_id):
        with sqlite3.connect(db_path) as cx:
            rep = causal_chain_report(cx, test_id)
            notes, narrative = get_notes(cx, test_id), get_narrative(cx, test_id)
        return Response(render_report_html(rep, notes, narrative), mimetype="text/html")

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
