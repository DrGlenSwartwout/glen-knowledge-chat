"""Manual headless-verification harness for the portal stress-pattern detail feature.

Serves the REAL static/client-portal.html at /portal/<anything> and returns a
controlled /api/portal payload: two findings WITH a description + one WITHOUT,
status=confirmed, so the pattern chips render as a mix of clickable + plain.
Pure stdlib: no Flask, no Doppler, no Pinecone, no full-app boot.

Run:   python3 tests/manual/portal_pattern_harness.py [PORT]   (default 8799)
Open:  http://localhost:PORT/portal/testtoken
"""
import json
import sys
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path
from urllib.parse import urlparse

STATIC = Path(__file__).resolve().parents[2] / "static" / "client-portal.html"

FINDINGS = [
    {"code": "ED1", "name": "Source Driver", "rank": 1,
     "description": "The Source Driver bioenergetically supports the strength of the body's fields."},
    {"code": "ET1", "name": "Heart Driver", "rank": 2,
     "description": "The Heart Driver bioenergetically supports the heartbeat and the midbrain."},
    {"code": "ER9", "name": "Environmental Load", "rank": 3, "description": ""},
]
D = {"name": "Test Client", "biofield_status": "confirmed", "blurred": False,
     "actionable": False, "scan_date": "2026-07-01", "scan_dates": ["2026-07-01"],
     "greeting": "Aloha Test,", "layers": [{"n": 1, "title": "Surface", "meaning": "x"}],
     "findings": FINDINGS, "reorder_items": [], "messages": [],
     "membership_category": "none", "notify_on": True, "tos_agreed": True,
     "element_state": None, "element_backdrop_enabled": False}
V = {"biofield": {"visible": True, "status": "confirmed", "blurred": False,
     "scan_date": "2026-07-01", "scan_dates": ["2026-07-01"],
     "layers": [{"n": 1, "title": "Surface", "meaning": "x"}]}}


class H(BaseHTTPRequestHandler):
    def _send(self, code, body, ctype):
        self.send_response(code)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self):
        p = urlparse(self.path).path
        if p.startswith("/portal/"):
            self._send(200, STATIC.read_bytes(), "text/html; charset=utf-8")
        elif p.endswith("/view"):
            self._send(200, json.dumps(V).encode(), "application/json")
        elif p.startswith("/api/portal/"):
            self._send(200, json.dumps(D).encode(), "application/json")
        else:
            self._send(200, b"null", "application/json")

    def log_message(self, *a):
        pass


if __name__ == "__main__":
    port = int(sys.argv[1]) if len(sys.argv) > 1 else 8799
    print(f"serving http://localhost:{port}/portal/testtoken")
    HTTPServer(("127.0.0.1", port), H).serve_forever()
