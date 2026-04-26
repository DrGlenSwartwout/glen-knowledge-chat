"""Command Center Dashboard — modular integrations for the /dashboard route."""

import os
from functools import wraps
from flask import request, jsonify

CONSOLE_SECRET = os.environ.get("CONSOLE_SECRET", "")


def require_console_key(fn):
    """Decorator: enforce X-Console-Key header (or ?key= query) matches CONSOLE_SECRET."""
    @wraps(fn)
    def wrapper(*args, **kwargs):
        if not CONSOLE_SECRET:
            return fn(*args, **kwargs)  # auth disabled if no secret set
        key = request.headers.get("X-Console-Key", "") or request.args.get("key", "")
        if key != CONSOLE_SECRET:
            return jsonify({"ok": False, "error": "unauthorized"}), 401
        return fn(*args, **kwargs)
    return wrapper


def ok(data, **extra):
    """Standard success envelope."""
    return jsonify({"ok": True, "data": data, **extra})


def fail(error, status=500, **extra):
    """Standard failure envelope."""
    return jsonify({"ok": False, "error": str(error), **extra}), status
