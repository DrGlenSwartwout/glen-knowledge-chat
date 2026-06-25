# shell_nav.py — pure helpers for the injected navigation shell (1a).
# No Flask import: everything here is unit-testable in isolation.

_EXCLUDE_PREFIXES = ("/console/", "/admin/", "/api/", "/static/")
_EXCLUDE_EXACT = ("/begin/state",)
_MEMBER_PREFIXES = ("/client-portal", "/coaching", "/affiliate-hub",
                    "/cert-portal", "/practitioner", "/dashboard", "/workspace")

_MARKER = 'id="journey-shell-assets"'


def should_inject(path: str, content_type: str, status: int) -> bool:
    """True only for public HTML 200 pages the shell should attach to."""
    if status != 200:
        return False
    if "text/html" not in (content_type or "").lower():
        return False
    p = (path or "").rstrip("/") or "/"
    if p in _EXCLUDE_EXACT:
        return False
    if any(p == pre.rstrip("/") or p.startswith(pre) for pre in _EXCLUDE_PREFIXES):
        return False
    return True


def resolve_mode(path: str, authenticated: bool) -> str:
    """Member when logged in OR on a member surface; funnel otherwise."""
    if authenticated:
        return "member"
    p = (path or "")
    if any(p.startswith(pre) for pre in _MEMBER_PREFIXES):
        return "member"
    return "funnel"


def inject_shell_html(html: str, mode: str, rewards1b: bool = False, rewards_gift: bool = False) -> str:
    """Insert the shell <link>+<script> tags before </head>. Idempotent; no-op when no </head>."""
    if _MARKER in (html or ""):
        return html
    if "</head>" not in html:
        return html
    mode = "member" if mode == "member" else "funnel"
    r1 = "true" if rewards1b else "false"
    rg = "true" if rewards_gift else "false"
    tags = (
        f'<link {_MARKER} rel="stylesheet" href="/static/shell.css">'
        f'<script>window.__SHELL__={{"mode":"{mode}","rewards1b":{r1},"rewardsGift":{rg}}};</script>'
        f'<script defer src="/static/shell.js"></script>'
    )
    return html.replace("</head>", tags + "\n</head>", 1)


def validate_shell_map(cfg: dict, land_keys) -> list:
    """Return a list of human-readable errors. Empty list == valid.
    Every land must map to a real engine land key; every land's category
    must have a style in `categories`."""
    errors = []
    lands = (cfg or {}).get("lands") or {}
    cats = (cfg or {}).get("categories") or {}
    valid = set(land_keys or ())
    for key, land in lands.items():
        if key not in valid:
            errors.append(f"unknown land '{key}' (not a JOURNEY_STEPS key)")
        cat = (land or {}).get("category")
        if cat not in cats:
            errors.append(f"land '{key}' references missing category style '{cat}'")
    return errors
