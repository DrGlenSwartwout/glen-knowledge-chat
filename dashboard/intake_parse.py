"""Parse a pasted Practice Better intake export into the declarative INTAKE_FORM
answer shape. Pure + LLM-agnostic: `parse` takes a `complete(system, user)->str`
callable so it is unit-testable without the network. Glen reviews/edits the
result before it is imported, so coercion is lenient-but-safe: unknown fields
and out-of-range scale values are dropped rather than guessed."""
import json

SYSTEM = ("You extract a clinical intake form into JSON. Return ONLY a JSON object "
          "keyed by the given field ids. For a scale field return the selected integer. "
          "For a table field return an array of row objects using the listed column ids. "
          "Use strings for text fields. Omit any field not present in the text. "
          "No commentary, JSON only.")


def _field_index(form):
    idx = {}
    for sec in form["sections"]:
        for f in sec["fields"]:
            idx[f["id"]] = f
    return idx


def build_parse_prompt(form, pasted_text):
    lines = ["Fields to extract (id: type; options for scales):"]
    for sec in form["sections"]:
        lines.append(f"[{sec['id']}] {sec['title']}")
        for f in sec["fields"]:
            if f["type"] == "scale":
                opts = "; ".join(f"{o['value']}={o['label']}" for o in f["options"])
                lines.append(f"  {f['id']} (scale): {opts}")
            elif f["type"] == "table":
                cols = ", ".join(c["id"] for c in f["columns"])
                lines.append(f"  {f['id']} (table rows of: {cols})")
            elif f["type"] == "consent":
                continue
            else:
                lines.append(f"  {f['id']} ({f['type']})")
    schema = "\n".join(lines)
    return f"{schema}\n\nIntake export text:\n\"\"\"\n{pasted_text}\n\"\"\"\n\nReturn the JSON now."


def coerce_parsed(form, raw):
    if not isinstance(raw, dict):
        return {}
    idx = _field_index(form)
    out = {}
    for fid, val in raw.items():
        f = idx.get(fid)
        if not f:
            continue
        t = f["type"]
        if t == "scale":
            try:
                iv = int(val)
            except (TypeError, ValueError):
                continue
            if iv in {o["value"] for o in f["options"]}:
                out[fid] = iv
        elif t == "table":
            if isinstance(val, list):
                cols = {c["id"] for c in f["columns"]}
                rows = [{k: r[k] for k in r if k in cols}
                        for r in val if isinstance(r, dict)]
                rows = [r for r in rows if r]
                if rows:
                    out[fid] = rows
        elif t == "consent":
            continue
        else:
            if val is not None and str(val).strip():
                out[fid] = val if isinstance(val, (str, int, float)) else str(val)
    return out


def parse(form, pasted_text, complete):
    try:
        raw = json.loads(complete(SYSTEM, build_parse_prompt(form, pasted_text)))
    except Exception:
        return {}
    return coerce_parsed(form, raw)
