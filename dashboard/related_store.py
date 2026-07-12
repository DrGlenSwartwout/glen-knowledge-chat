"""Disk persistence for related-products state. Manual picks are writable and live
on the /data disk (products.json is a read-only repo file). Harvested data is a
read-only versioned repo file."""
import json
import os

_REPO_DATA = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data")


def manual_path():
    d = os.environ.get("DATA_DIR") or _REPO_DATA
    return os.path.join(d, "related-manual.json")


def _read(path):
    try:
        with open(path) as f:
            return json.load(f)
    except (FileNotFoundError, ValueError):
        return {}


def load_manual(slug=None):
    data = _read(manual_path())
    if slug is None:
        return data
    return list(data.get(slug, []))


def save_manual(slug, related_slugs):
    path = manual_path()
    data = _read(path)
    data[slug] = list(related_slugs)
    tmp = path + ".tmp"
    with open(tmp, "w") as f:
        json.dump(data, f, indent=2, sort_keys=True)
    os.replace(tmp, path)


def load_harvested(slug=None):
    data = _read(os.path.join(_REPO_DATA, "related-harvested.json"))
    if slug is None:
        return data
    return list(data.get(slug, []))
