"""Cross-source de-duplication for farm rows (Phase 3, multi-source).

When a second directory is added (e.g. Regeneration International once its
endpoint returns), the same farm can appear in more than one source. Before
upserting the combined list, collapse duplicates so one farm = one row.

Match key, in priority order:
  1. website domain (normalized, sans scheme/www) — the strongest signal that
     two listings are the same business.
  2. name slug + coarse coordinates (~0.01° ≈ 1 km grid) — for farms with no
     website, same name at essentially the same spot.

Single-source crawls have unique source_urls and (almost) no dupes, so this is
a no-op there; it earns its keep only when sources are combined. Keeps the FIRST
occurrence (callers should order sources by trust) and logs the drop count.
"""
import re
from typing import Optional

from scrapers.farm_finder.models import NormalizedFarmRow


def _domain(url: Optional[str]) -> Optional[str]:
    if not url:
        return None
    m = re.sub(r"^https?://", "", url.strip().lower())
    host = m.split("/")[0]
    host = host[4:] if host.startswith("www.") else host
    return host or None


def _name_slug(name: str) -> str:
    return re.sub(r"[^a-z0-9]+", "", (name or "").lower())


def dedupe_key(row: NormalizedFarmRow):
    """Return a hashable key identifying the farm, or a unique fallback so rows
    with no usable signal are never wrongly merged."""
    dom = _domain(row.website)
    if dom:
        return ("domain", dom)
    if row.lat is not None and row.lng is not None:
        return ("geo", _name_slug(row.name), round(row.lat, 2), round(row.lng, 2))
    # No website and no coords: fall back to source_url so it stays distinct.
    return ("url", row.source_url or id(row))


def dedupe_farms(rows: list[NormalizedFarmRow], log=print) -> list[NormalizedFarmRow]:
    """Collapse duplicate farms across sources, keeping the first occurrence."""
    seen: set = set()
    out: list[NormalizedFarmRow] = []
    dropped = 0
    for row in rows:
        key = dedupe_key(row)
        if key in seen:
            dropped += 1
            continue
        seen.add(key)
        out.append(row)
    if dropped:
        log(f"farm dedupe: dropped {dropped} cross-source duplicate(s), "
            f"{len(out)} unique")
    return out
