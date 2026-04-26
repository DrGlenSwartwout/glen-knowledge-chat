"""Pinecone vector stats per namespace for remedy-match-llc."""

import os
from pinecone import Pinecone
from datetime import datetime, timezone
from .cache import cached, last_success

INDEX_NAME = "remedy-match-llc"
_pc = Pinecone(api_key=os.environ.get("PINECONE_API_KEY", ""))


@cached("pinecone.stats")
def index_stats():
    idx = _pc.Index(INDEX_NAME)
    stats = idx.describe_index_stats()
    namespaces = {}
    for ns_name, ns_data in (stats.get("namespaces") or {}).items():
        namespaces[ns_name or "(default)"] = ns_data.get("vector_count", 0)
    return {
        "index": INDEX_NAME,
        "total_vectors": stats.get("total_vector_count", 0),
        "dimension": stats.get("dimension", 0),
        "namespaces": namespaces,
        "last_success": last_success("pinecone.stats"),
        "as_of": datetime.now(timezone.utc).isoformat(),
    }
