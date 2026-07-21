"""OpenAI embeddings client with multi-key failover.

OpenAI is the only embeddings provider whose vectors match the existing Pinecone
index, so resilience to an OPENAI_API_KEY quota/auth outage means a SECOND OpenAI
key for the SAME embedding model, not a different provider (a different provider's
vectors live in a different space and can't be searched against this index).

Configure a backup via OPENAI_API_KEY_FALLBACK (a second key, ideally on a separate
OpenAI org/project with its own quota and billing). With no fallback configured the
wrapper behaves exactly like a single OpenAI client. Failover triggers on ANY error
from a key (429 quota/rate-limit, 401 auth, transient 5xx) and tries the next key.
"""
import os
from openai import OpenAI


class _EmbeddingFailover:
    """`.create(**kw)` tries each client's embeddings endpoint in order, failing
    over on any exception and raising the last error if every key fails."""
    def __init__(self, clients):
        self._clients = list(clients)

    def create(self, **kw):
        last = None
        for i, c in enumerate(self._clients):
            try:
                return c.embeddings.create(**kw)
            except Exception as e:  # 429/401/5xx — try the next key
                last = e
                if i + 1 < len(self._clients):
                    print(f"[openai] embeddings key #{i+1} failed "
                          f"({type(e).__name__}: {str(e)[:120]}); "
                          f"failing over to key #{i+2}", flush=True)
        if last is not None:
            raise last
        raise RuntimeError("no OpenAI embedding client configured")


class OpenAIWithEmbedFailover:
    """Drop-in replacement for an OpenAI() client. `.embeddings.create(...)` is
    multi-key with failover; every other attribute (.chat, .files, ...) delegates
    to the primary client, so existing call sites keep working unchanged."""
    def __init__(self, clients):
        if not clients:
            raise ValueError("OpenAIWithEmbedFailover requires >=1 client")
        self._clients = list(clients)
        self.embeddings = _EmbeddingFailover(self._clients)

    def __getattr__(self, name):
        # only reached for attributes not set on self (delegate to primary client)
        clients = self.__dict__.get("_clients")
        if not clients:
            raise AttributeError(name)
        return getattr(clients[0], name)


def build_openai_client(primary_key=None, fallback_keys=None):
    """Build the failover client from env (OPENAI_API_KEY + OPENAI_API_KEY_FALLBACK)
    unless explicit keys are passed. Always returns a usable client (an empty key,
    matching prior behavior, if nothing is configured)."""
    if primary_key is None:
        primary_key = os.environ.get("OPENAI_API_KEY", "")
    if fallback_keys is None:
        fb = os.environ.get("OPENAI_API_KEY_FALLBACK", "")
        fallback_keys = [fb] if fb else []
    keys = [k for k in ([primary_key] + list(fallback_keys)) if k]
    if not keys:
        # No key configured. The bare single-client path raised on a missing key;
        # but some openai SDK versions silently accept OpenAI(api_key="") instead,
        # which would MASK the misconfiguration (and makes the "surfaces loudly"
        # contract depend on SDK version + test order). Raise explicitly so it
        # surfaces loudly at startup regardless of SDK version — as documented.
        from openai import OpenAIError
        raise OpenAIError(
            "No OpenAI API key configured (set OPENAI_API_KEY); refusing to "
            "build a client with an empty key.")
    return OpenAIWithEmbedFailover([OpenAI(api_key=k) for k in keys])
