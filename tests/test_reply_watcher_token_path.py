"""Tests for reply_watcher's Gmail token-path resolution.

Regression test for the prod bug where the reply-watcher cron returned
HTTP 500 on Render because it only ever looked for the token at
~/.config/google/token.json (via GMAIL_TOKEN_PATH or the hardcoded
default), never at /data/google-token.json — the actual location on the
Render persistent disk that the send path (dashboard/inbox.py) already
checks. _resolve_token_path() now mirrors dashboard/inbox.py's
candidate-list approach: GMAIL_TOKEN_PATH env override, then
/data/google-token.json, then the local ~/.config/google/token.json
default — first existing file wins.

No real Google token or network access is used — a dummy file stands in
for the token, and only path resolution is exercised.
"""

from pathlib import Path


def test_resolves_data_disk_path_when_present(tmp_path, monkeypatch):
    """A token at a /data-style path should be found even without the
    env var set, mirroring Render's persistent-disk layout."""
    import reply_watcher

    data_token = tmp_path / "data" / "google-token.json"
    data_token.parent.mkdir(parents=True)
    data_token.write_text("{}")

    local_token = tmp_path / "home" / ".config" / "google" / "token.json"

    monkeypatch.delenv("GMAIL_TOKEN_PATH", raising=False)
    monkeypatch.setattr(
        reply_watcher, "_TOKEN_PATH_CANDIDATES",
        [str(data_token), str(local_token)],
    )

    resolved = reply_watcher._resolve_token_path()
    assert resolved == data_token


def test_honors_gmail_token_path_env_override(tmp_path, monkeypatch):
    """GMAIL_TOKEN_PATH must win over both candidate paths when set and
    the file it points to exists."""
    import reply_watcher

    override_token = tmp_path / "custom" / "token.json"
    override_token.parent.mkdir(parents=True)
    override_token.write_text("{}")

    data_token = tmp_path / "data" / "google-token.json"
    data_token.parent.mkdir(parents=True)
    data_token.write_text("{}")

    monkeypatch.setenv("GMAIL_TOKEN_PATH", str(override_token))
    monkeypatch.setattr(
        reply_watcher, "_TOKEN_PATH_CANDIDATES",
        [str(data_token), str(tmp_path / "home" / "token.json")],
    )

    resolved = reply_watcher._resolve_token_path()
    assert resolved == override_token


def test_falls_back_to_local_default_when_data_disk_absent(tmp_path, monkeypatch):
    """When no /data-style token exists (e.g. local dev), resolution
    should fall through to the local default path."""
    import reply_watcher

    data_token = tmp_path / "data" / "google-token.json"  # never created
    local_token = tmp_path / "home" / ".config" / "google" / "token.json"
    local_token.parent.mkdir(parents=True)
    local_token.write_text("{}")

    monkeypatch.delenv("GMAIL_TOKEN_PATH", raising=False)
    monkeypatch.setattr(
        reply_watcher, "_TOKEN_PATH_CANDIDATES",
        [str(data_token), str(local_token)],
    )

    resolved = reply_watcher._resolve_token_path()
    assert resolved == local_token


def test_raises_when_no_candidate_exists(tmp_path, monkeypatch):
    """No usable token anywhere should raise a clear RuntimeError, not a
    silent failure or an unrelated exception."""
    import reply_watcher

    monkeypatch.delenv("GMAIL_TOKEN_PATH", raising=False)
    monkeypatch.setattr(
        reply_watcher, "_TOKEN_PATH_CANDIDATES",
        [str(tmp_path / "data" / "google-token.json"),
         str(tmp_path / "home" / "token.json")],
    )

    try:
        reply_watcher._resolve_token_path()
        assert False, "expected RuntimeError"
    except RuntimeError as e:
        assert "No Gmail token" in str(e)
