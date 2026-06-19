import subprocess


def compute_trim_window(words, duration, *, lead=0.3, trail=0.7, min_removed=1.0):
    """Window (start, end) trimming dead air to first/last word +/- padding, or None to skip.

    Returns None when:
    - no words provided, or duration <= 0
    - the resulting trim is negligible (< min_removed after accounting for clamping)
    """
    try:
        dur = float(duration or 0)
        if not words or dur <= 0:
            return None
        first = float(words[0].get("start", 0))
        last = float(words[-1].get("end", 0))
        raw_start = first - lead
        raw_end = last + trail
        start = max(0.0, raw_start)
        end = min(dur, raw_end)
        if end <= start:
            return None
        # How much we actually cut from the clip (post-clamp)
        removed = start + (dur - end)
        # How much padding overflowed the clip edges — counts double toward effective removal
        overflow = max(0.0, -raw_start) + max(0.0, raw_end - dur)
        effective_removed = removed + 2.0 * overflow
        if effective_removed < min_removed:
            return None
        return (start, end)
    except (TypeError, ValueError, KeyError, IndexError):
        return None


def _ffmpeg_exe():
    import imageio_ffmpeg
    return imageio_ffmpeg.get_ffmpeg_exe()


def trim_video(src_path, dst_path, start, end, *, runner=None):
    """Re-encode a frame-accurate trimmed copy. Returns True on success, never raises."""
    run = runner or subprocess.run
    try:
        cmd = [_ffmpeg_exe(), "-y", "-ss", str(start), "-to", str(end), "-i", str(src_path),
               "-c:v", "libx264", "-c:a", "aac", "-movflags", "+faststart", str(dst_path)]
        import os
        res = run(cmd, capture_output=True, timeout=300)
        return getattr(res, "returncode", 1) == 0 and os.path.exists(str(dst_path))
    except Exception as e:  # noqa: BLE001
        print(f"[video-trim] failed: {e}", flush=True)
        return False
