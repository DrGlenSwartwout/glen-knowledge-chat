"""Helpers for test fixtures that `importlib.reload(app)`.

Reloading app.py re-runs its module-level bootstrap, which — when DATA_DIR is
set — spins up a never-shut-down BackgroundScheduler (`_start_scheduler`) plus a
daemon prewarm thread (`_prewarm_caches`). Those workers outlive the test that
reloaded and keep opening `db.connect(LOG_DB)` against whatever LOG_DB later,
unrelated tests point at. Each extra reloading fixture adds another un-stopped
worker to that shared, session-long pile-up.

`_quiet_reload_background_workers(monkeypatch)` neutralizes both so a reload
leaves nothing running behind it. Call it BEFORE `importlib.reload(app)`; the
patches target library/stdlib objects that the reload does not replace, so they
stay in force through the reload and are auto-undone at fixture teardown.
"""
import threading


def _quiet_reload_background_workers(monkeypatch):
    import apscheduler.schedulers.background as _bg

    monkeypatch.setattr(_bg.BackgroundScheduler, "start",
                        lambda self, *a, **k: None, raising=False)
    monkeypatch.setattr(_bg.BackgroundScheduler, "shutdown",
                        lambda self, *a, **k: None, raising=False)

    _orig_start = threading.Thread.start

    def _skip_prewarm(self, *a, **k):
        # app._prewarm_caches() starts exactly one daemon named "prewarm-money".
        # Skip only that; every other thread starts normally.
        if getattr(self, "name", "") == "prewarm-money":
            return
        return _orig_start(self, *a, **k)

    monkeypatch.setattr(threading.Thread, "start", _skip_prewarm, raising=False)
