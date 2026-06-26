"""Root conftest — runs before pytest collects any test module.

gevent's monkey-patch must happen before ssl/socket are imported; patching them
late recurses on Python 3.12+ (the same reason app.py patches at its very top).
Under pytest, importing a test module pulls in `app`, whose monkey.patch_all()
then runs *after* pytest has already imported ssl -> RecursionError at collection.

Patching here, in the rootdir conftest (the earliest hook pytest offers), runs
before any test module is imported, so by the time app.py calls patch_all() the
modules are already patched and it's a no-op. Guarded so it's a no-op if some
earlier import already patched (or if gevent isn't installed).
"""
try:
    from gevent import monkey
    if not monkey.is_module_patched("ssl"):
        monkey.patch_all()
except ImportError:
    pass
