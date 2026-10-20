"""Serialize local SQLite connection lifecycles across worker threads.

Some SQLite builds can deadlock while one thread opens a database and another
closes a WAL connection. Our single-user runtime keeps transactions short and
never awaits inside them; a reentrant process lock also permits nested reads.
"""

import sqlite3
import threading
from contextlib import contextmanager

_LOCK = threading.RLock()


@contextmanager
def connection(path, timeout=5, *, uri=False):
    with _LOCK:
        db = sqlite3.connect(path, timeout=timeout, uri=uri)
        db.execute(f"PRAGMA busy_timeout={int(timeout * 1000)}")
        db.row_factory = sqlite3.Row
        try:
            with db:
                yield db
        finally:
            db.close()
