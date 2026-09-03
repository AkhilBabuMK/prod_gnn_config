# -*- coding: utf-8 -*-
"""
One writer at a time.

Two loops running at once both write `model_state` and `forecast`. Their ticks
interleave, so the memory checkpoint ends up a blend of two different runs and
the forecast table holds rows from both. Nothing errors; the output just
quietly stops meaning anything. That has happened twice on this project — once
when a stop signal did not reach a child process and the old loop kept writing
under the new one.

A Postgres advisory lock is the right instrument rather than a lock file:

  * it is held by the CONNECTION, so it is released the moment the process
    dies, however it dies — no stale lock to clear by hand after a crash;
  * it is visible to every client of the database, including one started on
    another machine, which a file on this disk is not;
  * taking it costs one round trip and no table.

The lock is scoped to the corridor date being written, so a replay of the 12th
and a live loop on the 27th do not block each other — they touch different rows.
"""
from __future__ import annotations

import os
import zlib

# Distinct from any other advisory lock this database might use.
_NAMESPACE = 0x49525031                            # "IRP1" as bytes


def _key(scope: str) -> int:
    """A stable 32-bit key from the scope string. Advisory locks take two int4s;
    the namespace keeps us out of anyone else's key space."""
    return zlib.crc32(scope.encode("utf-8")) & 0x7FFFFFFF


class RunLock:
    """Hold the writer lock for one scope, or report who has it.

        with RunLock(conn, "2025-09-27") as lock:
            if not lock.held:
                ...refuse to start...

    Uses its OWN connection. Sharing the loop's connection would tie the lock's
    lifetime to that connection's transactions, and a rollback would drop it
    without anyone noticing.
    """

    def __init__(self, conn, scope: str):
        self.conn = conn
        self.scope = scope
        self.held = False
        self.holder = None

    def acquire(self) -> bool:
        with self.conn.cursor() as cur:
            cur.execute("SELECT pg_try_advisory_lock(%s, %s)",
                        (_NAMESPACE, _key(self.scope)))
            self.held = bool(cur.fetchone()[0])
            if not self.held:
                self.holder = self._who(cur)
            else:
                # Leave a trace of who holds it, so the next process can say
                # something more useful than "someone else".
                cur.execute("""
                    SELECT set_config('application_name',
                                      %s, false)""",
                            (f"irpilot {self.scope} pid {os.getpid()}",))
        self.conn.commit()
        return self.held

    def _who(self, cur):
        """Best effort: the backend holding this exact advisory lock."""
        cur.execute("""
            SELECT a.pid, a.application_name, a.backend_start
              FROM pg_locks l
              JOIN pg_stat_activity a ON a.pid = l.pid
             WHERE l.locktype = 'advisory'
               AND l.classid  = %s AND l.objid = %s
               AND l.granted
             LIMIT 1""", (_NAMESPACE, _key(self.scope)))
        r = cur.fetchone()
        if not r:
            return None
        pid, name, since = r
        return f"pid {pid} ({name or 'unnamed'}) since {since:%H:%M:%S}"

    def release(self) -> None:
        if not self.held:
            return
        with self.conn.cursor() as cur:
            cur.execute("SELECT pg_advisory_unlock(%s, %s)",
                        (_NAMESPACE, _key(self.scope)))
        self.conn.commit()
        self.held = False

    def __enter__(self):
        self.acquire()
        return self

    def __exit__(self, *exc):
        self.release()
        return False

    def explain(self) -> str:
        """The message to print when the lock could not be taken."""
        who = f"  It is held by {self.holder}." if self.holder else ""
        return (f"Another loop is already writing corridor date {self.scope}."
                f"{who}\n"
                f"  Two writers interleave their ticks and corrupt both runs.\n"
                f"  Stop the other one, or pass --date for a different day.")
