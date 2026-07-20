"""
EnvPool: manages a fixed set of IsaacLab env slots (indices 0..size-1).

Usage:
    pool = EnvPool(task_env, size=4)

    with pool.slot() as env_id:
        simulator.restore(task_env, snap, env_id=env_id)
        ...execute on env_id...

Thread-safe: acquire/release use a threading.Semaphore so multiple
threads can hold different slots simultaneously.
"""

import threading
from contextlib import contextmanager


class EnvPool:
    def __init__(self, env, size: int):
        self._env = env
        self._size = size
        self._lock = threading.Lock()
        self._free = list(range(size))   # available slot ids
        self._semaphore = threading.Semaphore(size)

    @property
    def env(self):
        return self._env

    @property
    def size(self) -> int:
        return self._size

    def acquire(self) -> int:
        """Block until a slot is free, then return its env_id."""
        self._semaphore.acquire()
        with self._lock:
            return self._free.pop()

    def release(self, env_id: int) -> None:
        """Return a slot to the pool."""
        with self._lock:
            self._free.append(env_id)
        self._semaphore.release()

    @contextmanager
    def slot(self):
        """Context manager: acquire a slot, yield its env_id, then release."""
        env_id = self.acquire()
        try:
            yield env_id
        finally:
            self.release(env_id)
