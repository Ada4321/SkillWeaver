import os
import random
import time
from contextlib import contextmanager

import numpy as np

try:
    import torch
except Exception:  # pragma: no cover
    torch = None


_USE_TIMESTAMP_SEED = False
_SEED_COUNTER = 0


def configure_timestamp_seed_control(enabled: bool) -> None:
    global _USE_TIMESTAMP_SEED, _SEED_COUNTER
    _USE_TIMESTAMP_SEED = bool(enabled)
    _SEED_COUNTER = 0


def maybe_seed_with_timestamp() -> int | None:
    global _SEED_COUNTER
    if not _USE_TIMESTAMP_SEED:
        return None

    seed = int(time.time_ns()) ^ int(os.getpid()) ^ (_SEED_COUNTER << 16)
    _SEED_COUNTER += 1

    random.seed(seed)
    np.random.seed(seed % (2**32))
    if torch is not None:
        torch.manual_seed(seed)
    return seed


def clear_seed_to_none() -> None:
    random.seed(None)
    np.random.seed(None)
    if torch is not None:
        torch.seed()


@contextmanager
def timestamp_seed_scope():
    seeded = maybe_seed_with_timestamp() is not None
    try:
        yield
    finally:
        if seeded:
            clear_seed_to_none()
