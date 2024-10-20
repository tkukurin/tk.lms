from typing import Callable
import re
import itertools as it

import requests
import diskcache
from loguru import logger
from pathlib import Path

from typing import Any, Callable
from types import SimpleNamespace as nspc
from time import time
from contextlib import contextmanager
from dataclasses import dataclass


rootdir = Path(__file__).parent.parent.parent.parent
datadir = rootdir / 'data'
cache = diskcache.Cache(datadir / "cache")


def memo(f: Callable, **kw):
    """Use instead of manual caching.
    
    E.g. 
        >>> fetch(url, cache_file)
        >>> data = open(cache_file)
        >>> # instead ...
        >>> data = utils.memo(fetch)(url)
    
    """
    return cache.memoize(**kw)(f)


def fetch(url: str) -> bytes:
    logger.debug(f"Downloading: {url}")
    if not (got := requests.get(url)).ok:
        logger.warning(f"some error downloading: {got}")
        return None
    else:
        data = got.content
    return data


@dataclass(frozen=False)
class TimeResult:
    t0: float
    t1: float
    name: str


@contextmanager
def timed(name: ty.Any = "_", show: ty.Callable=print):
    """Use to ad-hoc time.
    """
    val = TimeResult(name=name, t0=time(), t1=None)
    val.t0 = time()
    yield val
    val.t1 = time()
    show(f"t ({name}): {val.t1-val.t0:.2f}s")
