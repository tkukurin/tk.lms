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


@contextmanager
def timed(name: Any = "_", show: Callable = logger.debug):
    """Use to ad-hoc time.
    """
    val = nspc(name=name, t1=None, t2=None)
    val.t1 = time()
    yield val
    val.t2 = time()
    show(f"t ({name}): {val.t2-val.t1}s")
