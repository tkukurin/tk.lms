import re
import itertools as it

import requests
import diskcache
from pathlib import Path

from typing import Any, Callable
from types import SimpleNamespace as nspc
from time import time
from contextlib import contextmanager
from dataclasses import dataclass

from loguru import logger as L
from torch.nn.functional import max_pool1d_with_indices
# TODO rich
# from rich import logging

rootdir = Path(__file__).parent.parent.parent.parent
# will be sth like '/.../.venv/lib/python3.11/...'
if '/lib/python' in str(rootdir.absolute()):
    L.warning(
        f"I think we are running in venv: {rootdir}."
        "Suggest to install via `pip install -e .`"
    )
datadir = rootdir / 'data'
cache = diskcache.Cache(datadir / "cache")


def shrt(text: str, to: int = 10) -> str:
    """Shorten text to `to` characters."""
    from rich.text import Text
    t = Text(text)
    t.truncate(to, overflow="ellipsis")
    return str(t)


def memo(f: Callable, **kw):
    """Use instead of manual caching.

    E.g.
        >>> fetch(url, cache_file)
        >>> data = open(cache_file)
        >>> # instead ...
        >>> data = utils.memo(fetch)(url)

    """
    return cache.memoize(**kw)(f)


def fetch(url: str) -> bytes | None:
    L.debug(f"Downloading: {url}")
    if not (got := requests.get(url)).ok:
        L.warning(f"some error downloading: {got}")
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
def timed(name: Any = "_", show: Callable = L.debug):
    """Use to ad-hoc time.
    """
    val = TimeResult(name=name, t0=time(), t1=None)
    val.t0 = time()
    yield val
    val.t1 = time()
    show(f"t ({name}): {val.t1-val.t0:.2f}s")
