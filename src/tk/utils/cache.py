"""Cache utils.
"""
import diskcache
import functools as ft

from typing import Callable
from tk.utils.utils import datadir
from tk.utils.log import L

cache = diskcache.Cache(datadir / "cache")


def log_call(f: Callable):
    """Wrap a function to log its calls.

    Can compose with memo to check if memo is accessed.
    """
    @ft.wraps(f)
    def _inner(*args, **kw):
        L.info(f"CALL|{f.__name__}: {args} {kw}")
        return f(*args, **kw)
    return _inner


def memo(f: Callable, seed: int | None = None, **kw):
    """Use instead of manual caching.

    Use `seed` if you set global seeds.

    E.g.
        >>> fetch(url, cache_file)
        >>> data = open(cache_file)
        >>> # instead ...
        >>> data = utils.memo(fetch)(url)
    """
    if seed: kw['tag'] = seed
    return cache.memoize(**kw)(f)
