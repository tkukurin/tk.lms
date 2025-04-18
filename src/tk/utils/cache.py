"""Cache utils.
"""
import diskcache
import functools as ft
import collections as cols

from typing import *
from tk.utils.utils import datadir
from tk.utils.log import L

cache = diskcache.Cache(datadir / "cache")


def logged(f: Callable) -> Callable:
    """log `f` calls, compose with memo to check if accessed"""
    def _inner(*args, **kw):
        L.info(f"CALL|{f.__name__}: {args} {kw}")
        return f(*args, **kw)
    return ft.wraps(f)(_inner)


def memo(f: Callable, seed: int | None = None, **kw) -> Callable:
    """easy-cache results `memo(f)(...)`.
    `seed` if you use global seeds (eg random.seed(42)).
    This allows you to keep some notion of seed-aware caching.
    ofc be careful.
    """
    if seed is not None: kw['tag'] = seed
    return cache.memoize(**kw)(f)


def unmemo(f: Callable) -> Callable:
    """unwrap => get fn without memoization `unmemo(f)(...)`."""
    return getattr(f, "__wrapped__", f)


def memos_all(cache: diskcache.Cache = cache) -> tuple[list[Any], list[Any]]:
    """Get all cached keys and values."""
    keys = list(cache.iterkeys())
    values = [cache.get(k) for k in keys]
    return keys, values
