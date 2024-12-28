"""Cache utils.
"""
import diskcache

from typing import Callable
from tk.utils.utils import datadir

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
