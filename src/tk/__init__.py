import sys
import os
import logging
import datetime as dt
L = logging.getLogger(__name__)

from pathlib import Path
try:
    from . import utils
    from .utils.utils import fetch, rootdir, datadir, timed
except:
    L.exception("Loading utils failed")

in_colab = 'google.colab' in sys.modules
try: in_ipy = get_ipython() is not None
except: in_ipy = False

__all__ = [
    'in_colab', 'datadir', 'rootdir', 'utils',
    'fetch', 'datadir', 'timed',
]


default_dfmt = "%Y%m"
default_dtfmt = "%Y%m%d_%H%M"

def now(f: str | None = default_dfmt) -> str:
    now = dt.datetime.now()
    if f is None:  # "I want the raw date"
        return now
    return now.strftime(f)

def xpdir(f: str = default_dfmt) -> Path:
    """Get path to experiment dir.

    This will not generate the dir!
    If suffix contains percent, assumes current date wanted.
    """
    suffix = f
    if '%' in f:  # assume dateformat wanted
        suffix = now(suffix)
    L.info(f"Using {datadir / suffix} as experiment directory")
    return datadir / suffix
