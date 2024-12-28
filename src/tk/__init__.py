import sys
import os
import logging
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