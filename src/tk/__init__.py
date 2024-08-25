import sys
import os

from pathlib import Path
from . import utils
from .utils.utils import memo, fetch, rootdir, datadir, timed

in_colab = 'google.colab' in sys.modules
try: in_ipy = get_ipython() is not None
except: in_ipy = False

__all__ = [
    'in_colab', 'datadir', 'rootdir', 'utils',
    'memo', 'fetch', 'datadir', 'timed',
]