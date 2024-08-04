import sys

from pathlib import Path
from . import utils
from .utils.utils import memo, fetch, datadir, timed

in_colab = 'google.colab' in sys.modules

__all__ = [
    'in_colab', 'datadir', 'rootdir', 'utils',
    'memo', 'fetch', 'datadir', 'timed',
]