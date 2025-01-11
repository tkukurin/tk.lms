import os
import signal
import re
import itertools as it
import sys
import pdb
import traceback


import requests
from pathlib import Path

from typing import Any, Callable
from types import SimpleNamespace as nspc
from time import time
from contextlib import contextmanager
from dataclasses import dataclass
from tk.utils.log import L
import warnings


rootdir = Path(__file__).parent.parent.parent.parent
# will be sth like '/.../.venv/lib/python3.11/...'
if '/lib/python' in str(rootdir.absolute()):
    warnings.warn(
        f"I think we are running in venv: {rootdir}. "
        "Suggest to install via `pip install -e .` "
    )
datadir = rootdir / 'data'


def shrt(text: str, to: int = 10) -> str:
    """Shorten text to `to` characters."""
    from rich.text import Text
    t = Text(text)
    t.truncate(to, overflow="ellipsis")
    return str(t)


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


def post_mortem_debug(enable_traceback=True):
    """Call and enter pdb after an exception occurs globally.
    """
    def excepthook(exc_type, exc_value, exc_traceback):
        if enable_traceback:
            traceback.print_exception(exc_type, exc_value, exc_traceback)
        print("\nEntering post-mortem debugging...\n")
        pdb.post_mortem(exc_traceback)

    sys.excepthook = excepthook


def setup_signals(f: Callable):
  """No exit on SIGINT, exit on SIGQUIT, call `f` beforehand.

  Use e.g. to save a model or something.
  """

  def sigint_handler(sig, unused_frame):
    f(sig)
    L.info(r'Use `Ctrl+\` to save and exit.')

  prev_sigquit_handler = signal.getsignal(signal.SIGQUIT)
  def sigquit_handler(sig, unused_frame):
    signal.signal(signal.SIGQUIT, prev_sigquit_handler)
    f(sig)
    L.info(r'Exiting on `Ctrl+\`')
    # Re-raise for clean exit.
    os.kill(os.getpid(), signal.SIGQUIT)

  signal.signal(signal.SIGINT, sigint_handler)
  signal.signal(signal.SIGQUIT, sigquit_handler)


def seed_all(seed: int):
#   import jax
  import numpy as np
  import random
  import torch

  random.seed(seed)
  np.random.seed(seed)
#   jax.random.PRNGKey(seed)
  torch.manual_seed(seed)
