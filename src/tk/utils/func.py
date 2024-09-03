from collections import defaultdict
from operator import *
from functools import *
from itertools import *
from collections import *
from typing import NamedTuple
# NB, PyTreeNode is base class for a node in pytree
# so I don't have to type isinstance frequently
# prefer this to namedtuple
from flax.struct import dataclass, field, PyTreeNode, serialization
from flax.struct import serialization as flax_serialize
import itertools as it

from operator import (
    itemgetter as iget
)
from loguru import logger
from typing import Callable, Iterable, TypeVar

K = TypeVar('K')
V = TypeVar('V')


def groupby(i: Iterable[V], f: Callable[[V], K] | None = None) -> dict[K, list[V]]:
    outs = defaultdict(list)
    for k, v in it.groupby(i, f):
        outs[k].extend(v)
    logger.debug({k: len(v) for k, v in outs.items()})
    return dict(outs)
