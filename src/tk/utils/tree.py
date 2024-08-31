"""Tree ops are lovely and useful.
"""
from flax.traverse_util import *
from jax.tree_util import *
from jax.tree import *
# this ain't gonna work for typing but
# maybe useful as self reference
from typing import Container as Tree