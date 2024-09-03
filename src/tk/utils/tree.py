"""Tree ops are lovely and useful.
"""
from flax.traverse_util import *
from jax.tree_util import *
from jax.tree import *
# this ain't gonna work for typing but
# maybe useful as self reference
from typing import Container as Tree
# idk these seem like treeops
from tk.utils.func import groupby
import jax.tree as t
import jax.tree_util as tu
