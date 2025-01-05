# %%
import itertools as it
import functools as ft
import random
from typing import Dict, List, Iterator, Any, Iterable

import numpy as np

def interleave(xs: Iterable, sep: Any) -> list:
    xs = list(xs)
    return ft.reduce(lambda acc, x: acc + [sep, x], xs[1:], xs[:1])

def generate_samples(
    rng: np.random.Generator,
    func_arities: Dict[str, List[int]],
    n_lines: int | None = None,
    seed: tuple[str] = ('I', )
) -> Iterator:
    """Generate infinite stream of DSL programs."""
    funcs = list(func_arities)
    while True:
        n_statements = n_lines if n_lines else random.randint(3, 8)
        vars_available = list(seed)
        tokens = []
        for _ in range(n_statements):
            func_id = str(rng.choice(funcs))
            n_args = rng.choice(
                func_arities[func_id])
            next_var_num = len(vars_available)
            args = rng.choice(vars_available, size=n_args)
            args = it.chain(*interleave(
                args,
                ','))
            line_tokens = [
                'x', 
                str(next_var_num), 
                '=', func_id, 
                '(', *map(str, args), ')', ';'
            ]
            vars_available.append(f'x{next_var_num}')
            tokens.extend(line_tokens)
        yield ['<prog>', *tokens, '<end>']

# %%
if __name__ == "__main__":
    s = generate_samples(
        np.random.default_rng(0), 
        {'f': [1, 2, 3], 'g': [2, 3]}, 
        5, ('I', ))
    print(next(s))
    print()
    print(next(s))
# %%
