"""Custom generalization task with n-way dependencies.
"""

from tk.rpe.tasks import task
# %%
from typing import *
import functools

import chex
import jax
import jax.numpy as jnp
import random


if main := (__name__ == "__main__"):
    print("Running as script")


def interpret_toks(toks: list[str], positions: dict[str, list[int]]) -> dict[int, int]:
    state: list[int] = [1]
    is_start: bool = False
    outputs: dict[int, int] = {}
    for i, tok in enumerate(toks):
        if i in positions['start']:
            is_start = True
        elif i in positions['stop']: 
            is_start = False
            state = [1]  # reset all state
        elif i in positions['mul'] and is_start:
            state.append(int(tok))
        elif i in positions['num']:
            outputs[i] = int(tok) * state[-1]
    return outputs


def find_special(toks: list[str], special: dict[str, Collection[str]]) -> dict[str, Sequence[int]]:
    out = {}
    for i, tok in enumerate(toks):
        for k, vs in special.items():
            if tok in vs:
                out[k] = out.get(k, []) + [i]
    return out


assert( find_special(
    ['1', 'a', 'start', '100', 'b', '2', '200', '5', 'stop', '100', '5'],
    {'start': ['start'], 'stop': ['stop'], 'mul': {'100', '200'}, 'num': set(map(str, range(10)))},
)) == {'start': [2], 'stop': [8], 'mul': [3, 6, 9], 'num': [0, 5, 7, 10]}
assert interpret_toks(
    ['1', 'a', 'start', '100', 'b', '2', '200', '5', 'stop', '100', '5'],
    {'start': [2], 'stop': [8], 'mul': [3, 6, 9], 'num': [0, 5, 7, 10]}
) == {5: 200, 7: 1000, 10: 5, 0: 1}
# %%

def generate_multi_dependency_string(
    rng: random.Random, #chex.PRNGKey,
    n: int,
    ndeps: int,
    n_start_tokens: int = 2,
    n_stop_tokens: int = 2,
):
    """
    We are looking to generate contexts in which one particular token at the
    beginning of the sequence is going to modulate other succeeding tokens but
    not all of them.

    We have multiple "start" and a "stop" token for the modulation,
    given by the arguments to this function.

    TODO: In a more advanced setting we might split these further into
    multiple tokens as well.

    Example:
        >>> assert interpret_toks(
            ['1', 'a', 'start', '100', 'b', '2', '200', '5', 'stop', '100', '5'],
            {'start': [2], 'stop': [8], 'mul': [3, 6, 9], 'num': [5, 7, 10]}
        ) == {5: 200, 7: 1000, 10: 5}
    """
    start_tokens = [f'<start{i}>' for i in range(n_start_tokens)]
    stop_tokens = [f'<stop{i}>' for i in range(n_stop_tokens)]
    letters_mul = ['100', ]
    letters_num = list('0123456789')
    letters_str = list('abcdefghij')  #klmnopqrstuvwxyz')
    vocab = letters_str + letters_num + letters_mul + start_tokens + stop_tokens
    vocab = {v: i for i, v in enumerate(vocab)}
    if n < n_start_tokens + ndeps + n_stop_tokens:
        return rng.choices(letters_str, k=n), {}, vocab
    positions = rng.sample(range(n), k=4*ndeps)
    positions_start = positions[:ndeps]
    positions_end = positions[ndeps:2*ndeps]
    positions_mul = positions[2*ndeps:3*ndeps]
    positions_num = positions[3*ndeps:4*ndeps]
    toks = rng.choices(letters_str, k=n)
    for ps in positions_start: toks[ps] = rng.choice(start_tokens)
    for pe in positions_end: toks[pe] = rng.choice(stop_tokens)
    for pm in positions_mul: toks[pm] = rng.choice(letters_mul)
    for pn in positions_num: toks[pn] = rng.choice(letters_num)
    outputs = interpret_toks(toks, {
        'start': positions_start,
        'stop': positions_end,
        'mul': positions_mul,
        'num': positions_num
    })
    return toks, outputs, vocab

# %%
from jax import nn as jnn

def encode(toks, outputs, vocab, input_size=None, output_size=None):
    input_size = input_size or len(vocab)
    output_size = output_size or len(vocab)
    batch = {
        'input': jnp.array([vocab[tok] for tok in toks]),
        'output': jnp.array([vocab[outputs.get(t, t)] for t in toks])
    }
    inputs = jnn.one_hot(batch['input'], input_size)
    output = jnn.one_hot(batch['output'], output_size)
    return dict(input=inputs, output=output)


if main:
    t, o, v = generate_multi_dependency_string(
        random.Random(42), 32, 4, 1
    )
    print(o)
    print(encode(t, o, v))
# %%

# %%


class TwoValCopy(task.GeneralizationTask):
    """Task where we rely on exactly two values.
    """
    def __init__(self):
        super().__init__()
        *_, v = generate_multi_dependency_string(
            random.Random(42), 32, 4, 1
        )
        self.v = v

    def sample_batch(
        self, rng: chex.PRNGKey, batch_size: int, length: int
    ) -> task.Batch:
        batch = {'input': [], 'output': []}
        for _ in range(batch_size):
            k, rng = jax.random.split(rng)
            t, o, v = generate_multi_dependency_string(
                random.Random(k[0].item()), length, 1, 1
            )
            cur = encode(t, o, v)
            batch['input'].append(cur['input'])
            batch['output'].append(cur['output'])
        return {
            'input': jnp.stack(batch['input']),
            'output': jnp.stack(batch['output'])
        }

    @property
    def num_classes(self) -> int:
        return len(self.v)
    @property
    def input_size(self) -> int:
        return len(self.v)
    @property
    def output_size(self) -> int:
        return len(self.v)

    def output_length(self, input_length: int) -> int:
        """Returns the output length for a given input length."""
        return input_length
    
# %%
