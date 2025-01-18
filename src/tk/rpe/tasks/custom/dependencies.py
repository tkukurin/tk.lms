"""Custom generalization task with n-way dependencies.

Example train dynamics 
* sin/cos embeds
* sequence <=19 toks
* 2 it/s on T4

---

{'step': 0, 'train_loss': 3.717655897140503, 'train_accuracy': 0.013822115957736969}
{'step': 100, 'train_loss': 2.959900140762329, 'train_accuracy': 0.0703125}
{'step': 200, 'train_loss': 2.7089343070983887, 'train_accuracy': 0.1647135466337204}
{'step': 300, 'train_loss': 2.5639524459838867, 'train_accuracy': 0.1931818276643753}
{'step': 400, 'train_loss': 2.5840137004852295, 'train_accuracy': 0.16899670660495758}
{'step': 500, 'train_loss': 2.2197928428649902, 'train_accuracy': 0.22940342128276825}
{'step': 600, 'train_loss': 1.8815103769302368, 'train_accuracy': 0.3179687559604645}
{'step': 700, 'train_loss': 0.7366040945053101, 'train_accuracy': 0.56640625}

---

{'length': 1, 'accuracy': np.float64(1.0)}
{'length': 2, 'accuracy': np.float64(0.89453125)}
{'length': 3, 'accuracy': np.float64(0.8274739682674408)}
{'length': 4, 'accuracy': np.float64(0.78369140625)}
{'length': 5, 'accuracy': np.float64(0.7632812485098839)}
{'length': 6, 'accuracy': np.float64(0.7578125149011612)}
{'length': 7, 'accuracy': np.float64(0.718191996216774)}
{'length': 8, 'accuracy': np.float64(0.645263671875)}
{'length': 9, 'accuracy': np.float64(0.582899309694767)}
{'length': 10, 'accuracy': np.float64(0.574414074420929)}
{'length': 11, 'accuracy': np.float64(0.547585241496563)}
{'length': 12, 'accuracy': np.float64(0.5136718973517418)}
{'length': 13, 'accuracy': np.float64(0.5220853611826897)}
{'length': 14, 'accuracy': np.float64(0.459263414144516)}
{'length': 15, 'accuracy': np.float64(0.39700523391366005)}
{'length': 16, 'accuracy': np.float64(0.374267578125)}
{'length': 17, 'accuracy': np.float64(0.3877527602016926)}
{'length': 18, 'accuracy': np.float64(0.4044053815305233)}
{'length': 19, 'accuracy': np.float64(0.3842516466975212)}
{'length': 20, 'accuracy': np.float64(0.29521485045552254)}
{'length': 21, 'accuracy': np.float64(0.1935453899204731)}
{'length': 22, 'accuracy': np.float64(0.14701705053448677)}
{'length': 23, 'accuracy': np.float64(0.13637907803058624)}
{'length': 24, 'accuracy': np.float64(0.12744140904396772)}
{'length': 25, 'accuracy': np.float64(0.12484374735504389)}
{'length': 26, 'accuracy': np.float64(0.1240234412252903)}
{'length': 27, 'accuracy': np.float64(0.12666377425193787)}
"""

from rpe.tasks import task
# %%
import dataclasses as dc
from typing import *
import functools as ft
import itertools as it

import chex
import jax
import jax.numpy as jnp
import random

from typing import TypedDict


class TokTypes(TypedDict):
    starts: list[int]
    stops: list[int]
    muls: list[int]
    nums: list[int]
    chrs: list[int]


if main := (__name__ == "__main__"):
    print("Running as script")


def interpret_toks(toks: list[str], positions: dict[str, list[int]]) -> dict[int, int]:
    """Default interpret the multi-dependency case.
    """
    state: list[int] = [1]
    is_start: bool = False
    outputs: dict[int, int] = {}
    for i, tok in enumerate(toks):
        if i in positions['starts']:
            is_start = True
        elif i in positions['stops']: 
            is_start = False
            state = [1]  # reset all state
        elif i in positions['muls'] and is_start:
            state.append(int(tok))
        elif i in positions['nums']:
            outputs[i] = str((int(tok) * state[-1]) % 1000)
    return outputs


def find_special(toks: list[str], special: dict[str, Collection[str]]) -> dict[str, list[int]]:
    out = {}
    for i, tok in enumerate(toks):
        for k, vs in special.items():
            if tok in vs:
                out[k] = out.get(k, []) + [i]
    return out


assert( find_special(
    ['1', 'a', '<<', '100', 'b', '2', '200', '5', '>>', '100', '5'],
    {'starts': ['<<'], 'stops': ['>>'], 'muls': {'100', '200'}, 'nums': set(map(str, range(10)))},
)) == {'starts': [2], 'stops': [8], 'muls': [3, 6, 9], 'nums': [0, 5, 7, 10]}
assert interpret_toks(
    ['1', 'a', '<<', '100', 'b', '2', '200', '5', '>>', '100', '5'],
    {'starts': [2], 'stops': [8], 'muls': [3, 6, 9], 'nums': [0, 5, 7, 10]}
) == {k: str(v) for k, v in ({5: 200, 7: 0, 10: 5, 0: 1}).items()}
# %%

@dc.dataclass
class Voc:
    kind: TokTypes = dc.field(repr=False)
    t2i: dict = dc.field(repr=True)
    __getitem__ = lambda self, key: self.t2i[key]
    __len__ = lambda self: len(self.t2i)
    get = lambda self, key: self.t2i.get(key)


def mkvocab(
    *,
    nums = tuple('0123456789'),
    chrs = tuple('abcdefghij'), #klmnopqrstuvwxyz')
    muls = tuple(map(str, range(0, 1000, 100))),
    **adds,
):
    ks = set(TokTypes.__annotations__)
    kind = {
        **{k: [] for k in ks},
        **dict(muls=muls, chrs=chrs, nums=nums),
        **adds,
    }
    assert set(kind) <= ks, set(kind)
    vocab = it.chain(nums, muls, chrs, *adds.values())
    vocab = {v: i for i, v in enumerate(vocab)}
    return Voc(kind, vocab)


def generate_multi_dependency_string(
    rng: random.Random, #chex.PRNGKey,
    n: int,
    ndeps: int,
    vocab: Voc,
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
    if 0 == (ndeps_all := min(4, n // ndeps) * ndeps):
        ndeps_all = rng.randint(0, n)
    positions = rng.sample(range(n), k=ndeps_all)
    toks = rng.choices(vocab.kind['chrs'], k=n)
    pos_out = ['starts', 'stops', 'muls', 'nums']
    rng.shuffle(pos_out)
    outs = {k: [] for k in pos_out}
    for i, kind in enumerate(pos_out):
        outs[kind] = positions[i * ndeps: (i + 1) * ndeps]
        if opts := vocab.kind[kind]:
            for ix in outs[kind]:
                toks[ix] = rng.choice(opts)
    outputs = interpret_toks(toks, outs)
    return toks, outputs

if main: print(mkvocab())
# %%
from jax import nn as jnn

def encode(
    toks: list, ix2out: dict, vocab: dict,
    input_size=None, output_size=None
):
    """encode: identity for token or multiply
    """
    inps = [vocab[t] for t in toks]
    outs = [vocab[ix2out.get(i, t)] for i, t in enumerate(toks)]
    batch = {
        'input': jnn.one_hot(jnp.array(inps), input_size or len(vocab)),
        'output': jnn.one_hot(jnp.array(outs), output_size or len(vocab)),
    }
    if not hasattr(encode, "logged"):
        setattr(encode, "logged", True)
        logval = list(zip(enumerate(toks), inps, outs))
        print(f"===\nENCODE:\n{vocab=}\n{ix2out=}\n{logval}")
    return batch


if main:
    t, o = generate_multi_dependency_string(
        random.Random(42), 64, 5, (
            v := mkvocab(
                starts=("<<", "[["), 
                stops=(">>", "]]")))
    )
    print(t)
    print(o)
    batch = encode(t, o, v)
    print(batch['output'])
    print(jnp.argmax(batch['output'], axis=-1))
    print(jnp.sum(batch['output'], axis=-1))

# %%


class MultiTokenDep(task.GeneralizationTask):
    """Task where we rely on multi-token interactions."""

    def __init__(self):
        super().__init__()
        self.nstart = 2
        self.nstop = 2
        self.ndeps = 3
        self.vocab = mkvocab(
            starts=[f'<s{i}>' for i in range(self.nstart)], 
            stops=[f'</s{i}>' for i in range(self.nstop)]
        )
        self.vocab_size = len(self.vocab)

    def sample_batch(
        self, rng: chex.PRNGKey, batch_size: int, length: int
    ) -> task.Batch:
        batch = {'input': [], 'output': []}
        for _ in range(batch_size):
            k, rng = jax.random.split(rng)
            t, o = generate_multi_dependency_string(
                random.Random(k[0].item()), length, self.ndeps, self.vocab
            )
            cur = encode(t, o, self.vocab)
            batch['input'].append(cur['input'])
            batch['output'].append(cur['output'])
        return {
            'input': jnp.stack(batch['input']),
            'output': jnp.stack(batch['output'])
        }

    num_classes = property(lambda s: s.vocab_size)
    input_size = property(lambda s: s.vocab_size)
    output_size = property(lambda s: s.vocab_size)
    output_length = lambda s, input_length: input_length
    
# %%
