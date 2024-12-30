"""Create dataset (I, O, program lines) from the DSL in `tk.arc.p3`.
"""
# %%
from __future__ import annotations
import pandas as pd
from tk.arc.p3.dsl import *
from tk.arc.p3 import dsl, solver
import inspect

dslfunc2callable = {
    name: call for name, call 
    in inspect.getmembers(dsl, inspect.isfunction)
}
# %%
import ast

def represent_arg(node: ast.AST):
    if hasattr(node, 'id'):
        return node.id
    if hasattr(node, 'value'):
        return node.value
    if isinstance(node, ast.UnaryOp):
        op = -1 if isinstance(node, ast.USub) else 1
        return op * node.operand.value
    raise Exception(f'Unknown node type: {node}')


def get_structured_func_lines(function: Callable | ast.Module):
    if not isinstance(function, ast.Module):
        function = ast.parse(inspect.getsource(function))
    funcs = []
    arguments = []
    variables = []
    for node in ast.walk(function):
        if isinstance(node, ast.Assign):
            assert isinstance(node.value, ast.Call)
            funcs.append(node.value.func.id)
            arguments.append([represent_arg(arg) for arg in node.value.args])
            variables.append(node.targets[0].id)
        if isinstance(node, ast.Return):
            if isinstance(node.value, ast.Name):
                # we always have `return O`
                assert node.value.id == 'O'
    return variables, funcs, arguments


pid2solvecall = {
    name.removeprefix('solve_'): call 
    for name, call in 
    inspect.getmembers(solver, inspect.isfunction)
    if name.startswith('solve_')
}

# get the function whose string representation contains 'G'
for name, call in pid2solvecall.items():
    if name in (
        '007bbfb7'
    ):
        continue
    if 'G' in (s := inspect.getsource(call)):
        print(s)
        print(name)
        break
get_structured_func_lines(pid2solvecall['017c7c7b'])
# %%

from typing import NamedTuple

class Func(NamedTuple):
    vars: list[str]
    funcs: list[str]
    args: list[list]
    def zip(self):
        zippd = zip(self.vars, self.funcs, self.args)
        return [FuncW(*x) for x in zippd]

class FuncW(NamedTuple):
    var: str
    func: str
    args: list
    def unzip(self, *other: FuncW):
        vars_all = [self.var] + [o.var for o in other]
        funcs_all = [self.func] + [o.func for o in other]
        args_all = [self.args] + [o.args for o in other]
        return Func(vars_all, funcs_all, args_all)

class IO(NamedTuple):
    i: list[dsl.Grid]
    o: list[dsl.Grid]
    def zip(self):
        return [IOW(x, y) for x, y in zip(self.i, self.o)]

class IOW(NamedTuple):
    i: dsl.Grid
    o: dsl.Grid
    def unzip(self, *other: IOW):
        return IO(
            [self.i] + [o.i for o in other],
            [self.o] + [o.o for o in other]
        )

class ProblemSpec(NamedTuple):
    id: str
    train: IO
    test: IO
    func: Func

class ProblemSpecW(NamedTuple):
    id: str
    train: list[IOW]
    test: list[IOW]
    func: FuncW


pid2funcrepr = {}
for problem_id, callable in pid2solvecall.items():
    variables, funcs, arguments = get_structured_func_lines(callable)
    pid2funcrepr[problem_id] = Func(variables, funcs, arguments)
# %%
from tk.arc.p3 import get_data
sample_id = '007bbfb7'
input_examples = get_data('training')
print(len(pid2funcrepr))
print(len(input_examples['train']))
print(len(input_examples['test']))
print()
print(input_examples['train'][sample_id])
print(input_examples['test'][sample_id])
# %%
id2train = {
    k: IO(
        [v['input'] for v in vs],
        [v['output'] for v in vs])
    for k, vs in input_examples['train'].items()}
id2test = {
    k: IO(
        [v['input'] for v in vs],
        [v['output'] for v in vs])
    for k, vs in input_examples['test'].items()}
assert (
    id2train.keys() == 
    id2test.keys() == 
    pid2funcrepr.keys()
)
# %%

problems = {
    k: ProblemSpec(
        k,
        id2train[k],
        id2test[k],
        pid2funcrepr[k]
    )
    for k in pid2funcrepr
}
# %%
from rich import print as rp
rp(problems[sample_id])

# %%
# import jax

# def tokenize(value: str | int):
#     if isinstance(value, (IO, )):
#         return value
#     elif value in problems:
#         return value
#     elif value in dslfunc2callable:
#         return value
#     return list(str(value))


# problems_tok = jax.tree.map(
#     lambda x: tokenize(x), 
#     problems, 
#     is_leaf=lambda x: (
#         isinstance(x, IO)
#         or isinstance(x, str) 
#         or isinstance(x, int)
#     )
# )

# %%
import jax
import functools as ft
from enum import Enum, auto


class TokenType(Enum):
    IN = auto()
    OUT = auto()
    VAR = auto()
    FUNC = auto()
    ARG = auto()
    MISC = auto()


def interleave(xs: Iterable, sep: Any) -> list:
    xs = list(xs)
    return ft.reduce(lambda acc, x: acc + [sep, x], xs[1:], xs[:1])

def flat(xs):
    return jax.tree.flatten(xs)[0]

assert interleave([1, 2, 3], 0) == [1, 0, 2, 0, 3]
assert flat([[1, 2], [3, 4]]) == [1, 2, 3, 4]


class SepConfig(NamedTuple):
    sep_2d: str | None = '\n'
    sep_io: str | None = '=>'
    sep_args: str | None = ','
    sep_func: str | None = ';'
    sep_var: str | None = '='
    sep_prog: str | None = '<prog>'
    sep_end: str | None = '<end>'
    sep_pad: str | None = '<pad>'


def tokenize_with_sep(
    spec: ProblemSpec,
    sep_config: SepConfig = SepConfig(),
    padlen: int | None = None,
    vocab: set | dict | None = None,
):
    """
    NB: 
    * all entries of vocab will not be tokenized.
    * padlen, if provided, pads all examples to given value.
    * truncation is not performed.
    """
    vocab = vocab or {}
    out = []
    types = []

    def jointly(
        xs: str | None | Iterable[str | None], 
        type: TokenType = TokenType.MISC
    ):
        if xs is None: return
        if isinstance(xs, str): xs = [xs]
        to_extend = [x for x in xs if x is not None]
        out.extend(to_extend)
        types.extend([type] * len(to_extend))

    for gridin, gridout in spec.train.zip():
        gridin = interleave(gridin, [sep_config.sep_2d])
        gridout = interleave(gridout, [sep_config.sep_2d])
        jointly(map(str, flat(gridin)), TokenType.IN)
        jointly(sep_config.sep_io)
        jointly(map(str, flat(gridout)), TokenType.OUT)

    jointly(sep_config.sep_prog) 
    tokenize_arg = lambda a: a if a in vocab else list(a)

    for v, f, a in spec.func.zip():
        jointly(tokenize_arg(v), TokenType.VAR)
        jointly(sep_config.sep_var)
        jointly(tokenize_arg(f), TokenType.FUNC)
        jointly('(', TokenType.MISC)
        args_sep = interleave(a, sep_config.sep_args)
        jointly(flat([tokenize_arg(str(x)) for x in args_sep]), TokenType.ARG)
        jointly(')', TokenType.MISC)
        jointly(sep_config.sep_func)
    jointly(sep_config.sep_end)
    if padlen is not None:
        n = padlen - len(out)
        jointly([sep_config.sep_pad] * n)
    return out, types

from absl import logging

def maybe_truncate_train_examples(
    spec: ProblemSpec,
    max_tokens: int,
    spec2toks: Callable[[ProblemSpec], tuple],
):
    out, types = spec2toks(spec)
    meta = {'skipped': []}
    while len(out) > max_tokens:
        if len(spec.train.i) == 0:
            logging.warning(f"{spec.id}: Can't truncate further ({len(out)=})")
            return None, meta
        meta['skipped'].append(IOW(spec.train.i[-1], spec.train.o[-1]))
        spec = ProblemSpec(
            spec.id,
            IO(
                spec.train.i[:-1],
                spec.train.o[:-1]
            ),
            spec.test,
            spec.func
        )
        out, types = spec2toks(spec)
    return (out, types), meta


f1, t1 = tokenize_with_sep(
    problems[sample_id], 
    vocab=dslfunc2callable,
    padlen=650,
)
print(f1)
# "unit test"
assert f1[-1] == '<pad>'
assert len(f1) == len(t1) == 650

# %%
maybe_truncate_train_examples(
    problems[sample_id],
    256, 
    ft.partial(
        tokenize_with_sep,
        sep_config=SepConfig(), 
        vocab=dslfunc2callable)
)
# %%
from tk.arc.p3 import const
constants_from_solve = {x for x in dir(const) if not x.startswith('__')}
print(f"{constants_from_solve=}")
# %%
vocab = set(dslfunc2callable) | set(constants_from_solve)
problems_tokenized = {
    k: maybe_truncate_train_examples(
        spec,
        max_tokens=2048, 
        spec2toks=ft.partial(
            tokenize_with_sep, 
            sep_config=SepConfig(),
            padlen=2048,
            vocab=vocab
        )
    )
    for k, spec in problems.items()
}

# %%

def induce_vocab(problems_tokenized):
    vocab = set()
    for (toks, types), meta in problems_tokenized.values():
        vocab.update(toks)
    return {v: i for i, v in enumerate(vocab)}

vocab = induce_vocab(problems_tokenized)
rp(vocab)

# %%

def encode(
    toks: list[str], 
    vocab: dict[str, int]
):
    return [vocab[t] for t in toks]

encode(
    problems_tokenized[sample_id][0][0],
    vocab=vocab
)
# %%
import datasets as hfd
dataset = hfd.Dataset.from_dict({
    'input_ids': [
        encode(toks, vocab)
        for (toks, _), _ in problems_tokenized.values()
        if len(toks) <= 2048
    ],
    'token_type_ids': [
        [t.value for t in types]
        for (_, types), _ in problems_tokenized.values()
        if len(types) <= 2048
    ]
}).with_format('jax')
print(dataset)
print(dataset['input_ids'][0])
print(dataset['token_type_ids'][0])
print(type(dataset['token_type_ids'][0]))
print(len(dataset['token_type_ids'][0]))
print(set(map(len, dataset['input_ids'])))
# %%
import tk
import json
dataset.save_to_disk(tk.datadir / 'mhodel_rearc')
with open(tk.datadir / 'mhodel_rearc' / 'vocab.json', 'w') as f:
    json.dump(vocab, f)
logging.info(f"Saved dataset to {tk.datadir / 'mhodel_rearc'}")
logging.info(f"Vocab size: {len(vocab)}")
# out = tk.datadir / 'mhodel_rearc'
# with open(out, 'wb') as f:
#     pickle.dump(problems_tokenized, f)
# %%
