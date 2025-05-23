"""Create dataset (I, O, program lines) from the DSL in `tk.arc.p3`.
"""
# %%
from __future__ import annotations

DO_SAVE = {
    'base': False,
    'prog_only': False,
}

# %%
import tk
import json
import inspect
import ast
from pathlib import Path

from typing import *
from absl import logging
from tk.arc.p3 import dsl, solver, const


def represent_arg(node: ast.AST) -> str:
    """NB we return everything as str

    (meaning even number constants such as 32)
    """
    if hasattr(node, 'id'):
        return str(node.id)
    if hasattr(node, 'value'):
        return str(node.value)
    if isinstance(node, ast.UnaryOp):
        op = -1 if isinstance(node, ast.USub) else 1
        return str(op * node.operand.value)
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

class Io(NamedTuple):
    i: list[dsl.Grid]
    o: list[dsl.Grid]
    def zip(self):
        return [IoW(x, y) for x, y in zip(self.i, self.o)]

class IoW(NamedTuple):
    i: dsl.Grid
    o: dsl.Grid
    def unzip(self, *other: IoW):
        return Io(
            [self.i] + [o.i for o in other],
            [self.o] + [o.o for o in other]
        )

class ProblemSpec(NamedTuple):
    id: str
    train: Io
    test: Io
    func: Func

class ProblemSpecW(NamedTuple):
    id: str
    train: list[IoW]
    test: list[IoW]
    func: FuncW


def _get_problems(input_examples: dict[str, dict]) -> tuple:
    # %%
    pid2solvecall = {
        name.removeprefix('solve_'): call 
        for name, call in 
        inspect.getmembers(solver, inspect.isfunction)
        if name.startswith('solve_')
    }
    pid2funcrepr = {}
    for problem_id, callable in pid2solvecall.items():
        variables, funcs, arguments = get_structured_func_lines(callable)
        pid2funcrepr[problem_id] = Func(variables, funcs, arguments)
    # %%
    id2train = {
        k: Io(
            [v['input'] for v in vs],
            [v['output'] for v in vs])
        for k, vs in input_examples['train'].items()}
    id2test = {
        k: Io(
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
    return id2train, id2test, pid2funcrepr, problems


def get_problems(input_examples: dict[str, dict]):
    *_, problems = _get_problems(input_examples)
    return problems


# %%
ismain = __name__ == '__main__'

if ismain:
    # %%
    from tk.arc.p3 import get_data
    input_examples = get_data('training')
    id2train, id2test, pid2funcrepr, problems = _get_problems(input_examples)

    sample_id = '007bbfb7'
    print(len(pid2funcrepr))
    print(len(input_examples['train']))
    print(len(input_examples['test']))
    print()
    print(input_examples['train'][sample_id])
    print(input_examples['test'][sample_id])
    from rich import print as rp
    rp(problems[sample_id])

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
    * all entries of vocab will be kept as-is.
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
        to_extend = [str(x) for x in xs if x is not None]
        out.extend(to_extend)
        types.extend([type] * len(to_extend))

    for gridin, gridout in spec.train.zip():
        gridin = interleave(gridin, [sep_config.sep_2d])
        gridout = interleave(gridout, [sep_config.sep_2d])
        jointly(map(str, flat(gridin)), TokenType.IN)
        jointly(sep_config.sep_io)
        jointly(map(str, flat(gridout)), TokenType.OUT)
        jointly(sep_config.sep_func)

    jointly(sep_config.sep_prog) 
    tokenize_arg = lambda a: str(a) if str(a) in vocab else list(str(a))

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


def detokenize_with_sep(
    tokens: list[str],
    # types: list[TokenType],
    sep_config: SepConfig = SepConfig(),
) -> tuple:
    """Reverses tokenize_with_sep operation to reconstruct a ProblemSpec.
    
    Args:
        tokens: List of string tokens
        types: List of TokenType values corresponding to each token
        sep_config: Separator configuration used during tokenization
    
    Returns:
        Io and Func objects
    """
    prog_idx = next((i for i, x in enumerate(tokens) if x == sep_config.sep_prog), None)
    end_idx = next((i for i, x in enumerate(tokens) if x == sep_config.sep_end), len(tokens))
    
    io_tokens = []
    prog_tokens = tokens
    if prog_idx:
        io_tokens = tokens[:prog_idx]
        # io_types = types[:prog_idx]
        prog_tokens = tokens[prog_idx + 1:end_idx]
        # prog_types = types[prog_idx + 1:end_idx]

    io_parts = [ [[]] ]
    for token in io_tokens:
        if token in (sep_config.sep_io, sep_config.sep_func):
            io_parts.append([[]])
        elif token == sep_config.sep_2d:
            io_parts[-1].append([])
        else:
            io_parts[-1][-1].append(int(token))

    inputs = [x for i, x in enumerate(io_parts) if i % 2 == 0]
    outputs = [x for i, x in enumerate(io_parts) if i % 2 == 1]

    vars_list = []
    funcs_list = []
    args_list = []
    
    current_tokens = []
    current_args = []
    # NB not proper state machine parsing, will only work for current DSL 
    for token in prog_tokens:
        if token == '(':
            funcs_list.append(''.join(current_tokens))
            current_tokens = []
        elif token in (sep_config.sep_args, ):
            current_args.append(''.join(current_tokens))
            current_tokens = []
        elif token in (')', sep_config.sep_func):
            if current_tokens:
                current_args.append(''.join(current_tokens))
            if current_args:
                args_list.append(current_args)
            current_args = []
            current_tokens = []
        elif token == sep_config.sep_var:
            vars_list.append(''.join(current_tokens))
            current_tokens = []
        else:
            current_tokens.append(token)

    return Io(inputs, outputs), Func(vars_list, funcs_list, args_list)


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
        meta['skipped'].append(IoW(spec.train.i[-1], spec.train.o[-1]))
        spec = ProblemSpec(
            spec.id,
            Io(
                spec.train.i[:-1],
                spec.train.o[:-1]
            ),
            spec.test,
            spec.func
        )
        out, types = spec2toks(spec)
    return (out, types), meta


if ismain:
    dslfunc2callable = {
        name: call for name, call 
        in inspect.getmembers(dsl, inspect.isfunction)
    }
    f1, t1 = tokenize_with_sep(
        problems[sample_id], 
        vocab=dslfunc2callable,
        padlen=650,
    )
    print(f1)
    # "unit test"
    assert f1[-1] == '<pad>'
    assert len(f1) == len(t1) == 650

    ios, func = detokenize_with_sep(f1)
    print(func.args)
    print(problems[sample_id].func.args)
    assert func.args == problems[sample_id].func.args

    # %%

    # check if equal, print diff if not
    # TODO move to test etc

    def eq(a, b, name):
        def _chk(xs, ys):
            if isinstance(xs, (list, tuple, )):
                return [_chk(x, y) for x, y in zip(xs, ys)]
            if xs != ys:
                print(f"NEQ {name}: {xs=}, {ys=}")

        _chk(a, b)

    eq(ios.i, problems[sample_id].train.i, 'train.i')
    eq(ios.o, problems[sample_id].train.o, 'train.o')
    eq(func.vars, problems[sample_id].func.vars, 'func.vars')
    eq(func.funcs, problems[sample_id].func.funcs, 'func.funcs')
    eq(func.args, problems[sample_id].func.args, 'func.args')

# %%
# maybe_truncate_train_examples(
#     problems[sample_id],
#     256, 
#     ft.partial(
#         tokenize_with_sep,
#         sep_config=SepConfig(), 
#         vocab=dslfunc2callable)
# )
# %%
# vocab = set(dslfunc2callable) | set(constants_from_solve)
# problems_tokenized = {
#     k: maybe_truncate_train_examples(
#         spec,
#         max_tokens=2048, 
#         spec2toks=ft.partial(
#             tokenize_with_sep, 
#             sep_config=SepConfig(),
#             padlen=2048,
#             vocab=vocab
#         )
#     )
#     for k, spec in problems.items()
# }

# %%

def induce_vocab(problems_tokenized, sep_config=SepConfig()):
    vocab = set()
    for (toks, types), meta in problems_tokenized.values():
        vocab.update(toks)
    vocab = {v: i for i, v in enumerate(vocab)}
    vocab['__config'] = {
        **sep_config._asdict(),
    }
    return vocab

def encode(
    toks: list[str], 
    vocab: dict[str, int]
):
    return [vocab[t] for t in toks]

# encode(
#     problems_tokenized[sample_id][0][0],
#     vocab=vocab
# )
# %%
# import datasets as hfd
# dataset = hfd.Dataset.from_dict({
#     'id': list(problems_tokenized.keys()),
#     'input_ids': [
#         encode(toks, vocab)
#         for (toks, _), _ in problems_tokenized.values()
#         if len(toks) <= 2048
#     ],
#     'token_type_ids': [
#         [t.value for t in types]
#         for (_, types), _ in problems_tokenized.values()
#         if len(types) <= 2048
#     ]
# })
# print(dataset)
# print(dataset['input_ids'][0])
# print(dataset['token_type_ids'][0])
# print(len(dataset['token_type_ids'][0]))
# print(set(map(len, dataset['input_ids'])))
# # %%
# def hfd_save(dataset, vocab, outdir: Path):
#     dataset.save_to_disk(outdir)
#     with open(outdir / 'vocab.json', 'w') as f:
#         json.dump(vocab, f)
#     logging.info(f"Saved dataset to {outdir}")
#     logging.info(f"Vocab size: {len(vocab)}")


# if DO_SAVE['base']:
#     hfd_save(
#         dataset.with_format('jax'),
#         vocab, 
#         tk.datadir / 'mhodel_rearc' / 'full'
#     )

# %%
# only_programs_dataset = {}
# # use '<prog>' as separator and extract later tokens
# for k, ((toks, *_), *_) in problems_tokenized.items():
#     toks_after_prog = toks[toks.index('<prog>'):toks.index('<end>')]
#     only_programs_dataset[k] = {'input_ids': toks_after_prog}

# # now ensure padding to maxlen
# maxlen = max(len(x['input_ids']) for x in only_programs_dataset.values())
# print(max(only_programs_dataset.values(), key=lambda x: len(x['input_ids'])))
# for k, v in only_programs_dataset.items():
#     v['input_ids'] = v['input_ids'] + [vocab['__config']['sep_pad']] * (maxlen - len(v['input_ids']))

# print(f"{maxlen=}")
# assert all(len(v['input_ids']) == maxlen for v in only_programs_dataset.values())
# # %%
# ds_programs = hfd.Dataset.from_dict({
#     'id': [k for k in only_programs_dataset],
#     'input_ids': [
#         encode(v['input_ids'], vocab)
#         for v in only_programs_dataset.values()],
# })
# print(ds_programs)
# print(ds_programs['input_ids'][0])
# # %%
# if DO_SAVE['prog_only']:
#     hfd_save(
#         ds_programs.with_format('jax'),
#         vocab, 
#         tk.datadir / 'mhodel_rearc' / 'prog_only'
#     )
# %%