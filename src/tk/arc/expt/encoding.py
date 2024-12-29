"""Various conversion conveniences.
"""
from __future__ import annotations

from collections import Counter
import itertools as it
from pathlib import Path
import inspect

import pandas as pd
import numpy as np
from loguru import logger
from typing import Callable, Literal
from tk.arc.p3 import dsl


def split_stored_df(df: pd.DataFrame | str | Path) -> tuple[
    pd.DataFrame, pd.DataFrame, pd.DataFrame, Tokenizer
]:
    """Process program data.
    
    Args:
        df: DataFrame or path to pickled DataFrame
        
    Returns:
        Tuple of (original_df, io_df, grouped_df, tokenizer)
    """
    if isinstance(df, (str, Path)):
        df = pd.read_pickle(df)
    assert isinstance(df, pd.DataFrame)
    
    df_io = df[['input', 'output']].drop_duplicates()
    df = df.drop(columns=['input', 'output'])
    
    df_grouped = (
        df
        .xs(0, level='example_id', drop_level=True)
        .groupby(level='id')
        .agg({
            'function': list,
            'arguments': list, 
            'variable': list
        })
    )

    primitive2fn = {
        name: fn for name, fn in inspect.getmembers(dsl, inspect.isfunction)
    }

    vocab = {
        # Program syntax
        'def', 'solve', 'return', '(', ')', ';', '=', ',', ' ',
        # Special tokens  
        '<pad>', '</I>', '</O>', 'I', 'O', '<prog>',
        # Numbers
        *'1234567890',
        # Variables
        'x',
        # Functions
        *primitive2fn.keys()
    }

    vocab.update(
        str(x) for x in df['function'].unique() if not str(x).startswith('x')
    )
    vocab.update(
        str(x) for x in df['variable'].unique() if not str(x).startswith('x')
    )
    vocab.update(
        str(x) for x in sum(df['arguments'], []) if not str(x).startswith('x')
    )

    # Create vocab dict
    vocab_dict = {token: idx for idx, token in enumerate(vocab)}
    assert all(isinstance(str(x), str) for x in vocab_dict), (
        "Non-string keys in vocab"
    )

    return df, df_io, df_grouped, Tokenizer(vocab_dict)


class Tokenizer:
    """dict semantics + info about special tokens.
    """

    def __init__(self, tok2id: dict):
        assert len(set(tok2id.values())) == len(tok2id), (
            f"Duplicate values in vocab: {[(k, v) for k, v in Counter(tok2id.values()).most_common() if v > 1]}"
        )
        self.tok2id = tok2id
        self.id2tok = {v: k for k, v in self.tok2id.items()}
        self.pad_token = '<pad>'
        self.input_end = '</I>'  
        self.output_end = '</O>'
        self.program_start = '<prog>'
        self.separators = {';', ',', '(', ')', '=', ' '}

    def __getitem__(self, key: str) -> int:
        return self.tok2id[key]

    def items(self):
        return self.tok2id.items()

    def keys(self):
        return self.tok2id.keys()
    
    def get(self, key: str, default):
        return self.tok2id.get(key, default)
    
    def __len__(self):
        return len(self.tok2id)

    @property
    def program_start_id(self) -> int:
        return self[self.program_start]
        
    @property
    def pad_id(self) -> int:
        return self[self.pad_token]
        
    @property
    def input_end_id(self) -> int:
        return self[self.input_end]
        
    @property
    def output_end_id(self) -> int:
        return self[self.output_end]
        
    def is_separator(self, token: str) -> bool:
        return token in self.separators
    
    def save(self, outdir: Path):
        pd.DataFrame(
            list(self.tok2id.items()), 
            columns=['token', 'id'], 
            dtype=('str', 'int')
        ).to_parquet(outdir / 'vocab.parquet')
        return outdir / 'vocab.parquet'

    @classmethod
    def load(cls, loc: Path | str):
        loc = Path(loc)
        if not str(loc).endswith('.parquet'):
            loc = loc / 'vocab.parquet'
        vocab = pd.read_parquet(loc)
        vocab['id'] = vocab['id'].astype(int)
        vdict = vocab.to_dict('list')
        self = cls(dict(zip(vdict['token'], vdict['id'])))
        return self

from datasets import Dataset

def load_data(loc: Path | str):
    ds = Dataset.load_from_disk(str(loc))
    ds.set_format(type='jax', columns=['input_ids', 'attention_mask', 'token_type_ids'])
    tok = Tokenizer.load(loc)
    return ds, tok

def save_data(ds: Dataset, tok: Tokenizer, loc: Path | str):
    ds.save_to_disk(str(loc))
    pd.DataFrame(
        list(tok.tok2id.items()), 
        columns=['token', 'id'], 
        dtype=('str', 'int')
    ).to_parquet(Path(loc) / 'vocab.parquet')


class SimpleArcGridSeqEncoder:
    """Convert ARC problems to sequences of tokens.

    See p3dsl_to_df.py for how to generate dataframes.
    """

    def __init__(
        self, 
        tok: Tokenizer,
        df_io: pd.DataFrame,
        df_grouped: pd.DataFrame,
    ):
        self.tok = tok
        self.df_io = df_io
        self.df_grouped = df_grouped

    def _encode_problem(self, id: str) -> tuple:
        problems = self.df_io.loc[id].reset_index()
        inputs = [
            grid_to_flat_tokens(p) + [self.tok.input_end] 
            for p in problems['input']]
        outputs = [
            grid_to_flat_tokens(p) + [self.tok.output_end] 
            for p in problems['output']]
        row = self.df_grouped.loc[id]
        program = [self.tok.program_start] + row_to_tokens(row)
        return inputs, outputs, program

    def encode_problem(self, id: str, max_length: int = 99999) -> tuple:
        """Encode a problem into a list of tokens"""
        inputs_all, outputs_all, program = self._encode_problem(id)
        inputs = [[self.tok[i] for i in inp] for inp in inputs_all]
        outputs = [[self.tok[i] for i in out] for out in outputs_all]
        program = [self.tok[i] for i in program]
        assert len(program) <= max_length, f"Invalid example {id}"
        assert len(inputs) == len(outputs), f"Invalid example {id}, {len(inputs)} != {len(outputs)}"
        def combine(xss, yss): 
            outs = [xs + ys for xs, ys in zip(xss, yss)]
            types = [[0] * len(xs) + [1] * len(ys) for xs, ys in zip(xss, yss)]
            return outs, types
        flat = lambda xss: [x for xs in xss for x in xs]
        io_toks, io_types = combine(inputs, outputs)
        tokens = flat(io_toks) + program
        types = flat(io_types) + [2] * len(program)
        i = len(inputs) - 1
        skipped = {'in': [], 'out': []}
        while len(tokens) > max_length:
            skipped['in'].append(inputs[i])
            skipped['out'].append(outputs[i])
            inputs, outputs = inputs[:i], outputs[:i]
            io_toks, io_types = combine(inputs, outputs)
            tokens = flat(io_toks) + program
            types = flat(io_types) + [2] * len(program)
            i -= 1
            if i == -1:
                logger.warning(
                    f"Problem {id} is too long for {max_length}"
                    f" ({len(program)} program toks, {len(inputs_all)} examples)"
                )
                return None, None, skipped
        return tokens, types, skipped

    def encode_all_with_padding(
        self, max_length: int = 99999, quantile: float = 0.75) -> tuple:
        """Encode all problems with padding to the maximum length.

        If max length is given we trim problems, see encode_problem.
        TODO make nicer.

        If quantile is given, we remove outliers.
        """
        length_hist = {}
        encoded = {}
        truncated = {}
        for i in self.df_io.index.unique(level=0):
            tokens, types, skipped = self.encode_problem(i, max_length)
            if not tokens: continue
            if skipped.get('in'): truncated[i] = skipped
            length_hist[len(tokens)] = length_hist.get(len(tokens), 0) + 1
            encoded[i] = (tokens, types)
        skipped = {}
        max_length = max(length_hist.keys())
        if quantile is not None:
            qs: pd.Categorical = pd.qcut(list(length_hist.keys()), q=(0, quantile, 1.0))
            skipped = {
                k: v for k, v in encoded.items() if len(v[0]) >= qs[-1].left}
            encoded = {
                k: v for k, v in encoded.items() if len(v[0]) <  qs[-1].left}
            max_length = int(qs[-1].left + 1)
        encoded_padded = Dataset.from_list([
            {'id': k,
            'input_ids': tokens + [self.tok.pad_id] * (max_length - len(tokens)),
            'attention_mask': [1] * len(tokens) + [0] * (max_length - len(tokens)),
            'token_type_ids': types + [3] * (max_length - len(tokens))
            }
            for k, (tokens, types) in encoded.items()
        ]).with_format(
            type='jax', 
            columns=['input_ids', 'attention_mask', 'token_type_ids']
        )
        meta = dict(
            skipped_full=skipped, 
            truncated=truncated, 
            hist=length_hist)
        return encoded_padded, meta


def row_to_string(row: pd.Series):
    """Convert DataFrame row with list columns to semicolon-separated expressions"""
    expressions = []
    for var, func, args in zip(row['variable'], row['function'], row['arguments']):
        args = ','.join(s.strip() for s in map(str, args))
        expr = f"{var}={func}({args})"
        expressions.append(expr.replace(';', '\\;'))
    return ';'.join(expressions)


def row_to_tokens(row, splitx = True):
    """Convert DataFrame row with list columns to list of tokens"""
    tokens = []
    splitter = lambda x: [x]
    if splitx:
        def splitter(x):
            import re
            # if x.startswith('x'):  # x1
            if re.match(r'(x\d+|\d+)', x):
                return list(x)
            return [x]  # [function]
    for var, func, args in zip(row['variable'], row['function'], row['arguments']):
        args = it.chain(*(splitter(str(arg)) + [','] for arg in args))
        tokens.extend(map(str, (*splitter(var), '=', *splitter(func), '(', *args, ')')))
    return tokens


def compile_string(program: str) -> Callable:
    """Compile a string program into a callable function.
    
    Args:
        program: A string of semicolon-separated expressions
        
    Returns:
        A callable function that takes an input and returns the program output
    """
    body = [f"    {expr}" for expr in program.split(';')]
    body = '\n'.join(body)
    code = f"def program(I):\n{body}\n    return O"
    namespace = {}
    exec(code, globals(), namespace)
    return namespace['program']


def compile_row_using_dicts(row: pd.Series, fname2call: dict[str, Callable]) -> Callable:
    """Compile a row of the dataframe using dictionaries of functions"""
    def program(I):
        ctx = {'I': I}
        last = None
        for var, func, args in zip(row['variable'], row['function'], row['arguments']):
            func = fname2call[func]
            args = [ctx[arg.strip()] if isinstance(arg, str) else arg for arg in args]
            ctx[var] = func(*args)
            last = ctx[var]
        return ctx['O'] if 'O' in ctx else last
    return program


def string_to_row(s: str) -> pd.Series:
    """Convert semicolon-separated expressions back to lists of variables, functions and arguments"""
    expressions = s.split(';')
    variables = []
    functions = []
    arguments = []
    
    for expr in expressions:
        var, rest = expr.split('=')
        func = rest[:rest.index('(')]
        args = rest[rest.index('(')+1:-1].split(',')
        args = [arg.strip() for arg in args]
        
        variables.append(var)
        functions.append(func)
        arguments.append(args)
        
    return pd.Series({
        'variable': variables,
        'function': functions,
        'arguments': arguments,
    })


Grid = list[list[int | str]] | tuple[tuple[int | str]]


def grid_to_string(grid: Grid | np.ndarray) -> str:
    return ';'.join([','.join(str(x) for x in row) for row in grid])


def string_to_grid(s: str) -> np.ndarray:
    return np.array([[int(x) for x in row.split(',')] for row in s.split(';')])


def grid_to_flat_tokens(grid: Grid, row_separator=';') -> list:
    return [str(x) for row in grid for x in tuple(row) + (row_separator,)]

