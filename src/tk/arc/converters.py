"""Various conversion conveniences.
"""
from collections import Counter
import itertools as it
from pathlib import Path

import pandas as pd
import numpy as np
from loguru import logger

from typing import Callable, Literal

def split_stored_df(df: pd.DataFrame | str | Path) -> tuple[
    pd.DataFrame, pd.DataFrame, pd.DataFrame, dict
]:
    """process data from p3dsl_to_df.py into useful dataframes and vocab
    """
    if isinstance(df, str | Path):
        df = pd.read_pickle(df)
    assert isinstance(df, pd.DataFrame)
    df_io = df[['input', 'output']].drop_duplicates()
    df = df.drop(columns=['input', 'output'])
    # claude also offers an alternative
    # df_filtered = df.loc[df.index.get_level_values('example_id') == 0]
    # df_filtered = df_filtered.droplevel('example_id')
    # this is probably more convenient
    df_grouped = (
        df
        # keep only where example_id is 0 (all programs should be the same)
        .xs(0, level='example_id', drop_level=True)
        .groupby(level='id')
        .agg({'function': list, 'arguments': list, 'variable': list })
    )  # type: ignore

    import inspect
    from tk.arc.p3 import dsl
    primitive2fn = {
        name: call for name, call in inspect.getmembers(dsl, inspect.isfunction)
    }

    vocab = set(
        set(df['function'].unique())
        | set(str(x) for x in df['variable'].unique() if not str(x).startswith('x'))
        | set(str(x) for x in sum(df['arguments'], []) if not str(x).startswith('x'))
        | set(primitive2fn.keys())
        | {'I', 'O', 'x', '(', ')', ';', '=', ',', ' ', }
        | set('1234567890')
        | {'def', 'solve', 'return'}
        | {'<pad>', '<sep>'}
    )
    vocab = {v: k for k, v in enumerate(vocab)}

    return df, df_io, df_grouped, vocab

class SimpleArcGridSeqEncoder:
    """Convert ARC problems to sequences of tokens.

    See p3dsl_to_df.py for how to generate dataframes.
    """

    def __init__(
        self, 
        tok2id: dict[str, int],
        df_io: pd.DataFrame,
        df_grouped: pd.DataFrame,
    ):
        self.df_io = df_io
        self.df_grouped = df_grouped
        self.tok2id = tok2id
        assert len(set(tok2id.values())) == len(tok2id), (
            f"Duplicate values in vocab: {[(k, v) for k, v in Counter(tok2id.values()).most_common() if v > 1]}"
        )

    def _encode_problem(self, id: str) -> tuple:
        problems = self.df_io.loc[id].reset_index()
        inputs = [
            grid_to_flat_tokens(p) + ['<sep>'] for p in problems['input']]
        outputs = [
            grid_to_flat_tokens(p) + ['<sep>'] for p in problems['output']]
        row = self.df_grouped.loc[id]
        program = row_to_tokens(row)
        return inputs, outputs, program

    def encode_problem(self, id: str, max_length: int = 99999) -> tuple:
        """Encode a problem into a list of tokens"""
        inputs_all, outputs_all, program = self._encode_problem(id)
        inputs = [[self.tok2id[i] for i in inp] for inp in inputs_all]
        outputs = [[self.tok2id[i] for i in out] for out in outputs_all]
        program = [self.tok2id[i] for i in program]
        assert len(program) <= max_length, f"Invalid example {id}"
        assert len(inputs) == len(outputs), f"Invalid example {id}, {len(inputs)} != {len(outputs)}"
        flat = lambda xss: [x for xs in xss for x in xs]
        tokens = flat(inputs) + flat(outputs) + program
        i = len(inputs) - 1
        skipped = {'in': [], 'out': []}
        while len(tokens) > max_length:
            skipped['in'].append(inputs[i])
            skipped['out'].append(outputs[i])
            inputs, outputs = inputs[:i], outputs[:i]
            tokens = flat(inputs) + flat(outputs) + program
            i -= 1
        if i == -1:
            logger.warning(
                f"Problem {id} is too long for {max_length}"
                f" ({len(program)} program toks, {len(inputs_all)} examples)"
            )
            return None, skipped
        return tokens, skipped

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
        for multiix, row in self.df_io.iterrows():
            i, *_ = multiix  # type: ignore (not sure why complaining)
            tokens, skipped = self.encode_problem(i, max_length)
            if not tokens: continue
            if skipped: truncated[i] = skipped
            length_hist[len(tokens)] = length_hist.get(len(tokens), 0) + 1
            encoded[i] = tokens
        skipped = {}
        max_length = max(length_hist.keys())
        if quantile is not None:
            qs: pd.Categorical = pd.qcut(list(length_hist.keys()), q=(0, quantile, 1.0))
            skipped = {k: v for k, v in encoded.items() if len(v) >= qs[-1].left}
            encoded = {k: v for k, v in encoded.items() if len(v) <  qs[-1].left}
            max_length = int(qs[-1].left + 1)
        encoded_padded = {
            k: tokens + [self.tok2id['<pad>']] * (max_length - len(tokens)) 
            for k, tokens in encoded.items()
        }
        meta = dict(
            skipped_full=skipped, 
            truncated=truncated, 
            hist=length_hist)
        return encoded_padded, meta

    @classmethod
    def outdir(cls):
        from tk import datadir
        return datadir / 'mhodel_rearc'
    
    def save(self, encoded: dict):
        """save encoded data to a consistent path.
        """
        import pandas as pd
        outdir = self.outdir()
        outdir.mkdir(exist_ok=True, parents=True)
        pd.DataFrame(encoded).T.to_parquet(
            outdir / 'paddedFilteredTrain.parquet')
        pd.DataFrame(
            list(self.tok2id.items()), 
            columns=['token', 'id'], 
            dtype=('str', 'int')
        ).to_parquet(outdir / 'vocab.parquet')
        return outdir

    @classmethod
    def load(cls, fmt: Literal['hfd', 'raw'] = 'raw'):
        outdir = cls.outdir()
        df = pd.read_parquet(outdir / 'paddedFilteredTrain.parquet')
        vocab = pd.read_parquet(outdir / 'vocab.parquet')
        vocab['id'] = vocab['id'].astype(int)
        vdict = vocab.to_dict('list')
        vocab = dict(zip(vdict['token'], vdict['id']))
        if fmt in ('hfd', 'huggingface', 'hf'):
            from datasets import Dataset
            dataset = Dataset.from_dict({
                'input_ids': df.values,
                'attention_mask': df.values != vocab['<pad>'],
            })
            dataset.set_format(type='jax', columns=['input_ids', 'attention_mask'])
            return dataset, vocab
        elif fmt in ('raw', ):
            return df, vocab
        raise ValueError(f"Unknown format {fmt}")


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
        tokens.extend(map(str, (*splitter(var), '=', func, '(', *args, ')')))
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

