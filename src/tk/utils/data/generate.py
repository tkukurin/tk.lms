import random
import functools as ft
import itertools as it
import typing as ty

from torch.utils.data import DataLoader, Dataset


class OpData(Dataset):
    """Dataset to generate training examples for constrained envs.
    """

    def __init__(self, ops: list = [("+", sum)], n: int = 10, eq: str = "=", eof: str | None = None):
        """Generate operations based on sequential numbers.

        If eof is given then it will be appended to every generated value.
        NB, everything is kept in memory.
        We could optimize it by changing that.
        """
        self.data = []
        for numlist in it.product(range(n), range(n)):
            for opdata in ops:
                if isinstance(opdata, str):
                    strop = opdata
                else:
                    strop, op = opdata
                for numlist_perm in it.permutations(numlist):
                    strdata = strop.join(map(str, numlist_perm))
                    if isinstance(opdata, str):
                        result = eval(strdata)
                    else:
                        result = op(numlist_perm)
                    item = f"{strdata}{eq}{result}{eof or ''}"
                    if item not in self.data:
                        self.data.append(item)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, i: int):
        return self.data[i]


def ndigit_data(
    ndigits: int,
    n: int = 100,
) -> list[int]:
    """Generate datas with ndigits."""
    min_ = pow(10, ndigits - 1)
    max_ = pow(10, ndigits)
    out = random.sample(range(min_, max_), n)
    return out


def ndigit_data_to_tuples(
    *data: list[int],
    op: ty.Callable[[list[int]], int],
) -> list[tuple]:
    return [(elems, op(elems)) for elems in it.product(*data)]


def ndigit_tuples_format(
    tuples: list[tuple],
    op: str,
    eq: str = "=",
    fmt: str = "d",
    eof: str = "",
) -> list[str]:
    out = []
    for xs, y in tuples:
        lhs = op.join(f"{x:{fmt}}" for x in xs)
        out.append(f"{lhs}{eq}{y:{fmt}}{eof}")
    return out


def _eval(elems: list[int], op: str) -> str:
    op_eval = op.join(map(str, elems))
    return eval(op_eval)


def _eval_reverse(elems: list[int], op: str) -> str:
    op_eval = op.join(map(str, elems))
    return "".join(reversed(eval(op_eval)))


def strfmt_ops(
    *datas: list[int],
    op: str,
    eq: str = "=",
    eval_fn: ty.Callable  = _eval,
    eof: str = "",
    fmt: str = "d",
) -> list[str]:
    """Generate string versions of op data.
    
    Use e.g. 04d format to align all numbers to the same length.
    """
    out = ndigit_data_to_tuples(*datas, op=ft.partial(eval_fn, op=op))
    return ndigit_tuples_format(out, op, eq, fmt, eof)