"""Simple vocab generation (no tokenizer).
"""
# %%
import numpy as np
import dataclasses as dc
import itertools as it

from typing import TypedDict


class TokTypes(TypedDict):
    starts: list[int]
    stops: list[int]
    muls: list[int]
    nums: list[int]
    chrs: list[int]


VOC_NUM_CHR = dict(
    nums = tuple('0123456789'),
    chrs = tuple('abcdefghijklmnopqrstuvwxyz'),
)


@dc.dataclass
class Voc:
    kind: dict = dc.field(repr=False)
    t2i: dict = dc.field(repr=True)
    __getitem__ = lambda s, key: s.t2i[key]
    __len__ = lambda s: len(s.t2i)
    __contains__ = lambda s, key: key in s.t2i
    get = lambda s, key: s.t2i.get(key)
    keys = lambda s: s.t2i.keys()
    values = lambda s: s.t2i.values()
    __iter__ = lambda s: iter(s.t2i)
    ofkind = lambda s, kind: s.kind[kind]
    inverse = lambda s: Voc(
        {k: [s.t2i[v] for v in vs] for k, vs in s.kind.items()},
        {v: k for k, v in s.t2i.items()})
    @classmethod
    def make(cls, kind2tok: dict[str, list[str]]):
        uniqs = set(it.chain(*kind2tok.values()))
        t2i = {v: i for i, v in enumerate(uniqs)}
        return cls(kind2tok, t2i)

# %%

_TEST_GRAMMAR = """
    S -> <s> NP VP </s>
    NP -> Det N
    VP -> V NP
    Det -> the
    N -> cat
    N -> dog
    V -> chased
    """

from tk.utils import cfg as clib

def mockgen_cfg(
    grammar: str,
    gen: np.random.Generator = np.random.default_rng(),
    seqlen: int | None = None
):
    cfg = clib.parse(grammar)
    voc = Voc.make({
        'terminals': cfg.terminals,
        'special': ['<s>', '</s>', '<pad>'],
    })
    assert ('</s>' in voc and '<pad>' in voc), (
        f'expecting </s> and <pad>: {voc}'
    )
    pad = voc['<pad>']
    end = voc['</s>']
    def nxt():
        seq = clib.generate(cfg, gen)
        seq = [voc[x] for x in seq]
        closer = seq.index(end)
        seq = seq[:closer + 1]
        pads = [pad] * (seqlen - len(seq)) if seqlen else []
        seq = seq + pads
        mask = [1] * (closer + 1) + [0] * (len(seq) - (closer + 1))
        return seq, mask
    return voc, nxt


def mockgen(
    gen: np.random.Generator = np.random.default_rng(),
    seqlen: int | None = None
):
    voc = Voc.make({
        '00': ['<s>', ],
        '10': ['hello', ],
        '11': ['dear', 'world', 'cruel' ],
        '12': ['world', '</s>' ],
        '20': ['</s>'],
        'misc': ['<pad>',],
    })
    pad = voc['<pad>']
    end = voc['</s>']
    def nxt():
        seq = ['00', '10', '11', '12', '20']
        seq = [voc.ofkind(x) for x in seq]
        seq = [gen.choice(xs) for xs in seq]
        seq = [voc[x] for x in seq]
        closer = seq.index(end)
        seq = seq[:closer + 1]
        pads = [pad] * (seqlen - len(seq)) if seqlen else []
        seq = seq + pads
        mask = [1] * (closer + 1) + [0] * (len(seq) - (closer + 1))
        return seq, mask
    return voc, nxt


if __name__ == '__main__':
    voc, nxt = mockgen_cfg(_TEST_GRAMMAR, seqlen=10)
    print(voc)
    toks, mask = nxt()
    print([
        voc.inverse()[tok] for tok in toks
    ])
    print(mask)
    assert len(mask) == len(toks)
# %%
