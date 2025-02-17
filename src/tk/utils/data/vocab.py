"""Simple vocab generation (no tokenizer).
"""
# %%
import numpy as np
import dataclasses as dc
import itertools as it

from typing import Literal, NamedTuple, TypedDict


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

_TURTLE_GRAMMAR = """
    S -> <s> S0 I </s>
    S0 -> ( N , N ) C <instr>
    I -> I I
    I -> M
    M -> up N
    M -> down N
    M -> color C
    N -> 1
    N -> 2
    N -> 3
    C -> reset
    C -> red
    C -> blue
    C -> green
"""

# internally-referential grammar 
# TODO make as parametrizable or something
_TURTLE_GRAMMAR_INTERNAL_REF = """
S -> <s> S0 I </s>
S0 -> ( N , N ) C <instr>
# VARS -> N
I -> I1|I2|I3|I4
I1 -> ( 1 ) M
I2 -> ( 2 ) M M 
I3 -> ( 3 ) M M M
I4 -> ( 4 ) M M M M
# I -> M
M -> up|down N
M -> color C|reset
N -> 1|2|3
C -> red|blue|green
"""

from tk.utils import cfg as clib


class Batch(NamedTuple):
    input_ids: list[int]
    attention_mask: list[int]
    output_ids: list[int] | None | int


def mockgen_cfg(
    grammar: str,
    gen: np.random.Generator = np.random.default_rng(),
    seqlen: int | None = None
):
    cfg = clib.parse(grammar)
    voc = Voc.make({
        'terminals': list(cfg.terminals),
        # TODO this is specifically for turtle
        'nums': list('0123456789'),
        'special': ['<s>', '</s>', '<pad>', '<sep>'],
    })
    voc.limit = seqlen
    assert ('</s>' in voc and '<pad>' in voc), (
        f'expecting </s> and <pad>: {voc}'
    )
    pad = voc['<pad>']
    end = voc['</s>']
    sep = voc['<sep>']
    def nxt(with_output: Literal['concat', 'y', None] = None):
        nfail = 0
        while len(seq := clib.generate(cfg, gen)) > (seqlen or 99999):
            nfail += 1
            if nfail > 128:  # TODO specifically for turtle, use proper cfg
                raise ValueError(f"failed to generate ({nfail=})")
        seq_str = seq
        seq = [voc[x] for x in seq]
        closer = seq.index(end)
        seq = seq[:closer + 1]
        output = None
        if with_output:
            traces = turtle_interpret(seq_str)
            output = traces[-1][-1]
            output = [voc[str(x)] for x in output]
            if with_output in ('concat', ):
                seq = seq[:-1] + [sep] + output + [end]
                output = None  # we concat'd above already
            else:
                assert output in (True, )
        mask = [1] * len(seq) + (
            [0] * (seqlen - len(seq)) if seqlen else [])
        seq = seq + (
            [pad] * (seqlen - len(seq)) if seqlen else [])
        return Batch(seq, mask, output)

    return voc, nxt


def mockgen_turtle(
    gen: np.random.Generator = np.random.default_rng(),
    seqlen: int | None = None
):
    """Install Lark for this.

    [example]: https://github.com/lark-parser/lark/blob/master/examples/turtle_dsl.py
    """
    import lark
    turtle_grammar = """
        start: "<s>" instruction+ "</s>"

        instruction: MOVEMENT NUMBER            -> movement
                | "c" COLOR [COLOR]          -> change_color
                | "fill" code_block          -> fill
                | "repeat" NUMBER code_block -> repeat

        code_block: "{" instruction+ "}"

        MOVEMENT: "f"|"b"|"l"|"r"
        COLOR: LETTER+

        %import common.LETTER
        %import common.INT -> NUMBER
        %import common.WS
        %ignore WS
    """
    parser = lark.Lark(turtle_grammar)
    return parser


def mockgen_pyform():
    import pyformlang.cfg as clib
    cfg = clib.CFG.from_text("""
        S -> "<s>" NP VP "</s>"
        NP -> Det N
        VP -> V NP
        Det -> "the"
        N -> "cat"|"dog"
        V -> "chased"
    """)
    return cfg


def turtle_interpret(xs: list[str]):
    """clunky ad hoc state machine for turtle grammar

    defines some behavior we want to imagine the program implements.
    not overly thought through, just a sketch.
    """
    xs = xs[xs.index('<s>') + 1:xs.index('</s>')]
    # ( 1 , 2 ) red <instr>
    state = state0 = int(xs[1]), int(xs[3]), xs[5]
    xstate = None
    ops = ('up', 'down', 'left', 'right', 'color')
    # mod = max(int(x) for x in xs if x.isdigit()) + 1
    # random choice -> test when char is never in input
    mod = 8
    traces = []
    for i, x in enumerate(xs[xs.index('<instr>') + 1:]):
        if xstate in ops:
            a, b, c = state
            ud = xstate in ('up', 'down')
            lr = xstate in ('left', 'right')
            m = -1 if xstate in ('down', 'left') else 1
            p = (m * int(x)) if ud or lr else 0
            da, db = (p if ud else 0, p if lr else 0)
            if xstate in ('color', ):
                c = state0[-1] if x == 'reset' else x
            state = ((a + da) % mod, (b + db) % mod, c)
            traces.append((xstate, p, state))
            xstate = 'instr_or_end'
        elif xstate is None:
            assert x in ('(', ), f"{i=} {x=}, {xs=}"
            xstate = 'length_define'
        elif xstate in ('length_define', ):
            assert x.isdigit(), f"{i=} {x=}, {xs=}"
            xstate = 'length_close'
        elif xstate in ('length_close', ):
            assert x in (')', )
            xstate = 'length_closed'
        else:
            # NOTE(tk) I adhoc set this
            if x == '<sep>': break
            assert x in ops, f"{i=} {x=}, {xs=}"
            xstate = x
    return traces


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
        return Batch(seq, mask, None)
    return voc, nxt


if __name__ == '__main__':
    voc, nxt = mockgen_cfg(_TURTLE_GRAMMAR_INTERNAL_REF, seqlen=64)
    print(voc)
    toks, mask, outs, *_ = nxt('concat')
    tokstr = ([
        voc.inverse()[tok] for tok in toks
    ])
    print(tokstr)
    traces = turtle_interpret(tokstr)
    print('s0', traces[0])
    print('sN', traces[-1])
    print(traces)
    print(mask)
    print(outs)
    print(len(voc), voc.values())
    assert len(mask) == len(toks)
# %%
if __name__ == '__main__':
    for _ in range(50):
        voc, nxt = mockgen_cfg(_TURTLE_GRAMMAR_INTERNAL_REF, seqlen=64)
        print(voc)
        toks, mask, outs, *_ = nxt(True)
        tokstr = ([
            voc.inverse()[tok] for tok in toks
        ])
        print(tokstr)
        print(turtle_interpret(tokstr))
        print(mask)
        print(outs)
        assert len(mask) == len(toks)
# %%
# if __name__ == '__main__':
#     p = mockgen_turtle(seqlen=16)
#     print(p.rules)
#     print(p.grammar.rule_defs)
# %%
cfg = clib.parse(_TURTLE_GRAMMAR_INTERNAL_REF)
cfg.rules
# %%
