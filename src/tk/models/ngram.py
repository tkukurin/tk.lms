"""Some basic ngram-based generative transition model.
"""
from __future__ import annotations

import numpy as np
import itertools as it
from collections import Counter, defaultdict


def to_transition_row(
    dict_entry: dict[int, int],
    stoi: dict[str, int],
    vocab_size: int,
    smooth: int = 0,
):
    row = [smooth for _ in range(vocab_size)]
    for dst, count in dict_entry.items():
        row[stoi[dst]] = count + smooth
    row = np.array(row)
    if (sumrow := np.sum(row)) == 0:  # uniform
        return np.ones_like(row) / len(row)
    return row / sumrow


class Tokenizer:
    """Kinda inefficient ngram collector.

    Use cls#init to initialize from a raw text.
    """

    def __init__(self, transitions: dict[str, dict[str, int]],):
        self.totals = {
            src: sum(dst_count.values())
            for src, dst_count in transitions.items()
        }
        self.transitions = transitions

        self.vocab = defaultdict(int)
        for dst, count in it.chain(*(x.items() for x in self.transitions.values())):
            self.vocab[dst] += count

        self.stoi = {c: i for i, c in enumerate(self.vocab)}
        self.itos = {i: c for c, i in self.stoi.items()}
        #assert all(len(k) == 1 for k in self.vocab)
        #assert all(len(v) == 1 for v in self.stoi)

    @classmethod
    def init(
        cls,
        text: str,
        mincount: int = 2,
        maxn_or_tuples: int | tuple = 2,
    ) -> Tokenizer:
        ctr = Counter()
        ns = (
            maxn_or_tuples 
            if hasattr(maxn_or_tuples, "__len__")
            else range(1, maxn_or_tuples + 1)
        )
        for n in ns:
            idxs = range(0, len(text) - n)
            text_chunks = [tuple(text[i:i+n]) for i in idxs]
            continuations = (nxt for *_, nxt in text_chunks[1:])
            ctr_cur = Counter(zip(text_chunks, continuations))
            ctr += ctr_cur
        pruned_counts = defaultdict(dict)
        for (fst, snd), count in ctr.items():
            if count >= mincount:
                pruned_counts[fst][snd] = count
        return cls(pruned_counts)


def generate(
    self: Tokenizer, 
    start: str | None = None, 
    max_tokens: int = 64, 
    smooth = 0
):
    def _random() -> str:
        row = to_transition_row(
            self.vocab,
            smooth=smooth,
            stoi=self.stoi,
            vocab_size=len(self.vocab)
        )
        choice = np.random.multinomial(1, row).argmax(0)
        return self.itos[choice]

    outputs = [x for x in (start or _random())]
    maxn = max(map(len, self.transitions))
    for _ in range(max_tokens):
        lookback = min(len(outputs), maxn) + 1
        lasts = (tuple(outputs[-i:]) for i in range(1, lookback))
        lasts_with_transitions = [l for l in lasts if l in self.transitions]
        src = np.random.choice(len(lasts_with_transitions))
        src = lasts_with_transitions[src]
        row = to_transition_row(
            self.transitions[src],
            smooth=smooth,
            stoi=self.stoi,
            vocab_size=len(self.vocab)
        )
        choice = np.random.multinomial(1, row).argmax(0)
        outputs.append(self.itos[choice])

    return outputs, {}
    # return ''.join(outputs), {}
