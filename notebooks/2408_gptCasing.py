"""Test casing.

The idea is to check whether we observe significantly different behavior based
on casing.
Also see [howard].

[howard]: https://x.com/jeremyphoward/status/1828880731961209094
"""
# %%
from tk.utils import tree as tkt
from tk.utils import pprint  as tkp
from collections import defaultdict
from rich import print as rprint

import tqdm
import tk
import openai
from openai.types.chat.chat_completion import ChatCompletion
import json

from pathlib import Path
from tk.utils import memo


with Path("~/.apikeys.json").expanduser().open() as f:
  key = json.load(f)['openai-self']
  client = openai.Client(api_key=key)
seed = 42

# NOTE this neatly lets you add new examples below without extra API calls
# don't forget it's CACHING RESPONSES tho.
# DON'T DO MONTE CARLO EXPERIMENTATION
query = memo(client.chat.completions.create)

# %%
import regex as re


def completion_by_id(xs: tkt.Tree[ChatCompletion], id: str):
    def f(x):
        if not ii(ChatCompletion)(x):
            return None
        return x if re.search(x.id, id) else None

    xs = tkt.map(f, xs, is_leaf=ii(ChatCompletion))
    xs, _ = tkt.flatten(xs)
    return tuple(filter(bool, xs))

# %%
import itertools as it
import numpy as np

from dataclasses import dataclass
from typing import Callable, NamedTuple
from jax import tree

# single-character __repr__s
# 🍉 vs 🍓 => have pits on the inside/out (ha, get it)

class In(NamedTuple):
  variant: str
  query: str
  gt: str
  prompt = property(
      lambda s: f"answer with yes/no in {s.variant}: {s.query}")
  __str__ = prompt.fget
  __repr__ = lambda s: f"🍉【{s.variant}•{s.query}】"

print(repr(In("a", "b", "c")))
# %%

class Out(NamedTuple):
  input: In
  response: ChatCompletion
  text = property(lambda s: s.response.choices[0].message.content)
  __str__ = text.fget
  __repr__ = lambda s: f"🍓【{s.input!r}•{s.text}】"

variants = [
    "all lowercase",
    "all uppercase",
    "all camelcase",
]

def run_experiment(qgts: list[dict], seed=seed) -> list[Out]:
    responses: list[Out] = []
    for (q, gt), v in tqdm.tqdm(it.product(qgts, variants)):
      input_ = In(variant=v, query=q, gt=gt)
      response = query(
        model="gpt-4o-mini",
        messages=[
            {"role": "system", "content": "you are a helpful assistant."},
            {"role": "user", "content": input_.prompt}
        ],
        temperature=1,
        seed=seed,
        logprobs=True,
        top_logprobs=5,
      )
      responses.append(Out(input_, response))
    return responses

questions = {
    "y": [
        "are there black swans?",
        "are there white swans?",
        "is brunch before lunch?",
        "is brunch after breakfast?",
    ],
    "?": [
        "are there gray swans?",
        "is blue better than yellow?",
        "are flortls more lucrative than wollys?",
        "is brunch before lunch in terms of taste?",
    ],
    "n": [
        "is lunch before brunch?",
        "on opposite day, is lunch before brunch?",
        "is the tallest building in the world over 1 lightyear high?",
        "is the smallest building in the world over 1km high?",
        "is the smallest building in the world over 1km high?",
    ]
}
responses = run_experiment(
    ((q, gt) for gt, qs in questions.items() for q in qs)
)
# %%
def stratify_yn(responses):
    str2r = defaultdict(list)
    for r in responses:
        k = r.response.choices[0].message.content.lower().strip(' .,!?')
        str2r[k].append(r)
    return str2r

str2r = stratify_yn(responses)
tkp.p({k: len(v) for k, v in str2r.items()})

# %%
class TopProb(NamedTuple):
  tok: str
  prob: float
  __str__ = lambda s: s.tok
  __repr__ = lambda s: f"{s.tok}•{s.prob:.2f}"

class OutTok(NamedTuple):
  ref: Out
  tps: tuple[TopProb]
  __repr__ = lambda s: f"【{s.ref!r}•{s.tps!r}】"


def as_toks(cc: ChatCompletion, k: int = 2):
  toks = cc.choices[0].logprobs.content[0].top_logprobs[:k]
  return tuple(TopProb(t.token, np.exp(t.logprob)) for t in toks)

def as_tps(o: Out) -> OutTok:
  return OutTok(o, as_toks(o.response))

# just manually inspect
ii = lambda *ty: lambda x: isinstance(x, ty)
xs = tree.map(as_tps, str2r, is_leaf=ii(Out))
# %%
vals, defn = tree.flatten(xs, is_leaf=ii(OutTok))
tkp.p(vals[:3])
# interestingly, variants of the following emerge:
# 1. ("yes", " yes", ...)  # note the space in 2nd resp
# 2. ("yes", "no", ...)
# %%
import plotly.express as px
import plotly.io as pio
from IPython.display import display_png


def semantically_differ(*tps: TopProb):
    """check if tokens semantically differ.

    our answers should be yes/no so don't overcomplicate.
    """
    xs = {t.tok.strip(' .?:!').lower() for t in tps}
    return len(xs) == len(tps)


def metric_uncertainty_top2(*tps: TopProb):
    if not semantically_differ(*tps):
        return 0.0
    probs = np.array([t.prob for t in tps])
    entropy = np.sum(-np.log(probs) * probs)
    return entropy


uncertain = (x for x in vals if semantically_differ(*x.tps))
sorteds = sorted(uncertain, key=lambda x: x.tps[0].prob)
to_print = []
for i, u in enumerate(sorteds):
    to_print.append((
        u, 
        metric_uncertainty_top2(*u.tps)))
tkp.p(to_print)
# %%
fig = px.bar(
    [x.ref.input.gt for x in sorteds],
    color=[x.tps[0].prob for x in sorteds],
    text=[x.tps[0].tok for x in sorteds],
    title="Uncertain (per top-two differing answers)"
)
fig.update_traces(textposition='inside', )
fig_img = pio.to_image(fig)
display_png(fig_img, raw=True)
# %% analysis says:
# idk, seems the ambiguous questions kinda depend on casing
# below: construct more ambiguous questions => test

qambig = {
    ("?", "meaningless"): (
        "are flortls more lucrative than wollys?",
        "is wolly bigger than boxer?",
    ),
    ("?", "no order"): (
        "are dogs better than cats?",
        "are cats better than dogs?",
        "are dogs better than dogs?",
        "are cats better than cats?",
        "is blue better than yellow?",
        "is yellow better than blue?",
        "is blue better than blue?",
        "is yellow better than yellow?",
    ),
    ("?", "no action"): (
        "did you let the dogs out?",
        "did Mark let the dogs out?",
        "did you go to Mallorca for the weekend?",
        "did Mark go to Mallorca for the weekend?",
    ),
    # ("?", "make reasonable assumption", "n"): (
    ("?", "rassum", "n"): (
        "was your vacation longer than 2 years?",
        "is your 4-year old son employed?",
        "is this your 1-year old's lemonade stand?",
        "did your 10-year old son vote for president?",
    ),
    # ("?", "make reasonable assumption", "y"): (
    ("?", "rassum", "y"): (
        "is this your 4-year old's lemonade stand?",
        "was your vacation shorter than 2 years?",
    ),
    # ("?", "will the LLM try to give nuanced answer"): (
    ("?", "nuanced"): (
        "should I do drugs?",
        "should I take mental health medication?",
        "should I take aspirin?",
        "should I jump off a bridge?",
        "should I put all my money into bitcoin?",
    )
}

responses_ambig = run_experiment(
    ((q, gt) for gt, qs in qambig.items() for q in qs)
)
# %%
str2r_ambig = stratify_yn(responses_ambig)
print({k:len(v) for k, v in str2r_ambig.items()})
# HA. I knew it will _have_ to be moralizing for some of them
# {'no': 49, 'yes': 21, 'unknown': 1, "i can't provide medical advice. please consult a healthcare professional": 1, 'yes/no': 2, 'i cannot provide medical advice. please consult a healthcare professional': 1}
# %%
print({k: v for k, v in str2r_ambig.items() if k not in ("yes", "no")})
# %%
import plotly.express as px
xs_ambig = tree.map(as_tps, str2r_ambig, is_leaf=ii(Out))
to_plot: list[OutTok]
kvs = [(k, x) for k, xs in xs_ambig.items() for x in xs if k in ('yes', 'no')]
to_plot = [x for k, x in kvs]
facets = [k for k, x in kvs]
fig = px.bar(
    ['|'.join(x.ref.input.gt[1:]) for x in to_plot],
    color=[x.tps[0].prob for x in to_plot],
    text=[x.ref.text for x in to_plot],
    title=f"Counts faceted by LLM answer",
    facet_row=[fr for fr in facets],
)
fig.update_traces(textposition='inside', )
# %%
