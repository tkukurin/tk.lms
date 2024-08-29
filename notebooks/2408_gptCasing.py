"""Test casing.

The idea is to check whether we observe significantly different behavior based
on casing.
Also see [howard].

[howard]: https://x.com/jeremyphoward/status/1828880731961209094
"""
# %%
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
import itertools as it
import numpy as np

from dataclasses import dataclass
from typing import Callable, NamedTuple
from jax import tree


class In(NamedTuple):
  variant: str
  query: str
  gt: str
  prompt = property(
      lambda s: f"answer with yes/no in {s.variant}: {s.query}")
  __str__ = prompt.fget


class Out(NamedTuple):
  input: In
  response: ChatCompletion
  text = property(lambda s: s.response.choices[0].message.content)
  __str__ = text.fget
  __repr__ = lambda s: f"Out({s.input}|{s.text})"

variants = [
    "all lowercase",
    "all uppercase",
    "all camelcase",
]
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
kvs = ((q, gt) for gt, qs in questions.items() for q in qs)
responses: list[Out] = []
for (q, gt), variant in tqdm.tqdm(it.product(kvs, variants)):
  input_ = In(variant=variant, query=q, gt=gt)
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

# %%
str2r = defaultdict(list)
for r in responses:
  k = r.response.choices[0].message.content.lower().strip(' .,!?')
  str2r[k].append(r)
print({k: len(v) for k, v in str2r.items()})
# %%
print(str2r['yes'][0])
# %%
class TopProb(NamedTuple):
  token: str
  prob: float
  __str__ = lambda s: s.token
  __repr__ = lambda s: f"tp({s.token}|{s.prob:.2f})"

class OutTok(NamedTuple):
  ref: Out
  tps: list[TopProb]

def as_toks(cc: ChatCompletion, k: int = 2):
  toks = cc.choices[0].logprobs.content[0].top_logprobs[:k]
  return [TopProb(t.token, np.exp(t.logprob)) for t in toks]

def as_tps(o: Out) -> OutTok:
  return OutTok(o, as_toks(o.response))

# just manually inspect
ii = lambda *ty: lambda x: isinstance(x, ty)
xs = tree.map(as_tps, str2r, is_leaf=ii(Out))
# %%
vals, defn = tree.flatten(xs, is_leaf=ii(OutTok))
# %%
rprint(vals[:3])
# interestingly, variants of the following emerge:
# 1. ("yes", " yes", ...)  # note the space in 2nd resp
# 2. ("yes", "no", ...)
# %%
vals: list[OutTok]
uncertain = [
    x
    for x in vals
    if str(x.tps[0].token).strip().lower()
    != str(x.tps[1].token).strip().lower()
]
# %%
import plotly.express as px
sorteds = sorted(uncertain, key=lambda x: x.tps[0].prob)
px.bar(
    [x.ref.input.gt for x in sorteds],
    color=[x.tps[0].prob for x in sorteds],
    title="Uncertain (per top-two differing answers)"
)
# %% analysis says:
# idk, seems the ambiguous questions kinda depend on casing
# TODO: construct more ambiguous questions => test
