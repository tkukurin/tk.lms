"""Test casing.

The idea is to check whether we observe significantly different behavior based
on casing.
Also see [howard].

[howard]: https://x.com/jeremyphoward/status/1828880731961209094
"""
# %%
from collections import defaultdict
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
query = memo(client.chat.completions.create)
# %%
import itertools as it

variants = [
    "all lowercase",
    "all uppercase",
    "all camelcase",
]
questions = [
    "are there black swans?",
    "are there white swans?",
    "are there gray swans?",
    "is brunch before lunch?",
    "is lunch before brunch?",
]
responses: list[ChatCompletion] = []
for question, variant in it.product(questions, variants):
    response = query(
      model="gpt-4o-mini",
      messages=[
            {"role": "system", "content": "you are a helpful assistant."},
            {"role": "user", "content": f"answer with yes/no in {variant}: {question}"}
        ],
      temperature=1,
      seed=seed,
      logprobs=True,
      top_logprobs=5,
    )
    responses.append(response)

# %%
str2r = defaultdict(list)
for r in responses:
    k = r.choices[0].message.content.lower().strip(' .,!?')
    str2r[k].append(r)
print({k: len(v) for k, v in str2r.items()})
# %%
from jax import tree
import numpy as np

def as_toks(r: ChatCompletion, k: int = 2):
    lps = r.choices[0].logprobs.content[0].top_logprobs[:k]
    return [(lp.token, np.exp(lp.logprob)) for lp in lps]

def fmt(tpl: tuple):
    t, p = tpl
    return (t, int(p * 10000) / 100)

# just manually inspect
leafn = lambda x: isinstance(x, tuple)
xs = tree.map(as_toks, str2r)
xs = tree.map(fmt, xs, is_leaf=leafn)
# %%
vals, defn = tree.flatten(xs, is_leaf=lambda x: isinstance(x, list) and leafn(x[0]))
# %%
print(vals)
# %%
