"""Demo for my LM-talk.

The code below should be runnable from a Colab.
Runs on my ThinkPad X270 as well => not too resource hungry.
"""
# %%
# !pip install -q llama-cpp-python

# %%
import tk
import tqdm

try: from tk import datadir
except: datadir = '.'

import re
import itertools as it

try: from rich import print as pp
except: from pprint import pprint as pp

import requests

url = 'https://huggingface.co/microsoft/Phi-3-mini-4k-instruct-gguf/resolve/main/Phi-3-mini-4k-instruct-q4.gguf?download=true'
if not (out := datadir/"phi3_fp4.gguf").exists():
    print(f"Not found: {out.absolute()}")
    if not (got := requests.get(url)).ok:
        print("[WARN] some error downloading")
    else:
        print(f"Saving to {out}")
        with open(out, "wb") as f:
            f.write(got.content)

# %%
import diskcache
from pathlib import Path


print(f"{tk.in_colab=}")
persistent = True # @param {type:"boolean"}
if tk.in_colab and persistent:
  if not (colab := Path('/gdrive/MyDrive/tmp')).exists():
    from google.colab import drive
    drive.mount('/gdrive')
    colab.mkdir(parents=True, exist_ok=True)
  CACHE = diskcache.Cache(directory=str(colab/'2406_talk'))
else:
  CACHE = diskcache.Cache(directory=datadir / 'cache')

memo = CACHE.memoize(name='2406_talk')
print(f"{CACHE.directory=}")


def get_cached_prompts():
  """Note that we return (prompt, answer) pairs from our gen_ methods.

  This is just a convenience to review past queries.
  """
  keys = list(CACHE.iterkeys())
  values = [CACHE.get(k) for k in keys]
  return keys, values

# %%
from llama_cpp import Llama

phi_raw = Llama(
  model_path=str(datadir / "phi3_fp4.gguf"),
  # NB, fp16 is too slow. I _assume_ weights are constantly swapped out with hdd
  # model_path="phi3_fp16.gguf",
  n_ctx=4096,
  n_threads=8,
  n_gpu_layers=0,
  # You can specify this or manually do it in code
  # (as we do, cf. chat_handler_phi)
  chat_format=None,
  # You can use phifmt here as well.
  # We set to `None` in order to be more explicit.
  chat_handler=None,
  # Useful to print infos about the model first time
  verbose=False,
  # make reproducible
  seed=42,
  logits_all=True,
)
# suppress outputting timings
phi_raw.verbose = False


# %%

# convert text to integers
print(tokens := phi_raw.tokenize(b"Hello world"))
# check how it reverts back to text
print(detokens := phi_raw.detokenize(tokens))
# ask the model to generate _one_ next step
print(out := next(phi_raw.generate(tokens)))
# check what the model generated
print(deouts := phi_raw.detokenize([out]))


def manual_sample(text: str, topk: int=1, maxn: int = 10):
  """Sample from the model.

  If topk is 1, we get a deterministic token out.
  """
  tokens = phi_raw.tokenize(text.encode("utf8"))
  for step in tqdm.trange(maxn):
    out = next(phi_raw.generate(tokens, top_k=topk))
    # The `end` token is defined during the model training procedure
    if phi_raw.detokenize([out]) == '<|end|>':
      break
    tokens = tokens + [out]
  return tokens, phi_raw.detokenize(tokens)


tokens_out, model_out = manual_sample("Hello world")
print()
print(tokens_out, '\n  ->', model_out)

# %%

class _Phifmt:
  """Wrap messages into Phi format (user/assistant tags).

  This is training-dependent; see [modelfile] and [phipaper] for reference.

  [modelfile]: https://huggingface.co/microsoft/Phi-3-mini-4k-instruct-gguf/blob/main/Modelfile_q4
  [phipaper]: https://arxiv.org/pdf/2404.14219
  """

  role_user = '<|user|>\n'
  role_asst = '<|assistant|>\n'
  role_ends = '<|end|>\n'

  special_toks = dict(
      user = role_user, asst = role_asst, end = role_ends)

  def __init__(self):
    pass  # in theory you could override role_xyz here for different models

  def str2chat(self, text: str, role: str = "user") -> str:
    if role in ('user', ):
      return f'{self.role_user}{text}{self.role_ends}'
    elif role in ('gpt', 'assistant'):
      return f'{self.role_asst}{text}{self.role_ends}'
    raise ValueError(f'unk role {role}')

  def list2chat(self, prompt: list[dict]) -> str:
    messages = [self.str2chat(**x) for x in prompt]
    return ''.join(messages)

  def __call__(self, prompt: list | str) -> str:
    """ensure appropriate chat format is used."""
    if isinstance(prompt, (list, tuple)):
      if isinstance(prompt[0], str):
        roles = it.cycle(['user', 'gpt'])
        prompt = [{'role': x, 'text': y} for x, y in zip(roles, prompt)]
      prompt = self.list2chat(prompt)
    elif not prompt.startswith('<|'):
      prompt = self.str2chat(prompt, role="user")
    # ensure we end with assistant.
    # not a huge deal as the LM would normally just fill this in
    # but in the best case, it just wastes some compute
    if not prompt.strip().endswith('<|assistant|>'):
      prompt = f'{prompt}{self.role_asst}'
    return prompt


phifmt = _Phifmt()
assert (x := phifmt('test')) == (
    '<|user|>\n'
    'test<|end|>\n'
    '<|assistant|>\n'
), f"{x}"
assert (x := phifmt([
  {'role': 'gpt', 'text': 'test1'},
  {'role': 'user', 'text': 'test2'},
])) == (
    '<|assistant|>\n'
    'test1<|end|>\n'
    '<|user|>\n'
    'test2<|end|>\n'
    '<|assistant|>\n'
  ), f"{x}"
assert (x := phifmt(['a', 'b'])) == (
  '<|user|>\na<|end|>\n<|assistant|>\nb<|end|>\n<|assistant|>\n'
), f"{x}"

# %%

def gen(prompt: str, *, seed=42, **kw) -> tuple[str, str]:
  """Generate a completion prompt (no chat tokens!)
  """
  defaults = dict(
    max_tokens=256,
    seed=seed,
    stop=['<|end|>'],
    echo=False,
  )
  output = phi_raw(prompt, **{**defaults, **kw})
  return prompt, output['choices'][0]['text']

# %%

prompt_bowie_chat = phifmt([
  {'role': 'user', 'text': 'This is ground control to'},
  {'role': 'gpt', 'text': 'major Tom.'},
  {'role': 'user', 'text': 'You\'ve really'},
])
prompt_bowie_chat_question = phifmt([
  {'role': 'user', 'text': 'Please complete David Bowie lyrics: This is ground control'},
])
prompt_bowie_nonchat = "This is ground control to"

# %%
from textwrap import TextWrapper


def h1(text: str):
    wrapper = TextWrapper(width=80, break_long_words=True, replace_whitespace=False)
    wrapped_text = wrapper.wrap(text)
    print("=" * 80)
    for line in wrapped_text:
        print(line)
    print("=" * 80)


# %%
# chat: note a conversational continuation
p, o = memo(gen)(prompt_bowie_chat, max_tokens=16)
h1(p)
h1(o)
# %%
p, o = memo(gen)(prompt_bowie_chat_question, max_tokens=16)
h1(p)
h1(o)
# %%
p, o = memo(gen)(prompt_bowie_chat_question, top_k=1, max_tokens=16)
h1(p)
h1(o)
# %%
p, o = memo(gen)(prompt_bowie_nonchat, max_tokens=16)
h1(p)
h1(o)

# %%

# NOTE set to 10 for speed, feel fre to increase to e.g. 100 later
ntrials = 10 # @param {type:"slider", min:0, max:100, step:1}
prompt = phifmt(
  'Generate a random number between 1 and 10. '
  'Do not remind me you are a language model. '
  'Output a single number.'
)

randn_responses = []
for _ in tqdm.trange(ntrials):
  # NOTE: ensure seed is None!
  # remember that we set to 42 in "defaults"
  p, o = gen(prompt, seed=None)
  randn_responses.append(o)

pp("First few responses:")
pp(randn_responses[:5])
# %%
import pandas as pd
from collections import defaultdict


def group_cases(responses: list[str]):
    """Extract the useful LM responses.

    The "unambiguous" cases are assumed to have 1 digit in the full text.
    The "ambiguous" cases might have multiple digits.
    """

    def numextract(response: str):
        outs = re.findall(r'(\d+)', response, re.I)
        return list(outs)

    cases = defaultdict(list)
    for i, r in enumerate(responses):
        extracts = numextract(r)
        cases[len(extracts)].append((i, extracts))
    return cases


cases = group_cases(randn_responses)
xs = [y for x, y in cases[1]]
pd.DataFrame(xs).value_counts()
# %% [markdown]
# Intrinsic LLM evaluation:
# We assume text is distributed according to a unk. true distribution $p$.
# The model learned a distribution $q$ on an existing train set.
# Luckily, Shannon already figured out how to deal with this conundrum.
# BUT:
# _What even is a test set_?!
# %%
pledges = {
  # cf. https://en.wikipedia.org/wiki/Pledge_of_Allegiance
  "1892":
  "I pledge allegiance to my Flag and the Republic for which it stands, one "
  "nation, indivisible, with liberty and justice for all.",
  
  "1892-1923":
  "I pledge allegiance to my Flag and to the Republic for which it stands, one "
  "nation, indivisible, with liberty and justice for all.",

  "1923-1924":
  "I pledge allegiance to the Flag of the United States and to the Republic for "
  "which it stands, one nation, indivisible, with liberty and justice for all.",

  "1924-1954":
  "I pledge allegiance to the Flag of the United States of America and to the "
  "Republic for which it stands, one nation, indivisible, with liberty and "
  "justice for all.",

  "1954":
  "I pledge allegiance to the Flag of the United States of America, and to the "
  "Republic for which it stands, one nation under God, indivisible, with liberty "
  "and justice for all.",
}

completion_pledge = memo(phi_raw.create_completion)("I pledge allegiance", logprobs=5)
print((c_pledge := completion_pledge['choices'][0])['text'])
# %%
import pandas as pd
from IPython.display import display as show

show(df := pd.DataFrame(c_pledge['logprobs']['top_logprobs']))
probs = df.max(axis=1)
words = df.idxmax(axis=1)
# %%
pledge = pledges["1954"]
start = 0
for w in words:
  found = pledge.lower().find(w.lower(), start)
  if found > -1:
     start = found
  print(found)

# %%
import numpy as np
show(np.exp(probs))
# %% [markdown]
# # Datasets
# A sample of some common datasets you'll see mentioned in LM evals
# %%
import datasets
pp(ds := datasets.load_dataset('TIGER-Lab/MMLU-Pro'))
pp(ds['validation'][0])
# %%
pp(ds_swag := datasets.load_dataset('Rowan/hellaswag'))
pp(ds_swag['validation'][0])
# %%
pp(ds_bb_causal := datasets.load_dataset(
    'lighteval/big_bench_hard', 'causal_judgement', split='train'))
pp(ds_bb_logic3 := datasets.load_dataset(
    'lighteval/big_bench_hard', 'logical_deduction_three_objects', split='train'))
# %%
pp(ds_commonsense := datasets.load_dataset('chiayewken/commonsense-qa-2', split='train'))
pp(ds_commonsense[0])
# %%
pp(ds_commonsense := datasets.load_dataset('chiayewken/commonsense-qa-2', split='train'))
pp(ds_commonsense[0])
# %%
pp(ds_gsm8k := datasets.load_dataset('openai/gsm8k', 'main', split='train'))
pp(ds_gsm8k[0])
# %%