"""Demo for my LLMs talk: metrics.

Shows: metric failure modes, common datasets.
Implementation of perplexity.
"""

# %% [markdown]
# # Metrics
#
# ## Example: BLEU
# 
# reference: Attention is All you need: 
# * bleu(EN-to-DE)=0.284, 
# * bleu(EN-to-FR)=0.410

import evaluate
bleu = evaluate.load('bleu')

# %%
pred = "The quick brown fox jumps over the lazy dog."
# %%
try: from rich import print as pp
except: from pprint import pprint as pp

for ref in (
    'The fast brown fox leaps over the lazy dog',
    'The fast brown fox leaps over the lazy dog.',
    'The quick brown fox leaps over the lazy dog',
    'The quick brown fox jumps over the lazy dog',
    'The quick brown fox jumps over the lazy dog.',
):
    print()
    pp(ref)

    pp(bleu.compute(
        predictions=[pred],
        references=[ref],
    ))

# %%
pp(pred)

for ref in (
    'A swift brown animal leaps over a sleeping canine.',
    'The swift brown animal leaps over a sleeping canine.',
    'The quick brown animal leaps over a sleeping canine.',
    'The quick brown animal jumps over a sleeping canine.',
    'The quick brown animal jumps over a lazy canine.',
    'The quick brown animal jumps over a lazy dog.',
    'The quick brown fox leaps over a sleeping canine.',
):
    print()
    pp(ref)

    pp(bleu.compute(
        predictions=[pred],
        references=[ref],
    ))

# %% [markdown]
# ## Example: Perplexity
# What does this say about the training data?

# %%
ppx = evaluate.load('perplexity')

# %%
pp(ppx.compute(
    model_id='gpt2',
    predictions=[
      "I pledge allegiance to the flag",
    ],
    add_start_token=True,
))

# %%
pp(ppx.compute(
    model_id='gpt2',
    predictions=[
      "Ground control to Major Tom",
    ],
    add_start_token=True,
))

# %% [markdown]
# ## Example: Models-evaluating-models
# Ambiguity: which dimension of similarity do you care about?
# How to normalize numbers?

# %% 
import os
import requests

import diskcache
from pathlib import Path



def raw_get(payload: dict, api_key: str):
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {api_key}"
    }
    return requests.post(
        "https://api.openai.com/v1/chat/completions",
        headers=headers, json=payload)


if apikey := os.environ.get('OPENAI_API_KEY'):
    cache = diskcache.Cache(Path("~/cache.bootcamp").expanduser())
    memo_gpt = cache.memoize(tag='raw_oai')(lambda **data: raw_get(data, apikey))

    def gptsim(a, b):
        prompt = f"""How similar are the two sentences?
        A: {a}
        B: {b}
        Please just output a single numeric score in range 0-100.
        """.strip()
        response = memo_gpt(
            model="gpt-4-turbo",
            messages=[
                {
                    "role": "user",
                    "content": [{"type": "text", "text": prompt}],
                }
            ],
            max_tokens=35,
            seed=42,
        )
        text = response['choices'][0]['message']['content']
        return text

    pp(gptsim(
        "A quick brown fox jumps over the lazy dog", 
        "The quick brown fox jumps over the lazy dog",
    ))

    pp(gptsim(
        "A quick brown fox jumps over the lazy dog", 
        "The lethargic brown fox leaps over the sleepy dog",
    ))
else:
    print("Set environment: OPENAI_API_KEY")

# %% [markdown]
# # Datasets
# Knowledge-vs-"reasoning".
# Train-vs-test leakage.
# What is "test" when your input is _everything_.

# %%
import datasets
import contextlib


@contextlib.contextmanager
def mylog(new_log=lambda *a, **kw: None):
    # "poor man's disable logging"
    global pp
    pp_old = pp
    pp = new_log
    yield
    pp = pp_old


def h1(*s: str):
    sep = '-' * max(map(len, s))
    s = '\n'.join(s)
    pp(f'{s}\n{sep}')


with mylog():
    h1('', 'mmlu')
    pp(ds_mmlu := datasets.load_dataset('TIGER-Lab/MMLU-Pro'))
    pp(ds_mmlu['validation'][0])

    h1('', 'bigbench')
    pp(ds_bb_causal := datasets.load_dataset(
      'lighteval/big_bench_hard', 'causal_judgement', split='train', 
      trust_remote_code=True))
    pp(ds_bb_logic3 := datasets.load_dataset(
      'lighteval/big_bench_hard', 'logical_deduction_three_objects', split='train',
      trust_remote_code=True))

    print("Done")

# %%
pp(ds_mmlu['validation'][0])

# %%
pp(ds_bb_causal[0])

# %%
pp(ds_bb_logic3[0])

# %% [markdown]
# # Manual implementation: perplexity

# %%
import transformers
import torch

device = "cpu"

tokenizer = transformers.AutoTokenizer.from_pretrained("gpt2")
tokenizer.pad_token = tokenizer.eos_token

model = transformers.GPT2LMHeadModel.from_pretrained(
    "gpt2", 
    pad_token_id=tokenizer.eos_token_id
).to(device)
model.eval()

print('\nINPUT ::')
pp((tokenized := tokenizer("StackOverflow", return_tensors="pt")).input_ids)
print('\nMODEL GENERATION ::')
pp((out_tokens := model.generate(**tokenized.to(model.device), max_new_tokens=20)))
print('\nDECODED ::')
pp(tokenizer.decode(out_tokens[0]))

# %%

@torch.no_grad
def perplexity(
    predictions,
    add_start_token=True,
):
    encodings = tokenizer(
        predictions,
        add_special_tokens=False,
        padding=True,
        truncation=False,
        return_tensors="pt",
        return_attention_mask=True,
    ).to(model.device)

    iids = encodings["input_ids"]
    attn_mask = encodings["attention_mask"]

    if add_start_token:
        bos_tokens_tensor = torch.tensor(
            [[tokenizer.bos_token_id]] * iids.size(dim=0)).to(device)
        iids = torch.cat([bos_tokens_tensor, iids], dim=1)
        attn_mask = torch.cat(
            [torch.ones(bos_tokens_tensor.size(), dtype=torch.int64).to(device), attn_mask], dim=1
        )

    out_logits = model(iids, attention_mask=attn_mask).logits
    shift_logits = out_logits[..., :-1, :].contiguous()
    shift_labels = iids[..., 1:].contiguous()
    shift_attention_mask_batch = attn_mask[..., 1:].contiguous()

    def ce(pred: torch.Tensor, tgt: torch.Tensor):
        """
        -sum(tgt(x) * log(pred(x)))
        """
        log_softmax = torch.log_softmax(pred, -1)
        onehot = torch.nn.functional.one_hot(tgt, pred.shape[-1])
        return -torch.sum(onehot * log_softmax, dim=-1)

    # e ** ( -sum(p(x) * log(q(x))) / number_of_tokens )
    ppx = torch.exp(
        (ce(shift_logits, shift_labels) * shift_attention_mask_batch).sum(1)
        / shift_attention_mask_batch.sum(1)
    )

    return ppx.item()

# %%
perplexity(
    predictions=[
      "I pledge allegiance to the flag of the United States of America",
    ],
    add_start_token=True,
)

# %%
perplexity(
    predictions=[
      "Hey Jude, don't make it bad",
    ],
    add_start_token=True,
)

# %%
perplexity(
    predictions=[
      "Hast du etwas Zeit für mich? Dann singe ich ein Lied für dich",
    ],
    add_start_token=True,
)

# %%
perplexity(
    predictions=[
      "Ground control to Major Tom",
    ],
    add_start_token=True,
)

# %%