"""Quick processing to generate artefacts for [train].


[train]: train_hf.py
"""
# %%
import tk
from datasets import load_dataset
from transformers import AutoTokenizer
import tokenizers.implementations as toklib
from pathlib import Path
from transformers import AutoConfig

import csv
import tk
import numpy as np
import itertools as it
from typing import Iterable
import numpy as np
import jax.numpy as jnp
from torch.utils.data import DataLoader
from loguru import logger
import transformers


def make_vocab(texts: Iterable[str],):
    vocab = {'<pad>': 0, '<eof>': 1}
    for text in texts:
        for char in text:
            if char not in vocab:
                vocab[char] = len(vocab)
    return vocab


max_len = 16
nums = np.arange(10)
ops = [a + b for a, b in it.product(nums, nums)]
ops_fmt = [f"{a}+{b}={a+b}" for a, b in it.product(nums, nums)]
vocab = make_vocab(ops_fmt)

ops_tok = np.array(
    [
        [vocab[c]
         for c in op] +
        [vocab['<pad>']
         for _ in range(max_len - len(op))]
        for op in ops_fmt
    ]
)


def split(kind: str = 'one_doubledigit'):
    singles = {i for i, x in enumerate(ops) if x < 10}
    doubles = {i for i, x in enumerate(ops) if x >= 10}
    traini = singles | {i for i in doubles if ops[i] % 10 == 1}
    testi = set(list(range(len(ops)))) - traini
    return kind, traini, testi


special_tokens = [
    "<bof>",
    "<eof>",
    "<pad>",
    "<unk>"
]
uniq = sorted(set(''.join(ops_fmt)))
vocab = (
    {k: i for i, k in enumerate(special_tokens + uniq)}
)
logger.info(f"{vocab}")
kind, traini, testi = split()
train = [(i, ops_fmt[i]) for i in traini]
test = [(i, ops_fmt[i]) for i in testi]

logger.info(f'{len(train)=}')
logger.info(train[:2])
logger.info(f'{len(test)=}')
logger.info(test[:2])


def batch_iterator(raw_dataset, batch_size=1000):
    for i in range(0, len(raw_dataset), batch_size):
        yield raw_dataset[i: i + batch_size]

from transformers import GPT2Config, FlaxGPT2LMHeadModel

(model_dir := tk.datadir / "outputs" / "summing").mkdir(
    exist_ok=True, parents=True)
logger.info(vocab_full := list(
    set(special_tokens) 
    | set(uniq)))
tokenizer = toklib.CharBPETokenizer(
    suffix="",
)
tokenizer.train_from_iterator(
    [b for a, b in train],
    vocab_size=len(vocab_full),
    special_tokens=special_tokens,
    suffix="",
)

# %%
logger.info(tokenizer.get_vocab())
config = GPT2Config(
    vocab_size=len(vocab_full),
    n_layer=4,
    n_head=4,
    n_embd=32,
    bos_token_id=tokenizer.get_vocab()["<bof>"],
    eos_token_id=tokenizer.get_vocab()["<eof>"],
)
model = FlaxGPT2LMHeadModel(
    config,
    input_shape=(max_len, len(vocab_full))
)
config.save_pretrained(f"{model_dir}")
model.save_pretrained(f"{model_dir}")
tokenizer.save(f"{model_dir}/tokenizer.json")

# %%
# NOTE(tk) not sure what's a better way to add explicit special tokens
tokens = {
    'eos_token': '<eof>',
    'sep_token': '<eof>',
    'pad_token': '<pad>',
    'bos_token': '<bof>',
    'unk_token': '<unk>',
}
tokenizer = transformers.AutoTokenizer.from_pretrained(
    f"{model_dir}",
    model_max_length=max_len,
    **tokens
)
logger.info(f"{tokenizer.model_max_length=}")
logger.info(f"{tokenizer.vocab=}")
logger.info(f"{tokenizer.special_tokens_map=}")
# %%
tokenizer.save_pretrained(f"{model_dir}")

chk = tokenizer(
    train[0][1], 
    padding="max_length", 
    max_length=16)
logger.info(f"CHK: {chk}")

# %%

with open((name := tk.datadir / f"train_{kind}.csv"), "w") as f:
    logger.info(f"Writing {name}")
    writer = csv.writer(f)
    writer.writerow(('i', 'text', ))
    writer.writerows(
        (i, f"{tokenizer.bos_token}{x}{tokenizer.eos_token}", ) for i, x in train
    )

with open((name := tk.datadir / f"valid_{kind}.csv"), "w") as f:
    logger.info(f"Writing {name}")
    writer = csv.writer(f)
    writer.writerow(('i', 'text', ))
    writer.writerows(
        (i, f"{tokenizer.bos_token}{x}{tokenizer.eos_token}", ) for i, x in test
    )

# %%
logger.info(tokenizer("1+1=2"))

# %%
