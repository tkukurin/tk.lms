"""Some wrappers around HuggingFace tokenization utils.
"""
from __future__ import annotations

import json
import os
import string
import tempfile
from pathlib import Path
from typing import Any, Dict, Iterable, Sequence, Union

import tokenizers.implementations as toklib
import transformers
from huggingface_hub import unlike
from loguru import logger
from tokenizers import Tokenizer, models, pre_tokenizers
from tokenizers.implementations import CharBPETokenizer
from tokenizers.processors import TemplateProcessing
from transformers.tokenization_utils import AddedToken, PreTrainedTokenizer

special_tokens = {
    'eos_token': '<|eof|>',
    'sep_token': '<|eof|>',
    'pad_token': '<|pad|>',
    'bos_token': '<|bof|>',
    'unk_token': '<|unk|>',
    'cls_token': '<|bof|>',  # or cls?
    'reserved_token': '<|reserved>',
    'mask_token': '<|mask|>',
}
CFG_DEFAULT = transformers.GPT2Config()


def mkvocab(texts: Iterable, special_tokens: set = set(special_tokens.values())) -> dict:
    vocab = {k: i for i, k in enumerate(special_tokens)}
    for text in texts:
        for char in text:
            if char not in vocab:
                vocab[char] = len(vocab)
    return vocab


def mktokenizer_base(data, special_tokens: dict = special_tokens) -> toklib.BaseTokenizer:
    tokenizer = CharBPETokenizer(suffix="", unk_token=special_tokens['unk_token'])
    # TODO https://discuss.huggingface.co/t/add-bos-and-eos-when-encoding-a-sentence/21833
    # ??
    # https://github.com/huggingface/tokenizers/issues/704
    # tokenizer = Tokenizer(models.WordLevel(unk_token=special_tokens['unk_token']))
    # tokenizer.pre_tokenizer = pre_tokenizers.Split("", "isolated")

    sep = special_tokens['sep_token']
    bos = special_tokens['bos_token']
    eos = special_tokens['eos_token']
    tokenizer.post_processor = TemplateProcessing(
        single=f"{bos} $A {eos}",
        # see docs for TemplateProcessing for deets.
        #  [CLS]   ...   [SEP]   ...   [EOF]
        #    0      0      0      1      1
        pair=None, # f"{bos} $A {sep} $B:1 {eos}:1",
        special_tokens=[
            (special_tokens['bos_token'], 0),
            (special_tokens['eos_token'], 1),
            (special_tokens['unk_token'], 2),
        ],
    )

    tokenizer.train_from_iterator(
        data,
        # vocab_size=len(vocab_full),
        special_tokens=list(set(special_tokens.values())),
        min_frequency=1,
        # initial_alphabet=list(special_tokens),
        suffix="",
    )

    tokenizer.enable_padding(pad_id=1, pad_token=special_tokens['pad_token'])

    return tokenizer


def tokenizers_to_auto_tokenizer(
    tokenizer,
    config = CFG_DEFAULT,
    max_len: int = 512
) -> tuple[Any, transformers.PreTrainedTokenizerBase]:
    with tempfile.TemporaryDirectory() as tmpdir:
        config.save_pretrained(tmpdir)
        tokenizer.save(f"{tmpdir}/tokenizer.json")
        tokenizer = transformers.AutoTokenizer.from_pretrained(
            f"{tmpdir}",
            model_max_length=max_len,
            vocab_size=tokenizer.get_vocab_size(),
            **special_tokens
        )
        config = transformers.AutoConfig.from_pretrained(f"{tmpdir}")
    config.vocab_size = tokenizer.vocab_size
    config.bos_token_id = tokenizer.bos_token_id
    config.eos_token_id = tokenizer.eos_token_id
    config.unk_token_id = tokenizer.unk_token_id
    logger.info(f"{tokenizer.model_max_length=}")
    logger.info(f"{tokenizer.vocab=}")
    logger.info(f"{tokenizer.special_tokens_map=}")
    for x in (x for x in dir(tokenizer) if "_token_id" in x):
        logger.info(" ".join(map(str, [
            x[:x.find("_")],
            getattr(tokenizer, x),
            getattr(tokenizer, x.replace("_id", ""))
        ])))
    return config, tokenizer


def mktokenizer(
    data=string.printable,
    config=CFG_DEFAULT,
    max_len=512,
    special_tokens: dict = special_tokens
):
    tokenizer = mktokenizer_base(data, special_tokens)
    config, tokenizer = tokenizers_to_auto_tokenizer(
        tokenizer, config=config, max_len=max_len)
    return config, tokenizer
