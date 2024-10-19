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
from tokenizers import Tokenizer, models, pre_tokenizers, AddedToken
from tokenizers.implementations import CharBPETokenizer, BaseTokenizer
from tokenizers.processors import TemplateProcessing
from transformers.tokenization_utils import AddedToken, PreTrainedTokenizer

special_tokens = {
    'eos_token': '<|eof|>',
    'sep_token': '<|eof|>',
    'pad_token': '<|pad|>',
    'bos_token': '<|bof|>',
    'unk_token': '<|unk|>',
    'cls_token': '<|bof|>',  # or cls?
    'reserved_token': '<|reserved|>',
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
    # tokenizer = CharBPETokenizer(suffix="", unk_token=special_tokens['unk_token'])
    # https://discuss.huggingface.co/t/add-bos-and-eos-when-encoding-a-sentence/21833
    # https://github.com/huggingface/tokenizers/issues/704
    unk = special_tokens['unk_token']
    eos = special_tokens['eos_token']

    tokenizer = Tokenizer(models.WordLevel(
        vocab=None,
        unk_token=unk,
    ))
    tokenizer.pre_tokenizer = pre_tokenizers.Split(
        "", "isolated")
    trainer = tokenizer.model.get_trainer()

    tokenizer.train_from_iterator(
        data, trainer=trainer
        # vocab_size=len(vocab_full),
        # special_tokens=list(set(special_tokens.values())),
        # min_frequency=1,
        # initial_alphabet=list(special_tokens),
        # suffix="",
    )
    tokenizer.add_special_tokens(special_tokens := [  # type: ignore
        AddedToken(eos, special=True, single_word=True),
        AddedToken(unk, special=True, single_word=True),
        # AddedToken(bos, special=True, single_word=True),
        # AddedToken(sep, special=True, single_word=True),
    ])
    special_tokens = [
        (k.content, tokenizer.get_vocab()[k.content])
        for k in special_tokens
    ]
    tokenizer.post_processor = TemplateProcessing(
        single="$A",
        # see docs for TemplateProcessing for deets.
        #  [CLS]   ...   [SEP]   ...   [EOF]
        #    0      0      0      1      1
        pair=None, # f"{bos} $A {sep} $B:1 {eos}:1",
        special_tokens=special_tokens,
    )
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
            **special_tokens
        )
        config = transformers.AutoConfig.from_pretrained(f"{tmpdir}")
    # NB, idk tokenizer.vocab_size will still be wrong...
    config.vocab_size = len(tokenizer.get_vocab())
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

    def _decode(x, skip_special_tokens=False, *a, **kw):
        """idk this is a hacky way to fix the default decode which adds spaces."""
        id2t = tokenizer.convert_ids_to_tokens(
            x, skip_special_tokens=skip_special_tokens)
        return "".join(id2t)

    tokenizer.decode = _decode
    return config, tokenizer


def mktokenizer(
    data=string.printable + string.whitespace,
    config=CFG_DEFAULT,
    max_len=512,
    special_tokens: dict = special_tokens,
    return_og_tokenizer: bool = False,
):
    tokenizer_og = mktokenizer_base(data, special_tokens)
    config, tokenizer = tokenizers_to_auto_tokenizer(
        tokenizer_og, config=config, max_len=max_len)
    if return_og_tokenizer:
        return config, tokenizer_og, tokenizer
    return config, tokenizer
