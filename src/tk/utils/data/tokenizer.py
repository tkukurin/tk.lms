import transformers
import tokenizers.implementations as toklib
import tempfile

from loguru import logger
from typing import Iterable

special_tokens = {
    'eos_token': '<eof>',
    'sep_token': '<eof>',
    'pad_token': '<pad>',
    'bos_token': '<bof>',
    'unk_token': '<unk>',
}


def mkvocab(texts: Iterable, special_tokens: set = set(special_tokens.values())) -> dict:
    vocab = {k: i for i, k in enumerate(special_tokens)}
    for text in texts:
        for char in text:
            if char not in vocab:
                vocab[char] = len(vocab)
    return vocab


def mktokenizer_base(data, special_tokens: set = set(special_tokens.values())) -> toklib.BaseTokenizer:
    # uniq = sorted(set(''.join(data)))
    # logger.info(vocab_full := list(
    #     set(special_tokens) | set(uniq)))
    tokenizer = toklib.CharBPETokenizer(
        # vocab={k: i for i, k in enumerate(special_tokens)},
        suffix="",
    )
    tokenizer.train_from_iterator(
        data,
        # vocab_size=len(vocab_full),
        special_tokens=list(special_tokens),
        min_frequency=5,
        # initial_alphabet=list(special_tokens),
        suffix="",
    )
    return tokenizer


def mktokenizer(data, config, max_len=512, special_tokens: dict = special_tokens):
    tokenizer = mktokenizer_base(data, set(special_tokens.values()))
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
    logger.info(f"{tokenizer.model_max_length=}")
    logger.info(f"{tokenizer.vocab=}")
    logger.info(f"{tokenizer.special_tokens_map=}")
    return config, tokenizer
        
