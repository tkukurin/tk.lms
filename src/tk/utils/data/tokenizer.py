import transformers
import tokenizers.implementations as toklib

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


def mktokenizer(data, config_dir, max_len=512, special_tokens: dict = special_tokens):
    tokenizer = mktokenizer_base(data, set(special_tokens.values()))
    tokenizer.save(f"{config_dir}/tokenizer.json")
    tokenizer = transformers.AutoTokenizer.from_pretrained(
        f"{config_dir}",
        model_max_length=max_len,
        vocab_size=tokenizer.get_vocab_size(),
        # bos_token_id=tokenizer.bos_token_id,
        # eos_token_id=tokenizer.eos_token_id,
        **special_tokens
    )
    config = transformers.AutoConfig.from_pretrained(f"{config_dir}")
    config.bos_token_id = tokenizer.bos_token_id
    config.eos_token_id = tokenizer.eos_token_id

    config.save_pretrained(f"{config_dir}")
    tokenizer.save_pretrained(f"{config_dir}")

    logger.info(f"{tokenizer.model_max_length=}")
    logger.info(f"{tokenizer.vocab=}")
    logger.info(f"{tokenizer.special_tokens_map=}")
    return config, tokenizer
        
