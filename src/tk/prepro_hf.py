"""Quick processing to generate artefacts for [train].


[train]: train_hf.py
"""
# %%
from dataclasses import dataclass
from transformers import GPT2Config
import tk
from pathlib import Path

import csv
import tk
import numpy as np
import itertools as it
import numpy as np
from loguru import logger

from tk.utils.data.tokenizer import mktokenizer

# %%

def mkdata(outdir: None | Path = None):
    nums = np.arange(10)
    ops = [a + b for a, b in it.product(nums, nums)]
    ops_fmt = [f"{a}+{b}={a+b}" for a, b in it.product(nums, nums)]

    def split(kind: str = 'one_doubledigit'):
        singles = {i for i, x in enumerate(ops) if x < 10}
        doubles = {i for i, x in enumerate(ops) if x >= 10}
        traini = singles | {i for i in doubles if ops[i] % 10 == 1}
        testi = set(list(range(len(ops)))) - traini
        return kind, traini, testi
    
    kind, traini, testi = split()
    train = [(i, f"{ops_fmt[i]}") for i in traini]
    test = [(i, f"{ops_fmt[i]}") for i in testi]
    logger.info(f'{len(train)=}')
    logger.info(train[:2])
    logger.info(f'{len(test)=}')
    logger.info(test[:2])

    if outdir:
        with open((name := outdir / f"train_{kind}.csv"), "w") as f:
            logger.info(f"Writing {name}")
            writer = csv.writer(f)
            writer.writerow(('i', 'text', ))
            writer.writerows(train)

        with open((name := outdir / f"valid_{kind}.csv"), "w") as f:
            logger.info(f"Writing {name}")
            writer = csv.writer(f)
            writer.writerow(('i', 'text', ))
            writer.writerows(test)

    return train, test

# %%
@dataclass
class Cfg:
    output_dir: str = "."


def main(cfg: Cfg):
    logger.info(model_dir := Path(cfg.output_dir))
    train, test = mkdata(outdir=model_dir)
    tok_data = [x for i, x in train]
    config = GPT2Config(
        # NOTE(tk) explicitly set to invalid number
        # we expect to override these values
        # during experiment training time (train_hf.py)
        vocab_size=0,
        n_layer=0,
        n_head=0,
        n_embd=0,
    )
    config, tokenizer = mktokenizer(tok_data, config)
    tokenizer.save_pretrained(f"{model_dir}")
    config.save_pretrained(f"{model_dir}")
    logger.info(tokenizer("1+1=2"))
    return config, tokenizer


if False:  # running as notebook ... 
    output_dir = tk.datadir / "outputs" / "prepro/summing"
    output_dir.mkdir(exist_ok=True, parents=True)
    main(
        Cfg(
            output_dir=output_dir
        )
    )

# %%
if __name__ == '__main__':
    import hydra

    wrap = hydra.main(
        version_base="1.3", 
        config_path="configs", 
        config_name="prepro.yaml")
    wrap(main)()