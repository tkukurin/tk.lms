"""Minor scripts to fetch online data.
"""
import gzip

import numpy as np

from pathlib import Path
from typing import NamedTuple
from tk.utils.utils import fetch, datadir
from loguru import logger


class Xy(NamedTuple):
    x: np.ndarray
    y: np.ndarray


mnist_default_dir = datadir / "MNIST" / "raw"
mnist_locs = [
    'https://yann.lecun.com/exdb/mnist/train-images-idx3-ubyte.gz',
    'https://yann.lecun.com/exdb/mnist/train-labels-idx1-ubyte.gz',
    'https://yann.lecun.com/exdb/mnist/t10k-images-idx3-ubyte.gz',
    'https://yann.lecun.com/exdb/mnist/t10k-labels-idx3-ubyte.gz',
]


def tinyshakespeare() -> str:
    url = 'https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt'
    return fetch(url).decode('utf8')


def words(lang: str = 'en') -> set[str]:
    assert lang in ('en', )
    url = 'https://github.com/dwyl/english-words/raw/master/words.txt'
    text = fetch(url).decode('utf8')
    return {x for x in text.split('\n') if x.strip()}


def mnist(mnist_dir: str | Path = mnist_default_dir) -> tuple[dict, dict]:
    mnist_dir.mkdir(parents=True, exist_ok=True)
    for loc in mnist_locs:
        logger.info(f'Downloading: {loc}\n -> {mnist_dir}')
        with open(mnist_dir, 'wb') as f:
            data = fetch(loc)
            f.write(data)
    return (
        load_mnist_gzip(mnist_dir, "train"), 
        load_mnist_gzip("t10k"),
    )


def load_mnist_gzip(which='train', data_dir: Path = mnist_default_dir) -> dict:
    """Gunzip from [mnist_locs].
    """
    import numpy as np
    images_path = Path(data_dir) / f'{which}-images-idx3-ubyte.gz'
    labels_path = Path(data_dir) / f'{which}-labels-idx1-ubyte.gz'

    with gzip.open(labels_path, 'rb') as lbl_f:
        labels = np.frombuffer(lbl_f.read(), dtype=np.uint8, offset=8)
    with gzip.open(images_path, 'rb') as f:
        img_data = np.frombuffer(
            f.read(), dtype=np.uint8, offset=16
        ).reshape(-1, 28, 28)

    return Xy(img_data, labels)
