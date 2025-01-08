"""Doesn't do anything right now.

Placeholder for howto ml collections.
    $ python -m tk.debate --config.task='asdf'
"""
import subprocess
import dataclasses as dc
from pathlib import Path

from absl import app
from ml_collections import config_flags
from tk.utils import cli, pprint


@dc.dataclass
class Config:
    task: str = 'bootstrap'


CFG = config_flags.DEFINE_config_dataclass(
    'config', Config())


def main(argv):
    c: Config = CFG.value
    pprint.print(c)


if __name__ == '__main__':
    app.run(main)
