"""Call corresponding eval in this folder.

    $ python -m tk.debate -- --nodbg --config.task='mmlu'
"""
import subprocess
import dataclasses as dc
from pathlib import Path

from absl import app
from ml_collections import config_flags
from tk.utils import cli, pprint, utils
from absl import flags

F = flags.FLAGS
flags.DEFINE_boolean(
    'dbg', True, 'Debug mode',
    short_name='d'
)


@dc.dataclass
class Config:
    task: str = 'mmlu'


CFG = config_flags.DEFINE_config_dataclass(
    'config', Config())


def main(argv):
    c: Config = CFG.value
    utils.post_mortem_debug()
    pprint.print(c)
    try:
        import importlib
        module = importlib.import_module(f'tk.debate.{c.task}.gen_{c.task}')
        module.main(F.dbg)
    except ImportError as e:
        print(f"Error: {e}")


if __name__ == '__main__':
    app.run(main)
