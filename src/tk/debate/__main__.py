"""Call corresponding eval in this folder.

    $ python -m tk.debate --nodbg --task='mmlu'
"""
import subprocess
import dataclasses as dc
from pathlib import Path

from absl import app
from tk.utils import cli, pprint, utils
from absl import flags

from tk.utils.log import L


flags.DEFINE_string(
    'task', None, 'Task to run',
    required=True
)
flags.DEFINE_boolean(
    'dbg', True, 'Debug mode',
    short_name='d'
)
F = flags.FLAGS


def main(argv):
    utils.post_mortem_debug()
    try:
        import importlib
        import pkgutil
        package = importlib.import_module(f'tk.debate.{F.task}')
        L.info(f'looking in {package.__path__}')
        module = None
        for module_info in pkgutil.iter_modules(package.__path__):
            if module_info.name.startswith('gen_'):
                module = importlib.import_module(
                    f'tk.debate.{F.task}.{module_info.name}')
                break
        if not module:
            L.error(f"No module found in {package.__path__}")
            return
        module.main(F.dbg)
    except ImportError as e:
        print(f"Error: {e}")


if __name__ == '__main__':
    app.run(main)
