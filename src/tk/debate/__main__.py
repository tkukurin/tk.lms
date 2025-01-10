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
import pkgutil
import importlib


flags.DEFINE_string(
    'task', None, 'Task to run',
    required=True
)
flags.DEFINE_boolean(
    'dbg', True, 'Debug mode',
    short_name='d'
)
F = flags.FLAGS


def find_module_with_prefix(package, prefix):
    """Find first module with given prefix in package."""
    for module_info in pkgutil.iter_modules(package.__path__):
        if module_info.name.startswith(prefix):
            return importlib.import_module(
                f'tk.debate.{F.task}.{module_info.name}'
            )
    return None


def main(argv):
    utils.post_mortem_debug()
    try:
        package = importlib.import_module(f'tk.debate.{F.task}')
        L.info(f'looking in {package.__path__}')

        gen_module = find_module_with_prefix(package, 'gen_')
        eval_module = find_module_with_prefix(package, 'eval_')

        if not gen_module and not eval_module:
            L.error(f"No gen_ or eval_ modules found in {package.__path__}")
            return

        if gen_module:
            L.info("Running generation module")
            gen_module.main(F.dbg)
        
        if eval_module:
            L.info("Running evaluation module")
            eval_module.main(F.dbg)

    except ImportError as e:
        print(f"Error: {e}")


if __name__ == '__main__':
    app.run(main)
