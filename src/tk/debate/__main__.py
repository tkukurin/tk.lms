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
from ml_collections import config_flags


flags.DEFINE_string(
    'task', None, 'Task to run',
    required=True
)
flags.DEFINE_boolean(
    'dbg', True, 'Debug mode',
    short_name='d'
)
CFG = config_flags.DEFINE_config_file(
    "cfg",
    str(Path(__file__).parent / "configs" / "main.py"),
    "Path to the configuration file"
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
    cfg = CFG.value
    L.info(f"{cfg=}")
    utils.post_mortem_debug()

    from tk.debate import utils as dutil # hackyy
    dutil.generate_answer = dutil.LocalModel(cfg.model)

    do_gen = True
    if dutil.load(cfg, F.task, dbg=F.dbg):
        L.error("found result, not proceeding with generation. delete.")
        do_gen = False

    try:
        package = importlib.import_module(f'tk.debate.{F.task}')
        L.info(f'looking in {package.__path__}')

        gen_module = find_module_with_prefix(package, 'gen_')
        eval_module = find_module_with_prefix(package, 'eval_')

        if not gen_module and not eval_module:
            L.error(f"No gen_ or eval_ modules found in {package.__path__}")
            return

        if gen_module and do_gen:
            L.info("Running generation module")
            gen_module.main(cfg=cfg, dbg=F.dbg)
        else:
            L.warning(f"generation: {do_gen=} (or no module found)")
        
        if eval_module:
            L.info("Running evaluation module")
            eval_module.main(cfg=cfg, dbg=F.dbg)
        else:
            L.warning("No eval module found")

    except ImportError as e:
        print(f"Error: {e}")


if __name__ == '__main__':
    app.run(main)
