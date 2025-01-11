"""Call corresponding eval in this folder.

    $ python -m tk.debate --c.dbg=False
"""
import tk
import dataclasses as dc
import importlib
import pkgutil
import subprocess
from pathlib import Path

from absl import app
from ml_collections import config_flags

from tk.debate import utils as dutil
from tk.utils import cli, pprint, utils
from tk.utils.log import L
import warnings

warnings.filterwarnings(
    "ignore", message="Starting from v4.46,.*",
)
warnings.filterwarnings(
    "ignore",
    message="You seem to be using the pipelines sequentially on GPU.*",
)

CFG = config_flags.DEFINE_config_file(
    "c", str(Path(__file__).parent / "configs/main.py"),
    "Path to the configuration file"
)


def find_module_with_prefix(package, prefix, task):
  """Find first module with given prefix in package."""
  for module_info in pkgutil.iter_modules(package.__path__):
    if module_info.name.startswith(prefix):
      return importlib.import_module(f'tk.debate.{task}.{module_info.name}')
  return None


def main(argv):
  cfg = CFG.value
  L.add(dutil._mkpath(cfg))
  L.info(f"{cfg=}")

  utils.post_mortem_debug()
  utils.seed_all(cfg.seed)

  do_gen = True
  if data := dutil.load(cfg):
    L.error("found result, not proceeding with generation. delete.")
    do_gen = False

  # dutil.generate_answer = tk.utils.memo(
  #     dutil.LocalModel(cfg.model).__call__, seed=cfg.seed
  # )
  dutil.generate_answer = dutil.LocalModel(cfg.model, cached=True)

  try:
    package = importlib.import_module(f'tk.debate.{cfg.task}')
    L.info(f'looking in {package.__path__}')

    gen_module = find_module_with_prefix(package, 'gen_', cfg.task)
    eval_module = find_module_with_prefix(package, 'eval_', cfg.task)

    if not gen_module and not eval_module:
      L.error(f"No gen_ or eval_ modules found in {package.__path__}")
      return

    if gen_module and do_gen:
      L.info("Running generation module")
      gen_module.main(cfg=cfg)
    else:
      L.warning(f"generation: {do_gen=} (or no module found)")

    if eval_module:
      L.info("Running evaluation module")
      eval_module.main(cfg=cfg)
    else:
      L.warning("No eval module found")

  except ImportError as e:
    print(f"Error: {e}")


if __name__ == '__main__':
  app.run(main)
