"""Jaxline experiment runner.
"""

import datetime
import functools
import os
import signal
import threading

from absl import app
from absl import flags
from absl import logging
import dill
import jax
from tk.jaxline import experiment
from tk.jaxline import platform
from tk.jaxline import utils

FLAGS = flags.FLAGS


def _restore_state_to_in_memory_checkpointer(restore_path):
  """Initializes experiment state from a checkpoint."""
  python_state_path = (
    restore_path if str(restore_path).endswith('checkpoint.dill')
    else os.path.join(restore_path, 'checkpoint.dill'))
  with open(python_state_path, 'rb') as f:
    pretrained_state = dill.load(f)
  logging.info('Restored checkpoint from %s', python_state_path)

  from ml_collections import ConfigDict
  step = pretrained_state.pop('global_step')
  rng = utils.get_first(
    utils.bcast_local_devices(pretrained_state.pop('train_step_rng')))
  jaxline_state = ConfigDict(dict(
      global_step=step,
      train_step_rng=rng,
      experiment_module={
        k: utils.get_first(utils.bcast_local_devices(v))
        for k, v in pretrained_state.items()
      }))
  snapshot = utils.SnapshotNT(0, jaxline_state)
  utils.GLOBAL_CHECKPOINT_DICT['latest'] = utils.CheckpointNT(
      threading.local(), [snapshot])


def _get_step_date_label(global_step):
  # Date removing microseconds.
  date_str = datetime.datetime.now().isoformat().split('.')[0]
  return f'step_{global_step}_{date_str}'


def _setup_signals(save_model_fn):
  """Sets up a signal for model saving."""
  # Save a model on Ctrl+C.
  def sigint_handler(unused_sig, unused_frame):
    # Ideally, rather than saving immediately, we would then "wait" for a good
    # time to save. In practice this reads from an in-memory checkpoint that
    # only saves every 30 seconds or so, so chances of race conditions are very
    # small.
    save_model_fn()
    logging.info(r'Use `Ctrl+\` to save and exit.')

  # Exit on `Ctrl+\`, saving a model.
  prev_sigquit_handler = signal.getsignal(signal.SIGQUIT)
  def sigquit_handler(unused_sig, unused_frame):
    # Restore previous handler early, just in case something goes wrong in the
    # next lines, so it is possible to press again and exit.
    signal.signal(signal.SIGQUIT, prev_sigquit_handler)
    save_model_fn()
    logging.info(r'Exiting on `Ctrl+\`')

    # Re-raise for clean exit.
    os.kill(os.getpid(), signal.SIGQUIT)

  signal.signal(signal.SIGINT, sigint_handler)
  signal.signal(signal.SIGQUIT, sigquit_handler)


def _save_state_from_in_memory_checkpointer(
    save_path, experiment_class: experiment.AbstractExperiment):
  """Saves experiment state to a checkpoint."""
  logging.info('Saving model.')
  for (checkpoint_name,
       checkpoint) in utils.GLOBAL_CHECKPOINT_DICT.items():
    if not checkpoint.history:
      logging.info('Nothing to save in "%s"', checkpoint_name)
      continue

    pickle_nest = checkpoint.history[-1].pickle_nest
    global_step = pickle_nest['global_step']
    rng = pickle_nest['train_step_rng']
    module = pickle_nest['experiment_module']
    state_dict = {
      'global_step': global_step,
      'train_step_rng': rng,
    }
    for _, key in experiment_class.CHECKPOINT_ATTRS.items():
      state_dict[key] = jax.tree.map(
        lambda x: x.to_dict() if hasattr(x, 'to_dict') else x, module[key]
      )
    save_dir = os.path.join(
        save_path, checkpoint_name, _get_step_date_label(global_step))
    python_state_path = os.path.join(save_dir, 'checkpoint.dill')
    os.makedirs(save_dir, exist_ok=True)
    with open(python_state_path, 'wb') as f:
      dill.dump(state_dict, f)
    logging.info(
        'Saved "%s" checkpoint to %s', checkpoint_name, python_state_path)
    return os.path.abspath(save_dir)



def main(argv):
  logging.info(f'{(experiment_class := FLAGS.config.experiment_class)=}')
  
  if restore_path := FLAGS.config.restore_path:
    _restore_state_to_in_memory_checkpointer(restore_path)

  save_dir = os.path.join(FLAGS.config.checkpoint_dir, 'models')
  if FLAGS.config.one_off_evaluate:
    save_model_fn = lambda: None
  else:
    save_model_fn = functools.partial(
        _save_state_from_in_memory_checkpointer, save_dir, experiment_class)
  _setup_signals(save_model_fn)

  try:
    platform.main(experiment_class, argv)
  finally:
    save_model_fn()  # Save at the end of training or in case of exception.


if __name__ == '__main__':
  flags.mark_flag_as_required('config')
  app.run(lambda argv: main(argv))