"""Example script to train and evaluate a network."""

from pathlib import Path
from absl import app
import haiku as hk
import jax.numpy as jnp
import numpy as np

from tk.rpe.experiments import constants
from tk.rpe.experiments import curriculum as curriculum_lib
from tk.rpe.experiments import training
from tk.rpe.experiments import utils
from ml_collections import config_flags
from tk.utils.utils import post_mortem_debug

_CFG = config_flags.DEFINE_config_file(
  'c', 
  str(Path(__file__).parent / 'config.py'),
  'Path to the configuration file.')


def main(_) -> None:
  post_mortem_debug()

  cfg = _CFG.value
  params = cfg.arch

  curriculum = curriculum_lib.UniformCurriculum(
      values=list(range(1, cfg.sequence_length + 1))
  )
  task = constants.TASK_BUILDERS[cfg.task]()

  # Create the model.
  single_output = task.output_length(10) == 1
  print(params)
  model = constants.MODEL_BUILDERS[cfg.model](
      output_size=task.output_size,
      return_all_outputs=True,
      **params,
  )
  if cfg.is_autoregressive:
    if 'transformer' not in cfg.model:
      model = utils.make_model_with_targets_as_input(
          model, cfg.computation_steps_mult
      )
    model = utils.add_sampling_to_autoregressive_model(model, single_output)
  else:
    model = utils.make_model_with_empty_targets(
        model, task, cfg.computation_steps_mult, single_output
    )
  model = hk.transform(model)

  # Create the loss and accuracy based on the pointwise ones.
  def loss_fn(output, target):
    loss = jnp.mean(jnp.sum(task.pointwise_loss_fn(output, target), axis=-1))
    return loss, {}

  def accuracy_fn(output, target):
    mask = task.accuracy_mask(target)
    return jnp.sum(mask * task.accuracy_fn(output, target)) / jnp.sum(mask)

  # Create the final training parameters.
  training_params = training.ClassicTrainingParams(
      seed=0,
      model_init_seed=0,
      training_steps=cfg.training_steps,
      log_frequency=100,
      length_curriculum=curriculum,
      batch_size=cfg.batch_size,
      task=task,
      model=model,
      loss_fn=loss_fn,
      learning_rate=1e-3,
      l2_weight=0.0,
      accuracy_fn=accuracy_fn,
      compute_full_range_test=True,
      max_range_test_length=100,
      range_test_total_batch_size=512,
      range_test_sub_batch_size=64,
      is_autoregressive=cfg.is_autoregressive,
  )

  training_worker = training.TrainingWorker(training_params, use_tqdm=True)
  _, eval_results, _ = training_worker.run()

  # Gather results and print final score.
  accuracies = [r['accuracy'] for r in eval_results]
  score = np.mean(accuracies[cfg.sequence_length + 1 :])
  print(f'Score: {score}')


if __name__ == '__main__':
  app.run(main)
