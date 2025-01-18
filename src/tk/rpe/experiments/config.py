"""Config by default same as in the original implementation.
"""

from ml_collections import config_dict
from tk.rpe.models.positional_encodings import PositionalEncodings as PE


def get_config(positional_encodings: str = 'NOISY_RELATIVE') -> config_dict.ConfigDict:
  """Get the default configuration.

  Should fit into 16G gpu such as T4.
  """
  config = config_dict.ConfigDict()
  
  # Training parameters
  config.batch_size = 128
  config.sequence_length = 40
  config.task = 'missing_duplicate_string'
  config.model = 'transformer_encoder'
  config.is_autoregressive = False
  config.computation_steps_mult = 0
  config.training_steps = 1_000
  
  # Architecture parameters
  config.arch = config_dict.ConfigDict()
  config.arch.num_layers = 5
  config.arch.embedding_dim = 64
  config.arch.dropout_prob = 0.1

  config.arch.positional_encodings = positional_encodings
  # idk ml collections can't give dict as params :'(
  if config.arch.positional_encodings.startswith("NOISY"):
    config.arch.positional_encodings_params = ({
        'noise_max_length': 2048,
    })
  
  return config

