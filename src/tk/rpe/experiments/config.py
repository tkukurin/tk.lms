"""Config by default same as in the original implementation.
"""

from ml_collections import config_dict
from ml_collections import config_flags
from tk.rpe.models.positional_encodings import PositionalEncodings as PE


def get_config(
    positional_encodings: str = 'NOISY_RELATIVE',
):
  """Get the default configuration."""
  config = config_dict.ConfigDict()
  
  # Training parameters
  config.batch_size = 128
  config.sequence_length = 40
  config.task = 'missing_duplicate_string'
  config.model = 'transformer_encoder'
  config.is_autoregressive = False
  config.computation_steps_mult = 0
  config.training_steps = 10_000
  
  # Architecture parameters
  config.arch = config_dict.ConfigDict()
  config.arch.num_layers = 5
  config.arch.embedding_dim = 64
  config.arch.dropout_prob = 0.1

  pe = PE.from_string(positional_encodings)
  config.arch.positional_encodings = str(positional_encodings)
  if pe in (PE.NOISY_LEARNT, ):
    config.arch.positional_encodings_params = ({
        'noise_max_length': 2048,
    }).items()
  
  return config
