"""Load single model from the jaxline dill.

We also run a single inference step and decode based on some stored vocab JSON.
"""
# %%
import tk
import pickle

path = tk.datadir / 'outputs/arc-ckpt/models/best/step_2476_2025-01-05T23:17:45/checkpoint.dill'
with open(path, 'rb') as f:
    ckpt = pickle.load(f)
# %%
print(ckpt.keys())
state = ckpt['state']
# %%
from flax.training.train_state import TrainState
assert isinstance(state, TrainState)
# %%
import tk
import json
with open(tk.datadir / 'mhodel_rearc/prog_only' / 'vocab.json', 'r') as f:
    vocab = json.load(f)
# %%
import jax.numpy as jnp
instr = 'x 1 = gravitate ('.split()
inarr = jnp.array([vocab[i] for i in instr])
# %%
import jax
rng = jax.random.PRNGKey(42)
params = state.params
logits = state.apply_fn(
    params,
    inarr[None, :],
    rngs={'dropout': rng},
    train=False
)
# %%
logits_val, logits_ix = jax.lax.top_k(logits[0, -1, :], 5)
id2tok = {v: k for k, v in vocab.items() if not k.startswith('__')} 
print([id2tok[i.item()] for i in logits_ix])
# %%
