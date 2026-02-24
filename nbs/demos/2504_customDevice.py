"""Jax models on CPU/GPU, plus explicit sync.

Creates a linear model then checks device
"""
# %%
import jax
import jax.numpy as jnp
import numpy as np
from dataclasses import dataclass
from typing import NamedTuple
from tqdm import tqdm
import functools as ft

# %%
class Params(NamedTuple):
    w: jnp.ndarray
    b: jnp.ndarray

@dataclass
class LinearModel:
    params: Params
    
    @classmethod
    def init(cls, key: jnp.ndarray, in_dim: int, out_dim: int) -> 'LinearModel':
        """Initialize a linear model with random weights."""
        k1, k2 = jax.random.split(key)
        w = jax.random.normal(k1, (in_dim, out_dim)) * 0.1
        b = jax.random.normal(k2, (out_dim,)) * 0.01
        return cls(Params(w=w, b=b))
    
    @property
    def device(self): return self.params.w.device
    def __call__(self, x: jnp.ndarray) -> jnp.ndarray:
        return x @ self.params.w + self.params.b

# %% Loss and update functions
def mse_loss(model: LinearModel, x: jnp.ndarray, y: jnp.ndarray) -> jnp.ndarray:
    """Mean squared error loss."""
    preds = model(x)
    return jnp.mean((preds - y) ** 2)

@ft.partial(jax.jit)
def update_step(params: Params, x: jnp.ndarray, y: jnp.ndarray, lr: float = 0.01) -> Params:
    """Single gradient descent update step."""
    def loss_fn(p): return mse_loss(LinearModel(p), x, y)
    grads = jax.grad(loss_fn)(params)
    return Params(
        w=params.w - lr * grads.w,
        b=params.b - lr * grads.b
    )

# %% Generate synthetic data
def generate_data(key, n_samples=100, in_dim=5, out_dim=1):
    k1, k2 = jax.random.split(key)
    x = jax.random.normal(k1, (n_samples, in_dim))
    true_w = jax.random.normal(k2, (in_dim, out_dim))
    y = x @ true_w + 0.1 * jax.random.normal(k2, (n_samples, out_dim))
    return x, y

# %% Main demo
_key = jax.random.PRNGKey(42)
_in_dim, _out_dim = 5, 1

cpu = jax.devices('cpu')[0]
with jax.default_device(cpu):  # Initialize CPU model
    cpu_model = LinearModel.init(_key, _in_dim, _out_dim)

if (gpus := jax.devices('gpu')):  # same params but on gpu
    gpu = gpus[0]
    with jax.default_device(gpu):
        gpu_params = jax.tree_map(lambda x: jax.device_put(x, gpu), cpu_model.params)
        gpu_model = LinearModel(gpu_params)
else:
    print("No GPU found - using CPU for both models")
    gpu = cpu
    gpu_model = cpu_model

# %%
print(f"CPU model device: {cpu_model.device}")
print(f"GPU model device: {gpu_model.device}")

# %%
x, y = generate_data(_key, n_samples=1000, in_dim=_in_dim, out_dim=_out_dim)
x_gpu = jax.device_put(x, gpu)
y_gpu = jax.device_put(y, gpu)

print("\nInitial CPU weights:\n", cpu_model.params.w[0])
print("Initial GPU weights:\n", gpu_model.params.w[0])

print("\nTraining GPU model...")
params = gpu_model.params
for _ in tqdm(range(100)):
    params = update_step(params, x_gpu, y_gpu, lr=0.1)

gpu_model = LinearModel(params)
print("\nCPU weights:\n", cpu_model.params.w[0])
print("\nGPU weights:\n", gpu_model.params.w[0])

cpu_params = jax.tree_map(lambda x: jax.device_put(x, cpu), gpu_model.params)
cpu_model = LinearModel(cpu_params)
print("\nSynced CPU weights:\n", cpu_model.params.w[0])

print("\nVerification:")
initial_w = jax.device_put(gpu_params.w, cpu)
final_w = cpu_model.params.w
print(f"Weights changed: {not np.allclose(initial_w, final_w)}")
print(f"L2 distance: {jnp.sqrt(jnp.sum((initial_w - final_w)**2))}")

# %%
