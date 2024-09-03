import jax.numpy as jnp
import optax
import jax

from jax import random

from flax import linen as nn
from flax.training.train_state import TrainState


def train_step(
    x: jnp.ndarray, 
    y: jnp.ndarray, 
    mask: jnp.ndarray, 
    state: TrainState, 
    dropout_key: random.PRNGKey
) -> tuple[jnp.ndarray, TrainState]:
    dropout_key = jax.random.fold_in(
        dropout_key, state.step)

    def loss_fn(params: dict) -> jnp.ndarray:
        logits = state.apply_fn(
            {'params': params}, 
            x, 
            train=True, 
            rngs={'dropout': dropout_key})
        loss = cross_entropy_loss(logits, y, mask)
        return loss
    
    loss, grads = jax.value_and_grad(
        loss_fn, has_aux=False)(state.params)
    new_state = state.apply_gradients(grads=grads)
    return new_state, loss


def eval_step(
    x: jnp.ndarray, 
    y: jnp.ndarray, 
    mask: jnp.ndarray, 
    state: TrainState, 
    dropout_key: random.PRNGKey
) -> tuple[jnp.ndarray, TrainState]:
    logits = state.apply_fn(
        {'params': state.params}, 
        x, 
        train=False, 
        # should be able to skip??
        rngs={'dropout': dropout_key}
    )
    probs = nn.softmax(logits, axis=-1)
    yhat = logits.argmax(-1)
    metrics = dict(
        acc_all = ((yhat == y) * mask).sum() / mask.sum(),
        loss = cross_entropy_loss(logits, y, mask)
    )
    return probs, metrics


def create_train_state(
    rng, 
    model, 
    learning_rate, 
    input_shape=None, 
):
    input_shape = input_shape or (1, model.config.block_size)
    x = jnp.ones(input_shape, dtype=jnp.int32)
    params = model.init(rng, x)['params']
    tx = optax.adam(learning_rate)
    return TrainState.create(
        apply_fn=model.apply, params=params, tx=tx)



def cross_entropy_loss(logits: jnp.ndarray,
                       labels: jnp.ndarray,
                       mask: jnp.ndarray | None = None,
                       label_smoothing: float = 0.0,
                       ) -> jnp.ndarray:
    """Cross entropy, averaged over the first dimension (samples)."""
    if labels.shape[-1] != logits.shape[-1]:
        labels = nn.one_hot(labels, num_classes=logits.shape[-1])
    if label_smoothing > 0:
        smoothing = jnp.ones_like(labels) / labels.shape[-1]
        labels = ((1-label_smoothing) * labels + label_smoothing * smoothing)
    log_softmax_logits = jax.nn.log_softmax(logits)
    if mask is None:
        mask = jnp.ones(logits.shape[:-1])
    mask = mask.reshape([*logits.shape[:-1], 1])
    loss = -jnp.sum(labels * log_softmax_logits * mask) / mask.sum()
    return jnp.nan_to_num(loss)
