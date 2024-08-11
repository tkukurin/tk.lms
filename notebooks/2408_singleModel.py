"""Quick model training dynamics script.
"""
# %%
try: from rich import print as show
except: from IPython.display import display as show

import flax.traceback_util
import flax.traverse_util
import flax.traverse_util
import jax
import jax.numpy as jnp
import flax.linen as nn
import optax
from flax.training import train_state as tslib
from flax.configurations import Config
from jax import random
import numpy as np
from loguru import logger
import penzai.pz
import penzai.toolshed
import penzai.treescope
from tk.utils.data import tokenizer as toklib
from tk.prepro_hf import mkdata
from tk.models import gpt2
import functools as ft
from flax.training.train_state import TrainState


train, test = mkdata()
train = [x for _, x in train]
test = [x for _, x in test]
config, tokenizer = toklib.mktokenizer(train)

# %%
train_overfit = [x for x in train if x[0] == '0']
eval_overfit = [x for x in train if x[2] == '0']
logger.info(f"{train_overfit=}")
logger.info(f"{eval_overfit=}")

# %%
chr2id = {k:v for k,v in tokenizer.vocab.items()}
id2chr = {v:k for k, v in tokenizer.vocab.items()}
model = gpt2.GPT(gpt2.GPTConfig(
    block_size=16,
    vocab_size=len(chr2id),
    num_layers=4,
    num_heads=12,
    num_embeds=768
))

# %%
def create_train_state(rng, model, learning_rate, input_shape=None):
    input_shape = input_shape or (1, model.config.block_size)
    x = jnp.ones(input_shape, dtype=jnp.int32)
    params = model.init(rng, x)['params']
    tx = optax.adam(learning_rate)
    return TrainState.create(apply_fn=model.apply, params=params, tx=tx)


rng = jax.random.PRNGKey(42)
rng, trng = jax.random.split(rng)
train_state = create_train_state(trng, model, 1e-4)

# %%
def pad(seq, maxlen):
    pad = [tokenizer.pad_token_id] * (maxlen - 2 - len(seq))
    return [tokenizer.bos_token_id] + seq + [tokenizer.eos_token_id] + pad


def encode(seq, maxlen=None, tensorize=True):
    seq = [chr2id[chr] for chr in seq]
    mask = [1] * len(seq)
    if maxlen:
        seq = pad(seq, maxlen)
        mask = mask + ([0] * (len(seq) - len(mask)))
    if tensorize:
        seq = jnp.array(seq)
        mask = jnp.array(mask)
    return seq, mask


def labelize(seq, mask=None):
    return (
        jnp.array(seq[:-1]), 
        jnp.array(seq[1:]),
        jnp.array(mask[:-1]) if mask is not None else jnp.ones_like(seq[1:]),
    )


def decode(tokens):
    if isinstance(tokens, jnp.ndarray):
        tokens = tokens.tolist()
    return ''.join(id2chr[t] for t in tokens)


print(encode(train[0]))
print(encode(train[0], 16))
print(labelize(*encode(train[0], 16)))
print(decode(encode(train[0])[0]))
# %%

tokens = [labelize(*encode(x, maxlen=16)) for x in train_overfit]
logger.debug(x := jnp.array([a for a, b, c in tokens]))
y = jnp.array([b for a, b, c in tokens])
mask = jnp.array([c for a, b, c in tokens])

params = train_state.params
idx, mask = encode(train[0], 16)
idx, tgt, mask = labelize(idx, mask)
rng, mrng = jax.random.split(rng)
print(logits := model.apply(
    {"params": params}, idx[None, :], rngs={'dropout': mrng}))
print(logits.shape)

# %%

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


print(tgt)
print(oh_tgt := nn.one_hot(tgt, num_classes=17))
print(cross_entropy_loss(
    logits=oh_tgt * jnp.inf,
    labels=oh_tgt,
))

# %%

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


# %%
import flax
import altair as alt
import pandas as pd


def get_chart(kv: dict[tuple, tuple]):

    df = pd.DataFrame([
        {
            'Layer': key[0], 
            'Type': '.'.join(key[1:]), 
            'Mean': value[0],
            'Std': value[1]
        }
        for key, value in kv.items()
    ])
    
    chart = alt.Chart(df).mark_point(filled=True).encode(
        # x='Layer:N',
        # y='Mean:Q',
        x=alt.X('Layer:N', axis=alt.Axis(labelAngle=0, title='Layer')),
        y=alt.Y('Mean:Q', title='Mean Weight'),
        color='Type:N',
        tooltip=['Layer', 'Type', 'Mean', 'Std']
    ).properties(
        width=400,
        height=300
    ).interactive()

    error_bars = alt.Chart(df).mark_errorbar(extent='stdev').encode(
        x='Layer:N',
        y='Mean:Q',
        yError='Std:Q',
        color='Type:N'
    )

    return chart + error_bars


def get_stats(state: dict) -> dict:

    def stats(v: jnp.ndarray):
        mu = (v.mean())
        std = (v.std())
        return (mu.item(), std.item())

    stats_out = jax.tree.map(stats, state.params)
    stats_flat = flax.traverse_util.flatten_dict(stats_out)

    final_chart = get_chart({
        k: v for k, v in stats_flat.items() if (
            'bias' not in k
            and 
            all('ln' not in k for k in k)
        )
    })
    
    return {
        'chart': final_chart,
        'stats': stats_flat,
    }


# %%
rng = jax.random.PRNGKey(0)
state = create_train_state(
    rng, 
    model, 
    learning_rate=5e-5, 
)
data = get_stats(state)
print(data['stats'])
# %%
import itertools as it


def joint_keys(*dicts: dict):
    keys = set(it.chain(*dicts))
    return sorted(keys)


def tree_zip(*dicts: dict):
    keys = joint_keys(*dicts)
    outputs = {}
    for k in keys:
        outputs[k] = tuple(d.get(k) for d in dicts)
    return outputs


def diff(values: tuple[jnp.ndarray]):
    deltas = []
    for v1, v2 in zip(values, values[1:]):
        mu1, _ = v1
        mu2, _ = v2
        delta = (mu2 - mu1)
        deltas.append(delta)
    return deltas

# %%

from typing import Literal
import dataclasses as dcls
from typing import NamedTuple


class Enc(NamedTuple):
    x: jnp.ndarray
    y: jnp.ndarray
    mask: jnp.ndarray
    raw: list[str]


class Exp(NamedTuple):
    name: str
    train: Enc
    test: Enc
    # model: gpt2.GPT
    # state: TrainState
    # rng: random.PRNGKey


def experiment_data(name: Literal["autoregressive", "classification"]) -> tuple[
    gpt2.GPT,
    TrainState,
    random.PRNGKey,
    Exp
]:
    tokens_train = [
        labelize(*encode(x, maxlen=8)) for x in train_overfit
    ]
    tokens_eval = [
        labelize(*encode(x, maxlen=8)) for x in eval_overfit
    ]
    tokens = tokens_train + tokens_eval
    logger.debug(x := jnp.array([a for a, b, c in tokens]))
    logger.debug(y := jnp.array([b for a, b, c in tokens]))
    if name == "autoregressive":
        mask = jnp.array([c for a, b, c in tokens])
    else:
        mask = np.zeros_like(y)
        mask[:, 4] = 1
        assert len(np.unique(y[:, 4])) > 1, f"wrong setup:\n{y}"
    logger.debug(mask)
    n = len(mask) // 2
    assert n == len(tokens_train) == len(tokens_eval)
    xtr, ytr, masktr = x[:n], y[:n], mask[:n]
    xev, yev, maskev = x[n:], y[n:], mask[n:]

    state = create_train_state(
        (rng := random.PRNGKey(42)),
        (model := gpt2.GPT(gpt2.GPTConfig(
            block_size=16,
            vocab_size=len(id2chr),
            num_embeds=128,
            num_layers=4,
            num_heads=4,
            dropout_rate=0.1,
            use_bias=True,
        ))),
        learning_rate=5e-5
    )

    return model, state, rng, Exp(
        name=name,
        train=Enc(xtr, ytr, masktr, raw=tokens_train),
        test=Enc(xev, yev, maskev, raw=tokens_eval),
    )



model, state, rng, exp = experiment_data(name="autoregressive")
exp_raw_x = exp.train.raw
exp_x = exp.train.x
exp_y = exp.train.y
exp_mask = exp.train.mask
# %%
stat_history = []
rng, dkey = random.split(rng)
min_loss = -1, 9999, state.params
max_acc = -1, 0, state.params
# %%
def eval_step(x, y, mask, model, params):
    logits = model.apply(
        {'params': params}, 
        x, 
        train=False, 
        rngs={'dropout': rng}  # unused? can we skip?
    )
    yhat = logits.argmax(-1)
    metrics = dict(
        acc = (yhat[:, 4] == y[:, 4]).mean(),
        acc_all = (yhat == y).mean(),
        loss = cross_entropy_loss(logits, y, mask)
    )
    return yhat, metrics


num_epochs = 125
cur_epoch = len(stat_history)
for epoch in range(cur_epoch, cur_epoch + num_epochs):
    state, loss = train_step(
        exp.train.x, exp.train.y, exp.train.mask, state, dropout_key=rng)
    data = get_stats(state)
    # data['loss'] = loss
    train_yhat, train_metrics = eval_step(
        exp.train.x, exp.train.y, exp.train.mask, model, params=state.params)
    eval_yhat, eval_metrics = eval_step(
        exp.test.x, exp.test.y, exp.test.mask, model, params=state.params)
    data['train'] = train_metrics
    data['eval'] = eval_metrics
    if loss < min_loss[1]:
        print('Saving[loss]@', loss)
        min_loss = (epoch, loss, state.params.copy())
    if (acc := data['train']['acc_all']) > max_acc[1]:
        print('Saving[acca]@', acc)
        max_acc = (epoch, acc, state.params.copy())
    stat_history.append(data)
    print(f"Epoch {epoch+1}, Loss: {loss:.4f}")


# %%
import matplotlib.pyplot as plt
import matplotlib.colors as mrgb

clist = list(mrgb.TABLEAU_COLORS)

fig, grid = plt.subplots(
    nrows=2, ncols=1)

ax1 = grid[0]
ax1.set_ylabel('loss', color=clist[0])
ax1.plot([x['train']['loss'] for x in stat_history], color=clist[0], label='train')
ax1.plot([x['eval']['loss'] for x in stat_history], color=clist[0], label='eval', linestyle='dashed')
ax1.legend()
ax1.tick_params(axis='y', labelcolor=clist[0])

ax2 = ax1.twinx()
ax2.set_ylabel('acc', color=clist[1])
ax2.plot([x['train']['acc'] for x in stat_history], color=clist[1], label='train')
ax2.plot([x['eval']['acc'] for x in stat_history], color=clist[1], label='eval', linestyle='dashed')
ax2.legend()
ax2.tick_params(axis='y', labelcolor=clist[1])
ax2.set_yticks(np.arange(10 + 1) / 10)

ax3 = grid[1]
ax3.set_ylabel('diff', color=clist[3])
ax3.tick_params(axis='y', color=clist[3], labelcolor=clist[3])
total_change_per_epoch = np.array([
    np.mean([mu for mu, var in x['stats'].values()])
    for x in stat_history
])
total_var_per_epoch = np.array([
    np.mean([abs(var) for mu, var in x['stats'].values()])
    for x in stat_history
])
ax3.plot(total_change_per_epoch, color=clist[2])

ax4 = ax3.twinx()
ax4.plot(total_var_per_epoch, color=clist[3])
ax4.set_ylabel('var', color=clist[3])
ax4.tick_params(axis='y', color=clist[3], labelcolor=clist[3])

plt.show()

# %%
zipped = tree_zip(*(x['stats'] for x in stat_history[-5:]))
deltas_over_epochs = jax.tree.map(
    diff, zipped, is_leaf=lambda x: isinstance(x, (tuple, list)))

get_chart(deltas_over_epochs)
# %%
get_chart({k:v for k, v in deltas_over_epochs.items() if 
    any(x in k for x in ('wpe', 'wte'))
})
# %%
deltas_over_epochs.keys()
total_change = jax.tree.reduce(lambda a, b: abs(a) + abs(b), deltas_over_epochs)
print(total_change)
# %%
# params = state.params
*_, params = min_loss
bound_model = model.bind(
    {'params': params}, 
    rngs={'dropout': rng})
train_preds = bound_model(exp_x, train=False)
print(train_preds.shape)
# %%
# <bof>a + b = c
# 0    1 2 3 4 5
probs = nn.softmax(train_preds[:, 4])
print('after', len(stat_history), 'epochs')
# %%
import treescope  # comes with penzai
treescope.render_array(probs)
# %%
for i, x in enumerate(exp_raw_x):
    print(decode(x[0]))
    imx = jnp.argmax(probs[i]).item()
    print(id2chr[imx], '::', probs[i, imx], )
    print()
# %%
