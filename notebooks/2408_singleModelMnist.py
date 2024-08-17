"""WIP, almost the same objective as in singleModel but w/ MNIST digits.
"""
# %%
import gzip
import numpy as np
import matplotlib.pyplot as plt

import tk
import tk.utils as u

from pathlib import Path
from loguru import logger

from tk.utils.data.fetch import load_mnist_gzip

dataset = load_mnist_gzip(which='train')

# %%
print({k: v.shape for k, v in dataset._asdict().items()})
# %%
for i in range(10):
    plt.subplot(2, 5, i + 1)
    plt.imshow(dataset.x[i], cmap='gray')
    plt.title(f"Label: {dataset.y[i]}")
    plt.axis('off')
plt.show()
# %%
from tk.models import gpt2
from tk.utils.data.tokenizer import special_tokens

vocab = list(special_tokens.values()) + list('0123456789+=')
id2chr = {k:v for k, v in enumerate(vocab)}
chr2id = {v:k for k, v in id2chr.items()}
# %%
from collections import defaultdict

l2img = defaultdict(list)
for img, lbl in zip(dataset.x, dataset.y):
    l2img[lbl].append(img)

l2img = {k: np.stack(v) for k, v in l2img.items()}
print({k: v.shape for k, v in l2img.items()})
# %%
import jax
import itertools as it

from typing import Callable
from typing import NamedTuple


class Data(NamedTuple):
    """Batch of train-test images and labels.
    """
    fst: np.ndarray  # N x (wxh)
    snd: np.ndarray  # N x (wxh)
    fst_label: np.ndarray  # () or (N, )
    snd_label: np.ndarray  # () or (N, )
    fst_ixs: np.ndarray
    snd_ixs: np.ndarray
    txt: None | np.ndarray = None
    txt_label: None | np.ndarray = None


def create_train_test(
    l2img: dict[int, np.ndarray],
    fst_from: tuple = (0, ),
    snd_from: tuple = (0, 1, 2, 3, 4, 5, 6, 7, 8, 9),
    ntrain: int = 10,
    ntest: int = 10,
    operation: Callable = lambda *xs: sum(xs),
    rng: np.random.RandomState = np.random
) -> tuple:
    train_data = {}
    test_data = {}
    txt = np.array([chr2id['+'], chr2id['=']])[None, ...]
    
    for fst, snd in it.product(fst_from, snd_from):
        imgs_fst = l2img[fst]
        imgs_snd = l2img[snd]

        nfst = len(imgs_fst)
        nsnd = len(imgs_snd)
        label = operation(fst, snd)
        txt_label = np.array([
            chr2id[str(fst)],
            chr2id['+'],
            chr2id[str(snd)],
            chr2id['='],
            chr2id[str(label)],
        ])[None, ...]

        ixs_fst = rng.randint(0, nfst, ntrain + ntest)
        ixs_snd = rng.randint(0, nsnd, ntrain + ntest)
        train_data[label] = Data(
            imgs_fst[ixs_fst[:ntrain]], 
            imgs_snd[ixs_snd[:ntrain]],
            np.array(fst), 
            np.array(snd),
            ixs_fst[:ntrain],
            ixs_snd[:ntrain],
            txt,
            txt_label,
        )
        test_data[label] = Data(
            imgs_fst[ixs_fst[ntrain:]], 
            imgs_snd[ixs_snd[ntrain:]],
            np.array(fst), 
            np.array(snd),
            ixs_fst[ntrain:], 
            ixs_snd[ntrain:],
            txt,
            txt_label
        )
        
    return train_data, test_data


train, test = create_train_test(l2img)
print()
print(jax.tree.map(lambda xs: xs.shape, train))
print()
print(jax.tree.map(lambda xs: xs.shape, test))
# %% Sanity check correctness
import matplotlib.pyplot as plt

plt.figure(figsize=(10, 4))

for i, sum_lbl in enumerate(train):
    if i >= 6:
        break

    plt.subplot(2, 6, i * 2 + 1)
    plt.imshow(train[sum_lbl][0][0], cmap='gray')
    plt.title(f"Sum: {sum_lbl}")
    plt.axis('off')
    
    plt.subplot(2, 6, i * 2 + 2)
    plt.imshow(train[sum_lbl][1][0], cmap='gray')
    plt.axis('off')

plt.show()


# %%
import jax.numpy as jnp
import optax
from flax.training.train_state import TrainState


def create_train_state(
    rng, 
    model, 
    learning_rate, 
    input_shape=None, 
    input_shape_vis=None
):
    input_shape = input_shape or (1, model.config.block_size)
    x = jnp.ones(input_shape, dtype=jnp.int32)
    if isinstance(model, gpt2.GPTWithVision):
        input_shape_vis = input_shape_vis or (1, 1, 28, 28)
        xvis = jnp.ones(input_shape_vis, dtype=jnp.uint8)
        params = model.init(rng, txt=x, img=xvis)['params']
    else:
        params = model.init(rng, x)['params']
    tx = optax.adam(learning_rate)
    return TrainState.create(
        apply_fn=model.apply, params=params, tx=tx)


model = gpt2.GPTWithVision(gpt2.GPTConfig(
    block_size=16,
    vocab_size=len(chr2id),
    num_layers=4,
    num_heads=12,
    num_embeds=768
))
rng = jax.random.PRNGKey(42)
rng, trng = jax.random.split(rng)
train_state = create_train_state(trng, model, 1e-4)

# %%
pad_token_id = chr2id['<pad>']
bos_token_id = chr2id['<bof>']
eos_token_id = chr2id['<eof>']


def pad(seq, maxlen):
    pad = [pad_token_id] * (maxlen - 2 - len(seq))
    return [bos_token_id] + seq + [eos_token_id] + pad


def encode(seq, maxlen=None, tensorize=True):
    if isinstance(seq, np.ndarray):
        seq = seq.tolist()
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


sample = "1+1="
print(encode(sample))
print(encode(sample, 16))
print(labelize(*encode(sample, 16)))
print(decode(encode(sample)[0]))

# %%
def visual_encode(
    txt: None | jnp.ndarray | str = None,
    img: None | jnp.ndarray | tuple | list = None,
    maxlen: int = 16,
):
    if isinstance(txt, str):
        txt, mask = encode(txt, maxlen=maxlen)
    if isinstance(img, (tuple, list)):
        img = jnp.stack(img)
    return txt, img, mask

txt, img, mask = visual_encode(
    txt="+",
    img=[train[0].fst[0], train[0].snd[0]],
)
print(txt.shape, img.shape, mask.shape)
# %%
def tensorize(data: Data, i: int):
    """prepare for gpt input"""
    return jnp.stack([data.fst[i], data.snd[i]])


def tensorize_all(data: Data):
    # B x T x W x H
    X = jnp.stack([data.fst, data.snd]).swapaxes(0, 1)
    n, *_ = X.shape
    return (
        X, 
        jnp.repeat(data.txt, n, 0),
        jnp.repeat(data.txt_label, n, 0)
    )


print(tensorize(train[0], 0))
Xvis, Xtext, ytext = [], [], []
for k, v in train.items():
    a, b, c = tensorize_all(v)
    Xvis.append(a)
    Xtext.append(b)
    ytext.append(c)
Xvis = jnp.concatenate(Xvis)
Xtext = jnp.concatenate(Xtext)
ytext = jnp.concatenate(ytext)
print(Xvis.shape)
print(Xtext.shape, ytext.shape)

# %%
import numpy as np

# TODO for later
def put_in_correct_place(
    text: np.ndarray, img: np.ndarray, 
    ixs_text: np.ndarray, ixs_img: np.ndarray
) -> np.ndarray:
    max_len = max(np.max(ixs_text), np.max(ixs_img)) + 1
    result = np.empty(max_len, dtype=text.dtype)
    result[ixs_text] = text
    result[ixs_img] = img
    return result


text = np.array(['a', 'b', 'c'])
img = np.array(['x', 'y', 'z'])
ixs_text = np.array([0, 2, 4])
ixs_img = np.array([1, 3, 5])

result = put_in_correct_place(text, img, ixs_text, ixs_img)
print(result)  # Output: ['a', 'x', 'b', 'y', 'c', 'z']

# %%

model = gpt2.GPT(gpt2.GPTConfig(
    block_size=16,
    # NB output nan if vocab size is wrong
    vocab_size=max(id2chr) + 1,  # TODO figure out
    num_layers=4,
    num_heads=12,
    num_embeds=768
))
print(model.config.vocab_size)
print(Xtext)
rng = jax.random.PRNGKey(42)
rng, trng = jax.random.split(rng)
train_state = create_train_state(trng, model, 1e-4)
params = train_state.params
rng, mrng = jax.random.split(rng)
print(logits := model.apply(
    {"params": params}, 
    #jnp.array([[0, 15, 1, 0, ]]),
    Xtext[0:1, ...], 
    #img=Xvis[0:1, ...],
    rngs={'dropout': mrng}))
#print(logits.shape)
# %%
print(Xtext[0:1, ...].shape)
print(Xvis[0:1, ...].shape)

# %%
from flax import linen as nn


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


print(ytext)
print(oh_tgt := nn.one_hot(ytext, num_classes=17))
print(cross_entropy_loss(
    logits=oh_tgt * jnp.inf,
    labels=oh_tgt,
))

# %%
from jax import random


def train_step(
    x: jnp.ndarray, 
    y: jnp.ndarray, 
    mask: jnp.ndarray, 
    state: TrainState, 
    dropout_key: random.PRNGKey
) -> tuple[jnp.ndarray, TrainState]:
    x, xvis = x
    dropout_key = jax.random.fold_in(
        dropout_key, state.step)

    def loss_fn(params: dict) -> jnp.ndarray:
        logits = state.apply_fn(
            {'params': params}, 
            x, 
            img=xvis,
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


state = create_train_state(
    (rng := random.PRNGKey(42)),
    (model := gpt2.GPTWithVision(gpt2.GPTConfig(
        block_size=16,
        vocab_size=max(id2chr) + 1,
        num_embeds=128,
        num_layers=4,
        num_heads=4,
        dropout_rate=0.1,
        use_bias=True,
    ))),
    learning_rate=5e-5
)
# %%

def tensorize_all_lst(datas: dict[int, Data]):
    Xvis, Xtext, ytext = [], [], []
    for k, v in datas.items():
        a, b, c = tensorize_all(v)
        Xvis.append(a)
        Xtext.append(b)
        ytext.append(c)
    Xvis = jnp.concatenate(Xvis)
    Xtext = jnp.concatenate(Xtext)
    ytext = jnp.concatenate(ytext)
    return Xvis, Xtext, ytext

Xvis, Xtext, ytext = tensorize_all_lst(train)
xtvis, xttext, yttext = tensorize_all_lst(test)
ytext = ytext[:, [3, 4]]
yttext = yttext[:, [3, 4]]
exp = Exp(
    "visual",
    train=Enc(x=(Xtext, Xvis), y=ytext, mask=None, raw=None),
    test=Enc(x=(xttext, xtvis), y=yttext, mask=None, raw=None),
)

# %%
stat_history = []
rng, dkey = random.split(rng)
min_loss = -1, 9999, state.params
max_acc = -1, 0, state.params
# %%
class Saver:
    def __init__(self, model):
        self.crit2model = {
            'train_loss': model,
            'train_acc': model,
            'eval_acc': model,
        }
        self.crit2val = {
            'train_loss': 99999,
            'train_acc': 0,
            'eval_acc': 0,
        }

    def __call__(self, metrics, model):
        for k, v in self.crit2val.items():
            crit = (
                'acc' in k and v > metrics[k] or
                'loss' in k and v < metrics[k]
            )
            if crit:
                self.crit2val = metrics[k]
                self.crit2model = model
        return self


# %%
def eval_step(x, y, mask, model, params):
    (x, xvis) = x
    logits = model.apply(
        {'params': params}, 
        x, 
        img=xvis,
        train=False, 
        rngs={'dropout': rng}  # unused? can we skip?
    )
    yhat = logits.argmax(-1)
    metrics = dict(
        # NOTE expect "<bos>a+b="
        # prediction for next token is at ix 4
        acc = (yhat[:, 4] == y[:, 4]).mean(),
        acc_all = (yhat == y).mean(),
        loss = cross_entropy_loss(logits, y, mask)
    )
    return yhat, metrics


num_epochs = 100
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
from matplotlib.axes import Axes

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
ax3.set_ylabel('diff', color=clist[2])
ax3.tick_params(axis='y', color=clist[2], labelcolor=clist[2])
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

ax1.set_title(exp.name)
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
x, xvis = exp.train.x
train_preds = bound_model(x, xvis, train=False)
print(train_preds.shape)
# %%
# <bof>a + b = c
# 0    1 2 3 4 5
probs_train = nn.softmax(train_preds[:, 4])
print('after', len(stat_history), 'epochs')

# %%
import treescope  # comes with penzai
treescope.render_array(probs_train)

# %%
import treescope
x, xvis = exp.test.x
test_preds = bound_model(x, xvis, train=False)
print(test_preds.shape)
probs_test = nn.softmax(test_preds[:, 4])
print('after', len(stat_history), 'epochs')
treescope.render_array(probs_test)
# %%
data: Data
for label, data in test.items():
    label_convert = id2chr[data.txt_label[:, 4].item()]


# %%
