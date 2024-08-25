# %%
"""WIP, almost the same objective as in singleModel but w/ MNIST digits.
"""
# %%
# %load_ext autoreload
# %autoreload 2

# %%
import tqdm
try: from rich import print as rprint
except: rprint  = print

import jax.numpy as jnp
import numpy as np
import matplotlib.pyplot as plt

import itertools as it
import functools as ft

import tk
import tk.utils as u

from pathlib import Path
from loguru import logger

from tk.utils.data.fetch import load_mnist_gzip, mnist
dataset, _ = mnist()
print({k: v.shape for k, v in dataset._asdict().items()})
# %%
import jax
print(jax.devices())

# %%
import joblib

class _Reg:
    """adhoc temp storage"""
    data = {}

    def add(self, k, v, force=False):
        if not force:
            assert k not in self.data, f'{k} exists'
        self.data[k] = v
    
    def dump(self, name: str = ''):
        import tk
        suffix = 0
        s = f'2408_{name}{suffix:03d}'
        base = tk.datadir / s
        while (out := base.with_name(s)).exists():
            suffix += 1
            s = f'2408_{name}{suffix:03d}'

        joblib.dump(self.data, out)
        return out


registry = _Reg()
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

sym2op = {
    '+': lambda a, b: a+b, 
    '-': lambda a, b: a-b,
    '*': lambda a, b: a*b,
}
exps_todo = ['+', '-', '*']
exp_tokens = [k for k in exps_todo]
# make numbers correspond to their integer encoding
vocab = list(it.chain('0123456789=', special_tokens.values(), exp_tokens))
id2chr = {k:v for k, v in enumerate(vocab)}
chr2id = {v:k for k, v in id2chr.items()}
rprint(id2chr)
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

    xtxt: np.ndarray
    ytxt: np.ndarray
    mask: np.ndarray

    op: str | None = None
    
    @property
    def xvis(self):
        stack = jnp.stack([self.fst, self.snd])
        return stack.swapaxes(0, 1)


class Op(NamedTuple):
    symbol: str
    call: Callable

    def __call__(self, *xs):
        return self.call(*xs)

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


def nonoverlap_traintest(
    fst_from: tuple, 
    snd_from: tuple, 
    operations: tuple[Op],
    train_split: float = 0.8,
    rng: np.random.RandomState = np.random,
):
    alls = set()
    for op in operations:
        combos = it.product(fst_from, snd_from)
        alls |= set(op(a, b) for a, b in combos)
    alls = list(alls)
    ixs = list(range(len(alls)))
    rng.shuffle(ixs)
    train_split = int(train_split * len(ixs))
    return alls[:train_split], alls[train_split:]


def create_train_test(
    l2img: dict[int, np.ndarray],
    fst_from: tuple = (0, ),
    snd_from: tuple = (0, 1, 2, 3, 4, 5, 6, 7, 8, 9),
    ntrain: int = 10,
    ntest: int = 10,
    operations: tuple[Op] = (
        Op('+', lambda *xs: sum(xs)),
    ),
    rng: np.random.RandomState = np.random,
    train_sums: tuple | None = None,
    test_sums: tuple | None = None,
) -> tuple:
    train_data = defaultdict(list)
    test_data = defaultdict(list)
    
    for fst, snd in it.product(fst_from, snd_from):
        imgs_fst = l2img[fst]
        imgs_snd = l2img[snd]

        nfst = len(imgs_fst)
        nsnd = len(imgs_snd)
        ixs_fst = rng.randint(0, nfst, ntrain + ntest)
        ixs_snd = rng.randint(0, nsnd, ntrain + ntest)

        for op in operations:
            label = op(fst, snd)
            xtxt, ytxt, mtxt = labelize(*encode(
                f'{op.symbol}={label}', maxlen=8
            ))
            if (not train_sums) or label in train_sums:
                train_data[label].append(Data(
                    imgs_fst[ixs_fst[:ntrain]], 
                    imgs_snd[ixs_snd[:ntrain]],
                    np.array(fst), 
                    np.array(snd),
                    ixs_fst[:ntrain],
                    ixs_snd[:ntrain],
                    xtxt=xtxt[None, ...],
                    ytxt=ytxt[None, ...],
                    mask=mtxt[None, ...],
                    op=f'{fst}{op.symbol}{snd}',
                ))
            if (not test_sums) or label in test_sums:
                test_data[label].append(Data(
                    imgs_fst[ixs_fst[ntrain:]], 
                    imgs_snd[ixs_snd[ntrain:]],
                    np.array(fst), 
                    np.array(snd),
                    ixs_fst[ntrain:], 
                    ixs_snd[ntrain:],
                    xtxt=xtxt[None, ...],
                    ytxt=ytxt[None, ...],
                    mask=mtxt[None, ...],
                    op=f'{fst}{op.symbol}{snd}',
                ))
        
    return train_data, test_data


operations=[Op(k,sym2op[k]) for k in exps_todo]
logger.info([(o.symbol, o(1, 2)) for o in operations])
# %%
fst_from: tuple = (0, )
snd_from: tuple = (0, 1, 2, 3, 4, 5, 6, 7, 8, 9)
train_sums, test_sums = nonoverlap_traintest(
    fst_from, snd_from, operations)
assert len(train_sums) > len(test_sums)
assert not set(train_sums).intersection(test_sums)
# %%
if len(set(train_sums).intersection(test_sums)) == 0:
    test_sums.append(chosen := np.random.choice(train_sums))
    print(f'Added {chosen=} to test')
print(f'{set(test_sums).intersection(train_sums)=}')
# %%
train, test = create_train_test(
    l2img, 
    fst_from=fst_from,
    snd_from=snd_from,
    operations=operations,
    train_sums=train_sums,
    test_sums=test_sums,
)
assert set(train) == set(train_sums)
assert set(test) == set(test_sums)
# %%
print()
safe_shape = lambda x: getattr(x, 'shape', None)
print(jax.tree.map(safe_shape, train))
print()
print(jax.tree.map(safe_shape, test))
# %% Sanity check correctness
import matplotlib.pyplot as plt

plt.figure(figsize=(10, 4))

for i, k in enumerate(train):
    if i >= 6:
        break

    datas = train[k][0]

    plt.subplot(2, 6, i+1)
    imgs = np.hstack([
        datas.fst[0], datas.snd[0]
    ])
    plt.imshow(imgs, cmap='gray')
    plt.title(f"{datas.op}={k}")
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

# %%

def tensorize_all_lst(datas: dict[int, Data]):
    Xvis, Xtext, ytext, mask = [], [], [], []
    for k, vs in datas.items():
        for v in vs:
            print(v.fst)
            Xvis.append(v.xvis)
            n = v.xvis.shape[0]
            Xtext.append(jnp.repeat(v.xtxt, n, axis=0))
            ytext.append(jnp.repeat(v.ytxt, n, axis=0))
            mask.append(jnp.repeat(v.mask, n, axis=0))
    Xvis = jnp.concatenate(Xvis)
    Xtext = jnp.concatenate(Xtext)
    ytext = jnp.concatenate(ytext)
    mask = jnp.concatenate(mask)
    return Xvis, Xtext, ytext, mask

Xvis, Xtext, ytext, mask = tensorize_all_lst(train)
print(Xvis.shape)
print(Xtext.shape, ytext.shape)

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
        num_embeds=768,
        num_layers=8,
        num_heads=4,
        dropout_rate=0.1,
        use_bias=True,
    ))),
    learning_rate=5e-5
)
# %%
Xvis, Xtext, ytext, Xmask = tensorize_all_lst(train)
xtvis, xttext, yttext, xtmask = tensorize_all_lst(test)
exp = Exp(
    "visual",
    train=Enc(x=(Xtext, Xvis), y=ytext, mask=Xmask, raw=None),
    test=Enc(x=(xttext, xtvis), y=yttext, mask=xtmask, raw=None),
)
# %%
stat_history = []
rng, dkey = random.split(rng)
min_loss = -1, 9999, state.params
max_acc = -1, 0, state.params
max_acc_eval = -1, 0, state.params
# %%

def eval_step(x, y, mask, params, n=None):
    """
    n specifies batchsize (y.shape[0]).
    NB, only needed for jitting because we are using `jnp.where` in the code.
    TODO maybe I can do this outside of the fn, surely can be nicer.

    cf. https://jax.readthedocs.io/en/latest/errors.html#jax.errors.ConcretizationTypeError
    """
    (x, xvis) = x
    logits = model.apply(
        {'params': params}, 
        x, 
        img=xvis,
        train=False, 
        rngs={'dropout': rng}  # unused? can we skip?
    )
    probs = nn.softmax(logits, axis=-1)
    yhat = logits.argmax(-1)
    # NOTE expect ['<bos>' '+' '=' '1' '<eos>' '<pad>]
    # (since the images are embedded as prefix)
    sel = lambda xs: xs[:, [2, 3]]
    masksel = sel(mask)
    eqs = (sel(yhat) == sel(y)) * masksel
    eqs_row = eqs.sum(-1) 
    metrics = dict(
        ixs_correct = jnp.where(
            eqs_row == masksel.sum(-1), size=n)[0],
        ixs_incorrect = jnp.where(
            eqs_row != masksel.sum(-1), size=n)[0],
        acc = eqs.sum() / masksel.sum(),
        acc_all = ((yhat == y) * mask).sum() / mask.sum(),
        loss = cross_entropy_loss(logits, y, mask)
    )
    return yhat, probs, metrics


# NB, size is needed for static compilation.
n = exp.train.y.shape[0]
train_step_jit = jax.jit(train_step)
eval_step_jit = jax.jit(ft.partial(eval_step, n=n))
# %%
from matplotlib.axes import Axes


def plot_tops(probs, ixs: tuple=(0,), ax: Axes=None):
    ax = ax or plt.gca()
    for ix in ixs:
        ax.hist(
            probs[:, ix, :].max(-1), 
            histtype='barstacked',
            alpha=.5,)
    return ax


def plot_topks(
    probs, ix: int = 3,
    ax: Axes | None = None,
    variant: str = "train"
):
    if ax is None:
        fig, ax = plt.subplots(nrows=2, ncols=1)
    topk, topk_ixs = jax.lax.top_k(probs[:, ix, :], 3)
    chars = [id2chr[i] for ixs in topk_ixs.tolist() for i in ixs]
    ax[1].hist(chars, histtype='bar', alpha=.5)
    ax[1].set_title('characters predicted')
    for i in range(3):
        ax[0].hist(
            topk[:, i],
            histtype='barstacked',
            label=f"{i}",
            alpha=.5,
        )
    ax[0].legend()
    ax[0].set_title(f"Proba distribution @ {variant}({ix})")
    fig.tight_layout()
    return fig, ax

train_yhat, train_probs, train_metrics = eval_step_jit(
    exp.train.x, exp.train.y, exp.train.mask, params=state.params)
# %%
import tk
from aim import Run, Distribution, Image, Figure, Figures, Metric, Text


def guard_run(new_instance: Run, force: bool = False) -> Run:
    if old_run := globals().get('run'):
        print(f"{old_run=}")
        if not force:
            print('NOTE: run already there, returning cached.')
            return old_run
        else:
            print('Found run, backing up and finalizing, rm with `del __run`')
            globals()['__run'] = old_run
            print(f'{old_run.finalize()=}')
    run = new_instance
    run["hparams"] = dcls.asdict(model.config)
    print(f"{run=}")
    return run


run = guard_run(Run(
    repo=tk.rootdir,
    capture_terminal_logs=True,
    experiment="singleModelMnist"
))
# %%
import pandas as pd
# install nbformat>=4.2.0 for mimetypes!
import plotly.express as px
import plotly.graph_objects as go
import matplotlib.pyplot as plt
import matplotlib.colors as mrgb

clist = list(mrgb.TABLEAU_COLORS)
from matplotlib.axes import Axes


class ProgressCtx(NamedTuple):
    name: str
    epoch: int
    step: int | None = None

    @property 
    def aim(self):
        """log for run as kwargs, `run.track(**ctx.aim)`."""
        return dict(
            context=dict(subset=self.name),
            epoch=self.epoch, 
            step=self.step)


def plot_stat_history(stat_history: list[dict]):
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
    return fig, grid


def log_everything(ctx: ProgressCtx, probs: jnp.ndarray, metrics: dict):
    ix = 3
    plot = px.histogram(df := pd.DataFrame({
        id2chr[i]: probs[:, ix, i].tolist()
        for i in range(train_probs.shape[-1])
    }), title=f"Distribution of token {ix} ({ctx.name})")
    run.track(Figure(plot), "probs", **ctx.aim)
    run.track(metrics["acc"], "accuracy", **ctx.aim)
    run.track(metrics["loss"], "loss", **ctx.aim)
    run.track(metrics["acc_all"], "accuracy over all tokens", **ctx.aim)
    return plot

# %%
train_yhat, train_probs, train_metrics = eval_step_jit(
    exp.train.x, exp.train.y, exp.train.mask, params=state.params)
eval_yhat, eval_probs, eval_metrics = eval_step_jit(
    exp.test.x, exp.test.y, exp.test.mask, params=state.params)
log_everything(ProgressCtx("train", 0, 0), train_probs, train_metrics)
log_everything(ProgressCtx("eval", 0, 0), eval_probs, eval_metrics)
# %%
num_epochs = 200
cur_epoch = len(stat_history)
with tqdm.trange(cur_epoch, cur_epoch + num_epochs) as epochs:
    for epoch in epochs:
        state, loss = train_step_jit(
            exp.train.x, exp.train.y, exp.train.mask, state, dropout_key=rng)
        data = get_stats(state)
        train_yhat, train_probs, train_metrics = eval_step_jit(
            exp.train.x, exp.train.y, exp.train.mask, params=state.params)
        eval_yhat, eval_probs, eval_metrics = eval_step_jit(
            exp.test.x, exp.test.y, exp.test.mask, params=state.params)
        data['train'] = train_metrics
        data['eval'] = eval_metrics
        with plt.ioff():
            log_everything(ProgressCtx("train", epoch), train_probs, train_metrics)
            log_everything(ProgressCtx("eval", epoch), eval_probs, eval_metrics)
        if (epoch + 1) % 200 == 0:
            fig, ax = plot_stat_history(stat_history)
            run.track(Image(fig), "stat_history", epoch=epoch)
        saved = []
        if loss < min_loss[1]:
            min_loss = (epoch, loss, state.params.copy())
        if (acc := data['train']['acc_all']) > max_acc[1]:
            max_acc = (epoch, acc, state.params.copy())
        if (acc := data['eval']['acc_all']) > max_acc_eval[1]:
            max_acc_eval = (epoch, acc, state.params.copy())
        stat_history.append(data)
        saved = f'[l]@{min_loss[1]:.2f} [at]@{max_acc[1]:.2f} [ae]{max_acc_eval[1]:.2f}'
        epochs.set_postfix_str(f"l={loss:.3f} {saved}")

# %%
fails = stat_history[-1]['eval']['ixs_incorrect']
nonfails = stat_history[-1]['eval']['ixs_correct']
print((a := len(fails)), (b := len(nonfails)), b / (a + b), stat_history[-1]['eval']['acc'])
Xvis_fails = Xvis[fails]
sh = np.array([0, 25, -10])
plt.imshow(np.hstack(np.vstack(Xvis_fails[sh])), cmap='gray')
plt.axis('off')
plt.tight_layout()
# %%
fig, ax = plot_stat_history(stat_history)
plt.show()
# %%
zipped = tree_zip(*(x['stats'] for x in stat_history[-5:]))
deltas_over_epochs = jax.tree.map(
    diff, zipped, is_leaf=lambda x: isinstance(x, (tuple, list)))

# get_chart(deltas_over_epochs)
get_chart({
    k:np.abs(v) for k, v in deltas_over_epochs.items() if 
    any(x not in ''.join(k) for x in ('ln', ))
})
# %%
get_chart({
    k:v for k, v in deltas_over_epochs.items() if 
    any(x in k for x in ('wpe', 'wte'))
})
# %%
total_change = jax.tree.reduce(
    lambda a, b: abs(a) + abs(b), deltas_over_epochs)
print(total_change)
# %%
# params = state.params
*_, params = min_loss
bound_model = model.bind(
    {'params': params}, 
    rngs={'dropout': rng})
# %%
rprint(id2chr)
# %%
import treescope  # comes with penzai
x, xvis = exp.train.x
train_preds = bound_model(x, xvis, train=False)
print(train_preds.shape)
probs_train = (
    # NOTE expect "<bos>+=1<pad>"
    nn.softmax(train_preds)[:, [2, 3]]
    #+ nn.softmax(train_preds)[:, 2]
)
print('after', len(stat_history), 'epochs')
treescope.render_array(probs_train)
# %%
import treescope
x, xvis = exp.test.x
test_preds = bound_model(x, xvis, train=False)
print(test_preds.shape)
probs_test = nn.softmax(test_preds[:, [2,3]])
print('after', len(stat_history), 'epochs')
treescope.render_array(probs_test)
# %%
probs_test_wrong = stat_history[-1]['eval']['ixs_incorrect']
#probs_test_wrong = probs_test.arg
# %%
test_y = exp.test.y[:, [2, 3]]
masksel = exp.test.mask[:, [2, 3]]
eqs_row = (test_y == probs_test.argmax(-1)).sum(-1)
eqs_row
# %%
ixs_correct = jnp.where(
    eqs_row == masksel.sum(-1))[0]
ixs_incorrect = jnp.where(
    eqs_row != masksel.sum(-1))[0]
print(ixs_correct.shape, ixs_incorrect.shape)
# %%

# %%
plot_topks(probs_test[ixs_incorrect])
# %%
plot_topks(probs_test[ixs_correct])

# %%
