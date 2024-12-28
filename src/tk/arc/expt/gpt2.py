"""Implementation of a default jaxline experiment over the ARC dataset.
"""
from typing import cast
import jax

from tk.models.gpt2 import GPT, GPTConfig
from tk.jaxline import experiment
from tk.jaxline import utils
from tk.arc.converters import SimpleArcGridSeqEncoder
from flax.training import train_state as tslib
import jax.numpy as jnp
import optax

import datasets as hfd
import numpy as np

from ml_collections import config_dict


def create_train_state(rng, model, learning_rate, maxseq=2048):
    params = model.init(
        rng, 
        jnp.ones((1, maxseq), dtype=jnp.int32), train=False)
    tx = optax.adamw(learning_rate)
    return tslib.TrainState.create(
        apply_fn=model.apply, params=params, tx=tx)


def _forever_iter(ds: hfd.Dataset, bsiz: int, rng: jax.Array):
    """
    NOTE(tk) not sure if there's a better way to do this.
    """
    gen = np.random.default_rng(jax.device_get(rng))

    def _inner():
        while True:
            ds2 = ds.shuffle(generator=gen)
            yield from ds2.iter(bsiz)

    return _inner


class Experiment(experiment.AbstractExperiment):
    CHECKPOINT_ATTRS = {
    }
    NON_BROADCAST_CHECKPOINT_ATTRS = {
        'state': 'state',
        'cfg': 'cfg'
    }

    def __init__(self, mode: str, init_rng: jax.Array, **cfg):
        self.cfg = cfg = config_dict.ConfigDict(cfg)
        super().__init__(mode, init_rng)
        ds, vocab = SimpleArcGridSeqEncoder.load('hfd')
        r1, init_rng = jax.random.split(init_rng)
        self.dataset = ds.train_test_split(
            test_size=0.1, generator=np.random.default_rng(jax.device_get(r1))
        )
        bsiz = cfg.batch_size
        r1, r2, init_rng = jax.random.split(init_rng, 3)
        self._train = utils.py_prefetch(
            _forever_iter(self.dataset['train'], bsiz, r1),
            buffer_size=2,
        )
        self._test = utils.py_prefetch(
            _forever_iter(self.dataset['test'], bsiz, r2),
            buffer_size=2,
        )
        self.vocab = vocab
        model_cfg: GPTConfig = cast(GPTConfig, cfg.model_config)
        model_cfg = GPTConfig(**{
            **model_cfg.__dict__, 'vocab_size': len(vocab)})
        self.model = GPT(config=model_cfg)
        self.state = create_train_state(
            init_rng, self.model, cfg.lr, model_cfg.block_size)

    def step(self, *, global_step, rng, writer):
        batch = next(self._train)

        rng = utils.get_first(rng)
        global_step = np.array(utils.get_first(global_step))

        def loss_fn(params):
            logits = self.state.apply_fn(
                params, 
                batch['input_ids'], 
                rngs={'dropout': rng},
                train=True
            )
            shift_logits = logits[:, :-1]
            shift_labels = batch['input_ids'][:, 1:]
            loss = optax.softmax_cross_entropy_with_integer_labels(
                shift_logits, shift_labels)
            if 'attention_mask' in batch:
                padding_mask = batch['attention_mask'][:, 1:]
                loss = loss * padding_mask
                loss = loss.sum() / padding_mask.sum()
            else:
                loss = loss.mean()
            return loss

        loss, grads = jax.value_and_grad(loss_fn)(
            self.state.params)
        self.state = self.state.apply_gradients(grads=grads)

        metrics = dict(loss=loss.item())
        writer.write_scalars(global_step, metrics)
        return metrics
    
    def evaluate(self, *, global_step, rng, writer):
        batch = next(self._test)
        rng = utils.get_first(rng)
        global_step = np.array(utils.get_first(global_step))
        logits = self.state.apply_fn(
            self.state.params, 
            batch['input_ids'], 
            rngs={'dropout': rng},
            train=False
        )
        shift_logits = logits[:, :-1]
        shift_labels = batch['input_ids'][:, 1:]
        loss = optax.softmax_cross_entropy_with_integer_labels(
            shift_logits, shift_labels)
        if 'attention_mask' in batch:
            padding_mask = batch['attention_mask'][:, 1:]
            loss = loss * padding_mask
            loss = loss.sum() / padding_mask.sum()
        else:
            loss = loss.mean()
        return dict(loss=loss.item())

    def on_new_best_model(self, best_state):
        print("Best model!")
        print(f"{best_state=}")