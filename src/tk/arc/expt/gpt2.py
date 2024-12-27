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
            ds.shuffle(generator=gen)
            yield from ds.iter(bsiz)

    return _inner


class Experiment(experiment.AbstractExperiment):

    def __init__(self, mode, init_rng: jax.Array):
        # TODO(tk) config

        super().__init__(mode, init_rng)
        ds, vocab = SimpleArcGridSeqEncoder.load('hfd')
        # TODO how/where to use init_rng for shuffling
        self.dataset = ds.train_test_split(
            test_size=0.1, seed=42
        )
        bsiz = 4  # TODO cfg
        maxseq = 2048
        lr = 1e-4
        # self._train = iter(self.dataset['train'].batch(bsiz))
        r1, r2, init_rng = jax.random.split(init_rng, 3)
        self._train = utils.py_prefetch(
            _forever_iter(self.dataset['train'], bsiz, r1),
            buffer_size=2,
        )
        self._test = utils.py_prefetch(
            _forever_iter(self.dataset['test'], bsiz, r2),
            buffer_size=2,
        )
        # self._test = iter(self.dataset['test'].batch(bsiz))
        self.vocab = vocab
        self.model = GPT(config=GPTConfig(
            vocab_size=len(vocab),
            block_size=2048,
            num_heads=8,
            num_layers=6,
            num_embeds=256,
            use_bias=True,
            dtype='float32',
        ))
        self.state = create_train_state(
            init_rng, self.model, lr, maxseq)

    def step(self, *, global_step, rng, writer):
        batch = next(self._train)
        rng = utils.get_first(rng)

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

        return dict(loss=loss.item())
    
    def evaluate(self, *, global_step, rng, writer):
        batch = next(self._test)
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