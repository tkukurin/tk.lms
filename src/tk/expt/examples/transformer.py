"""Default jaxline experiment.

Run:
    python -m tk.expt.run \
        --config=path/to/baseline.py \
        --jaxline_mode train_eval_multithread \
        --jaxline_post_mortem --jaxline_disable_jit  # for debugging
"""
import tk
import json
import os
import signal
from tk.utils.data import vocab as vlib
import itertools as it

from absl import logging

from tk.jaxline import base_config
from tk.models.gpt2 import GPTConfig

from typing import Callable, cast
import jax
from tqdm import trange

from tk.models.gpt2 import GPT, GPTConfig
from tk.jaxline import experiment
from tk.jaxline import utils
from flax.training import train_state as tslib
import jax.numpy as jnp
import optax

# import datasets as hfd
import numpy as np

from ml_collections import config_dict
from flax import linen as nn
import inspect


def get_config(debug: str = '0'):
    # NB, I think with ml_collections param has to be a string
    truthy = ('1', 'T', 'True')
    assert debug in truthy + ('0', 'F', 'False')

    def m(default_value, debug_value):
        """Convenience for debug runs."""
        return debug_value if (debug in truthy) else default_value

    cfg: config_dict.ConfigDict = base_config.get_base_config()
    cfg.debug = debug in truthy
    cfg.project_name = "[untitled]"
    cfg.experiment_class = Experiment
    with_output = True
    cfg.experiment_kwargs = dict(
        losses=dict(
            autoregressive=True,
            output=with_output,
        ),
        grammar_template=vlib._TURTLE_GRAMMAR_INTERNAL_REF,
        debug=cfg.debug,
        lr=1e-4,
        batch_size=4,
        data=dict(
            sample_synth_programs=True,
        ),
        model_config=GPTConfig(
            vocab_size=-1,
            block_size=m(64, 16),
            num_heads=8,
            num_layers=6,
            num_embeds=256,
            use_bias=True,
            output_head=3 if with_output else None,
            dtype='float16',
        ),
    )

    cfg.eval_initial_weights = False
    cfg.max_checkpoints_to_keep = 2

    cfg.training_steps = int(5e5)
    cfg.log_train_data_interval = m(50, 1)
    cfg.log_tensors_interval = m(60, 1)
    cfg.save_checkpoint_interval = m(100, 10)
    cfg.train_checkpoint_all_hosts = False
    cfg.checkpoint_dir = tk.datadir / 'outputs' / 'untitled-ckpt'
    # can set as --config.restore_path during startup 
    # NB, needs to be present in scalar_values as you return from evaluate()
    cfg.best_model_eval_metric = 'eval/loss'
    cfg.best_model_eval_metric_higher_is_better = False

    cfg.restore_path = ''

    return cfg


def create_train_state(rng, model, learning_rate, maxseq=2048):
    params = model.init(
        rng, 
        jnp.ones((1, maxseq), dtype=jnp.int32), train=False)
    tx = optax.adamw(learning_rate)
    return tslib.TrainState.create(
        apply_fn=model.apply, params=params, tx=tx)



def _forever_iter(
    ds, bsiz: int, rng: jax.Array, cfg: config_dict.ConfigDict
):
    """Get infinite stream of batched samples."""

    gen = np.random.default_rng(jax.device_get(rng))

    if isinstance(ds, Callable):
        def _inner():
            while True:
                tokenss = []
                maskss = []
                outss = []
                for _ in range(bsiz):
                    tokens, mask, outs = ds(cfg.losses.output)
                    tokenss.append(tokens)
                    maskss.append(mask)
                    outss.append(outs)
                yield {
                    'input_ids': jnp.array(tokenss),
                    'attention_mask': jnp.array(maskss),
                    'output_ids': jnp.array(outss) if cfg.losses.output else None,
                }
        return _inner

    data = jnp.array(ds['input_ids'])
    if cfg.debug:
        data = data[:bsiz]
        logging.warning(f"DEBUG: 1 batch to overfit ({data.shape=}):\n{data}")
    nsample = min(len(data), bsiz)
    def _inner():
        while True:
            idxs = gen.choice(data.shape[0], nsample, replace=False)
            yield {'input_ids': data[idxs], 'attention_mask': data[idxs]}
    return _inner


def generate(
    model: GPT,
    rng: jax.Array,
    start_tokens: jax.Array,
    max_length: int,
    top_p: float = 0.9,
    terminal_token: int | None = None,
):
    """Generates text using nucleus sampling from a pre-trained language model.

    Notes:
        - Generation stops early if the pad token is generated
        - Uses nucleus (top-p) sampling for token selection
        - Decodes token IDs to strings using global id2tok mapping
    """
    curr_ids = start_tokens
    for _ in trange(max_length, desc="Generating"):
        output = model(curr_ids)
        # NOTE(tk) assume B, T, V
        next_token = jnp.argmax(output[:, -1, :], axis=-1)
        next_token = next_token[None, ...]
        curr_ids = jnp.concatenate([curr_ids, next_token], axis=1)
        # NOTE(tk) won't work for batched
        if next_token.item() == terminal_token:
            break
    curr_ids = jax.device_get(curr_ids)
    return curr_ids


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
        seqlen = cfg.model_config.block_size
        vocab, nxt = vlib.mockgen_cfg(
            cfg.grammar_template,
            np.random.default_rng(jax.device_get(init_rng)),
            seqlen=seqlen
        )
        
        self.pad_id = vocab['<pad>']
        self.start_id = vocab['<s>']
        self.tok2id = vocab
        self.id2tok = vocab.inverse()
        assert len(self.tok2id) == len(self.id2tok)
        
        r1, r2, r3, init_rng = jax.random.split(init_rng, 4)
        bsiz = cfg.batch_size
        self._train = utils.py_prefetch(
            _forever_iter(nxt, bsiz, r2, cfg),
            buffer_size=2,
        )
        self._test = utils.py_prefetch(
            _forever_iter(nxt, bsiz, r3, cfg), 
            buffer_size=2,
        )
        # Initialize model
        model_cfg: GPTConfig = cast(GPTConfig, cfg.model_config)
        model_cfg = GPTConfig(**{
            **model_cfg.__dict__, 
            'vocab_size': len(self.tok2id),
        })
        self.model = GPT(config=model_cfg)
        self.state = create_train_state(
            init_rng, self.model, cfg.lr, model_cfg.block_size)

    def step(self, *, global_step, rng, writer):
        batch = next(self._train)

        rng = utils.get_first(rng)
        global_step = np.array(utils.get_first(global_step))

        def loss_fn(params):
            B, T = batch['input_ids'].shape
            logits = self.state.apply_fn(
                params, 
                batch['input_ids'], 
                rngs={'dropout': rng},
                train=True
            )
            if self.cfg.losses.output:
                logits, outputs = logits
            shift_logits = logits[:, :-1]
            shift_labels = batch['input_ids'][:, 1:]
            loss_all = 0
            if self.cfg.losses.autoregressive:
                loss = optax.softmax_cross_entropy_with_integer_labels(
                    shift_logits, shift_labels)
                if 'attention_mask' in batch:
                    padding_mask = batch['attention_mask'][:, 1:]
                else:
                    padding_mask = shift_labels != self.pad_id
                loss = loss * padding_mask
                loss = loss.sum() / padding_mask.sum()
                loss_all += loss
            if self.cfg.losses.output:
                output_labels = batch['output_ids']
                l = self.cfg.model_config.output_head
                loss = optax.softmax_cross_entropy_with_integer_labels(
                    outputs.reshape((B, l, -1)), output_labels)
                loss = loss.sum() / (B * l)
                loss_all += loss
            return loss_all

        loss, grads = jax.value_and_grad(loss_fn)(
            self.state.params)
        self.state = self.state.apply_gradients(grads=grads)

        metrics = {'train/loss': loss.item()}
        writer.write_scalars(global_step, metrics)
        return metrics
    
    def evaluate(self, *, global_step, rng, writer):
        if not isinstance(self.cfg, config_dict.ConfigDict):
            # TODO(tk) when becomes dict?! [prob loading]
            logging.warning(f"{type(self.cfg)=} [wrapping]")
            self.cfg = config_dict.ConfigDict(self.cfg)
        batch = next(self._test)
        rng = utils.get_first(rng)
        rng, drng, grng = jax.random.split(rng, 3)
        global_step = int(utils.get_first(global_step))
        logits = self.state.apply_fn(
            self.state.params, 
            batch['input_ids'], 
            rngs={'dropout': drng},
            train=False
        )
        if self.cfg.losses.output:
            logits, _ = logits
        shift_logits = logits[:, :-1]
        shift_labels = batch['input_ids'][:, 1:]
        loss = optax.softmax_cross_entropy_with_integer_labels(
            shift_logits, shift_labels)

        if 'attention_mask' in batch:
            padding_mask = batch['attention_mask'][:, 1:]
        else:
            padding_mask = shift_labels != self.pad_id
        loss = loss * padding_mask
        loss = loss.sum() / padding_mask.sum()
        metrics = {'eval/loss': loss.item()}
        # since eval is multithread, can cause out of sync w train global step 
        # then wandb just disregards the metric. https://wandb.me/define-metric
        writer.write_scalars(None, metrics)

        model = self.model.bind(self.state.params, rngs={'dropout': drng})
        rng, sample_rng = jax.random.split(rng)
        eval_idx = jax.random.randint(
            sample_rng, (), 0, len(batch)).item()
        prompt = batch['input_ids'][eval_idx]
        # get until end of all examples
        sep_idx = jnp.where(prompt == self.start_id)[0][0]
        prompt = prompt[None, :sep_idx+1]
        model_sample = generate(
            model,
            grng, 
            prompt,
            max_length=25,
            top_p=0.9,
            terminal_token=self.pad_id,
        )
        for in_, out_ in zip(
            jax.device_get(prompt), 
            jax.device_get(model_sample)):
            inout_str = [self.id2tok[i] for i in out_]
            in_str = inout_str[:len(in_)]
            out_str = inout_str[len(in_):]
            logging.info(f"SAMPLE STEP\nPrompt: {in_str}\nOutput: {out_str}")
            writer.add_table(
                "eval/samples",
                id=f"Eval_step{global_step}",
                prompt=in_str,
                output=out_str,
            )

        return metrics

    def on_new_best_model(self, best_state: config_dict.ConfigDict):
        del best_state  # unused, just for interface compatibility
        # HACK: we save to disk on SIGINT. we also have to wait a few
        # seconds because checkpointer.save gets called after this.
        import threading
        threading.Timer(
            10.0, lambda: os.kill(os.getpid(), signal.SIGINT)).start()
