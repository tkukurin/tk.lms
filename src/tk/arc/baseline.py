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
import logging

from absl import logging

from tk.jaxline import base_config
from tk.models.gpt2 import GPTConfig

from typing import cast
import jax
from tqdm import trange

from tk.models.gpt2 import GPT, GPTConfig
from tk.jaxline import experiment
from tk.jaxline import utils
from flax.training import train_state as tslib
import jax.numpy as jnp
import optax

import datasets as hfd
import numpy as np

from ml_collections import config_dict
from flax import linen as nn
from tk.arc.generator import generate_samples
import inspect
from tk.arc.p3 import dsl


def get_config(debug: str = '0'):
    # NB, I think with ml_collections param has to be a string
    truthy = ('1', 'T', 'True')
    assert debug in truthy + ('0', 'F', 'False')

    def m(default_value, debug_value):
        """Convenience for debug runs."""
        return debug_value if (debug in truthy) else default_value

    cfg = base_config.get_base_config()
    cfg.debug = debug in truthy
    cfg.project_name = "ARC-AGI"
    cfg.experiment_class = Experiment
    cfg.experiment_kwargs = dict(
        debug=cfg.debug,
        lr=1e-4,
        batch_size=4,
        data=dict(
            sample_synth_programs=True,
        ),
        model_config=GPTConfig(
            vocab_size=-1,
            block_size=512,
            num_heads=8,
            num_layers=6,
            num_embeds=256,
            use_bias=True,
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
    cfg.checkpoint_dir = tk.datadir / 'outputs' / 'arc-ckpt'
    # can set as --config.restore_path during startup 
    # NB, needs to be present in scalar_values as you return from evaluate()
    cfg.best_model_eval_metric = 'eval/loss'
    cfg.best_model_eval_metric_higher_is_better = False

    cfg.restore_path = '' #tk.datadir / 'outputs' / 'arc-ckpt' / 'models' / 'latest' / 'step...'

    return cfg


def create_train_state(rng, model, learning_rate, maxseq=2048):
    params = model.init(
        rng, 
        jnp.ones((1, maxseq), dtype=jnp.int32), train=False)
    tx = optax.adamw(learning_rate)
    return tslib.TrainState.create(
        apply_fn=model.apply, params=params, tx=tx)


def _forever_iter(
    ds, bsiz: int, rng: jax.Array, debug: bool = False):
    """Get infinite stream of batched samples.
    
    Args:
        ds: Either a HF dataset or (rng, func_arities) tuple for synthetic data
    """
    if isinstance(ds, tuple):  # TODO better than tuple
        ds_rng, vocab, seqlen, func_arities = ds
        gen = generate_samples(
            rng=np.random.default_rng(jax.device_get(ds_rng)),
            func_arities=func_arities,
            seed=('I',)
        )
        pad = vocab['<pad>']
        def _inner():
            if debug:
                batch = [list(next(gen)) for _ in range(bsiz)]
                logging.warning(f"DEBUG: 1 batch to overfit:\n{batch}")
                batch = [[vocab[x] for x in toks] for toks in batch]
                batch = [toks + [pad] * (seqlen - len(toks)) for toks in batch]
                batch = {'input_ids': jnp.array(batch)}
                while True: yield batch
            else:
                while True:
                    batch = []
                    for _ in range(bsiz):
                        tokens = next(gen)
                        batch.append([
                            vocab[t] for t in tokens
                        ] + [pad] * (seqlen - len(tokens)))
                    yield {'input_ids': jnp.array(batch)}
        return _inner

    # Original implementation for real data
    gen = np.random.default_rng(jax.device_get(rng))
    data = jnp.array(ds['input_ids'])
    if debug:
        data = data[:bsiz]
        logging.warning(f"DEBUG: 1 batch to overfit ({data.shape=}):\n{data}")
    nsample = min(len(data), bsiz)

    def _inner():
        while True:
            idxs = gen.choice(data.shape[0], nsample, replace=False)
            yield {'input_ids': data[idxs]}
    return _inner


def nucleus_sample(logits, rng, p):
    probs = nn.softmax(logits, axis=-1)
    sorted_probs = jnp.sort(probs, descending=True)
    cumsum_probs = jnp.cumsum(sorted_probs, axis=-1)
    nucleus = cumsum_probs < p
    # ensure at least the first logit is selected
    nucleus = jnp.concat(
        [jnp.ones(nucleus.shape[:-1] + (1,)), nucleus[..., :-1]], axis=-1).astype(jnp.bool)
    # doesn't work in a JIT function
    # logits_ = logits.at[~nucleus].set(float('-inf'))
    logits_ = jnp.where(~nucleus, float('-inf'), logits)
    # probs = nn.softmax(logits_, axis=-1)
    return jax.random.categorical(
        rng, logits_, axis=-1, shape=logits_.shape[:-1])

@jax.jit
def _nucleus_generate_step(output, curr_ids, rng, top_p):
    next_token = nucleus_sample(output[:, -1], rng=rng, p=top_p)
    next_token = next_token[None, ...]
    return jnp.concatenate([curr_ids, next_token], axis=1), next_token


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
        rng, rng2 = jax.random.split(rng)
        curr_ids, next_token = _nucleus_generate_step(
            output, curr_ids, rng2, top_p)
        
        if next_token.item() == terminal_token:
            break
    
    curr_ids = jax.device_get(curr_ids)
    return curr_ids


def get_func_arities(vocab: dict | None = None) -> dict[str, list[int]]:
    """Extract function arities from DSL module."""
    func_arities = {}
    for name, func in inspect.getmembers(dsl, inspect.isfunction):
        if name not in (vocab or {}):
            logging.warning(f"{name=} not in vocab")
            continue
        sig = inspect.signature(func)
        n_params = len(sig.parameters)
        # Some functions can take variable args
        if any(p.kind == inspect.Parameter.VAR_POSITIONAL 
               for p in sig.parameters.values()):
            func_arities[name] = [n_params-1, n_params, n_params+1]
        else:
            func_arities[name] = [n_params]
    return func_arities


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
        ds_dir = tk.datadir / "mhodel_rearc" / "prog_only"
        with open(ds_dir / "vocab.json", 'r') as f:
            vocab = json.load(f)
        self.program_start_id = vocab['<prog>']
        self.pad_id = vocab['<pad>']
        self.tok2id = vocab
        self.id2tok = {v: k for k, v in vocab.items() if isinstance(v, int)}
        
        r1, r2, r3, init_rng = jax.random.split(init_rng, 4)
        bsiz = cfg.batch_size
        if cfg.data.sample_synth_programs:
            # Get function arities directly from DSL
            func_arities = get_func_arities(self.tok2id)
            logging.info(f"Using {len(func_arities)} functions from DSL")
            seqlen = cfg.model_config.block_size
            self._train = utils.py_prefetch(
                _forever_iter((r1, vocab, seqlen, func_arities), bsiz, r2, cfg.debug),
                buffer_size=2,
            )
            self._test = utils.py_prefetch(
                _forever_iter((r2, vocab, seqlen, func_arities), bsiz, r3, cfg.debug), 
                buffer_size=2,
            )
        else:
            # Load real dataset
            ds = hfd.Dataset.load_from_disk(ds_dir)
            self.dataset = ds.train_test_split(
                test_size=0.1,
                generator=np.random.default_rng(jax.device_get(r1))
            )
            self._train = utils.py_prefetch(
                _forever_iter(self.dataset['train'], bsiz, r2, cfg.debug),
                buffer_size=2,
            )
            self._test = utils.py_prefetch(
                _forever_iter(self.dataset['test'], bsiz, r3, cfg.debug),
                buffer_size=2,
            )

        # Initialize model
        model_cfg: GPTConfig = cast(GPTConfig, cfg.model_config)
        model_cfg = GPTConfig(**{
            **model_cfg.__dict__, 
            'vocab_size': len(self.tok2id)
        })
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
            padding_mask = shift_labels != self.pad_id
            loss = loss * padding_mask
            loss = loss.sum() / padding_mask.sum()
            return loss

        loss, grads = jax.value_and_grad(loss_fn)(
            self.state.params)
        self.state = self.state.apply_gradients(grads=grads)

        metrics = {'train/loss': loss.item()}
        writer.write_scalars(global_step, metrics)
        return metrics
    
    def evaluate(self, *, global_step, rng, writer):
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
        shift_logits = logits[:, :-1]
        shift_labels = batch['input_ids'][:, 1:]
        loss = optax.softmax_cross_entropy_with_integer_labels(
            shift_logits, shift_labels)
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
        sep_idx = jnp.where(prompt == self.program_start_id)[0][0]
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
