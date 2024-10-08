"""Pre-training/Fine-tuning causal language modeling.

I prefer using Hydra:
```bash
base=data/outputs/prepro/summing \
python src/tk/train_hf.py \
    +data.train_file=$base/train_one_doubledigit.csv \
    +data.block_size=8 \
    +data.validation_file=$base/valid_one_doubledigit.csv \
    +train.per_device_train_batch_size=8 \
    +train.per_device_eval_batch_size=8 \
    +train.num_train_epochs=750 \
    +train.do_train=true \
    +train.do_eval=true \
    +model.config_name=$base \
    +model.tokenizer_name=$base \
    model.kwargs.n_layer=4 \
    model.kwargs.n_head=4 \
    model.kwargs.n_embd=32 \
    model.kwargs.max_len=16 \
    #
```

[based on]: https://github.com/huggingface/transformers/blob/main/examples/flax/language-modeling/run_clm_flax.py
[ckpts]: https://huggingface.co/models?filter=text-generation
"""
import hydra
from omegaconf import DictConfig

import json
import sys
from loguru import logger
import math
import os
import time
from dataclasses import asdict, dataclass, field
from enum import Enum
from itertools import chain
from pathlib import Path
from typing import Callable, Optional

import datasets
import jax
import jax.numpy as jnp

from tk.utils.log import get_logger_write_method
# https://github.com/google/jax/blob/e3e08601840e7af05daa9fdf2c5a0f56777b5c90/docs/jax_array_migration.md?plain=1#L119
# problems with huggingface typing otherwise, and other libs
# https://github.com/huggingface/transformers/issues/25417
# https://github.com/luchris429/popjaxrl/issues/1
setattr(jnp, 'DeviceArray', jax.Array)
import numpy as np
import optax
from datasets import Dataset, load_dataset
from flax import jax_utils, traverse_util
from flax.jax_utils import pad_shard_unpad, unreplicate
from flax.training import train_state
from flax.training.common_utils import get_metrics, onehot, shard, shard_prng_key
from huggingface_hub import HfApi
from tqdm import tqdm

import transformers
from transformers import (
    CONFIG_MAPPING,
    FLAX_MODEL_FOR_CAUSAL_LM_MAPPING,
    AutoConfig,
    AutoTokenizer,
    FlaxAutoModelForCausalLM,
    is_tensorboard_available,
    set_seed,
)
from transformers.testing_utils import CaptureLogger


MODEL_CONFIG_CLASSES = list(FLAX_MODEL_FOR_CAUSAL_LM_MAPPING.keys())
MODEL_TYPES = tuple(conf.model_type for conf in MODEL_CONFIG_CLASSES)


@dataclass
class TrainingArguments:
    output_dir: None | str = field(
        default=None,
        metadata={"help": (
            "The output directory where the model predictions and checkpoints will be written. "
            "If unset, we'll use `cwd()` (NB, Hydra sets this to a version-controlled directory!)"
        )},
    )
    overwrite_output_dir: bool = field(
        default=False,
        metadata={
            "help": (
                "Overwrite the content of the output directory. "
                "Use this to continue training if output_dir points to a checkpoint directory."
            )
        },
    )
    do_train: bool = field(default=False, metadata={"help": "Whether to run training."})
    do_eval: bool = field(default=False, metadata={"help": "Whether to run eval on the dev set."})
    per_device_train_batch_size: int = field(
        default=8, metadata={"help": "Batch size per GPU/TPU core/CPU for training."}
    )
    per_device_eval_batch_size: int = field(
        default=8, metadata={"help": "Batch size per GPU/TPU core/CPU for evaluation."}
    )
    learning_rate: float = field(default=5e-5, metadata={"help": "The initial learning rate for AdamW."})
    weight_decay: float = field(default=0.0, metadata={"help": "Weight decay for AdamW if we apply some."})
    adam_beta1: float = field(default=0.9, metadata={"help": "Beta1 for AdamW optimizer"})
    adam_beta2: float = field(default=0.999, metadata={"help": "Beta2 for AdamW optimizer"})
    adam_epsilon: float = field(default=1e-8, metadata={"help": "Epsilon for AdamW optimizer."})
    adafactor: bool = field(default=False, metadata={"help": "Whether or not to replace AdamW by Adafactor."})
    num_train_epochs: float = field(default=3.0, metadata={"help": "Total number of training epochs to perform."})
    warmup_steps: int = field(default=0, metadata={"help": "Linear warmup over warmup_steps."})
    logging_steps: int = field(default=500, metadata={"help": "Log every X updates steps."})
    save_steps: int = field(default=500, metadata={"help": "Save checkpoint every X updates steps."})
    eval_steps: int = field(default=None, metadata={"help": "Run an evaluation every X steps."})
    seed: int = field(default=42, metadata={"help": "Random seed that will be set at the beginning of training."})
    push_to_hub: bool = field(
        default=False, metadata={"help": "Whether or not to upload the trained model to the model hub after training."}
    )
    hub_model_id: str = field(
        default=None, metadata={"help": "The name of the repository to keep in sync with the local `output_dir`."}
    )
    hub_token: str = field(default=None, metadata={"help": "The token to use to push to the Model Hub."})

    def __post_init__(self):
        if self.output_dir is not None:
            self.output_dir = os.path.expanduser(self.output_dir)

    def to_dict(self):
        """
        Serializes this instance while replace `Enum` by their values (for JSON serialization support). It obfuscates
        the token values by removing their value.
        """
        d = asdict(self)
        for k, v in d.items():
            if isinstance(v, Enum):
                d[k] = v.value
            if isinstance(v, list) and len(v) > 0 and isinstance(v[0], Enum):
                d[k] = [x.value for x in v]
            if k.endswith("_token"):
                d[k] = f"<{k.upper()}>"
        return d


@dataclass
class ModelArguments:
    model_name_or_path: Optional[str] = field(
        default=None,
        metadata={
            "help": (
                "The model checkpoint for weights initialization. Don't set if you want to train a model from scratch."
            )
        },
    )
    model_type: Optional[str] = field(
        default=None,
        metadata={"help": "If training from scratch, pass a model type from the list: " + ", ".join(MODEL_TYPES)},
    )
    config_name: Optional[str] = field(
        default=None, metadata={"help": "Pretrained config name or path if not the same as model_name"}
    )
    tokenizer_name: Optional[str] = field(
        default=None, metadata={"help": "Pretrained tokenizer name or path if not the same as model_name"}
    )
    cache_dir: Optional[str] = field(
        default=None, metadata={"help": "Where do you want to store the pretrained models downloaded from s3"}
    )
    use_fast_tokenizer: bool = field(
        default=True,
        metadata={"help": "Whether to use one of the fast tokenizer (backed by the tokenizers library) or not."},
    )
    dtype: Optional[str] = field(
        default="float32",
        metadata={
            "help": (
                "Floating-point format in which the model weights should be initialized and trained. Choose one of"
                " `[float32, float16, bfloat16]`."
            )
        },
    )
    token: str = field(
        default=None,
        metadata={
            "help": (
                "The token to use as HTTP bearer authorization for remote files. If not specified, will use the token "
                "generated when running `huggingface-cli login` (stored in `~/.huggingface`)."
            )
        },
    )
    trust_remote_code: bool = field(
        default=False,
        metadata={
            "help": (
                "Whether to trust the execution of code from datasets/models defined on the Hub."
                " This option should only be set to `True` for repositories you trust and in which you have read the"
                " code, as it will execute code present on the Hub on your local machine."
            )
        },
    )
    kwargs: dict = field(default_factory=dict, metadata={
        "help": "passed to autoconfig"
    })


@dataclass
class DataTrainingArguments:
    dataset_name: Optional[str] = field(
        default=None, metadata={"help": "The name of the dataset to use (via the datasets library)."}
    )
    dataset_config_name: Optional[str] = field(
        default=None, metadata={"help": "The configuration name of the dataset to use (via the datasets library)."}
    )
    train_file: Optional[str] = field(default=None, metadata={"help": "The input training data file (a text file)."})
    validation_file: Optional[str] = field(
        default=None,
        metadata={"help": "An optional input evaluation data file to evaluate the perplexity on (a text file)."},
    )
    max_train_samples: Optional[int] = field(
        default=None,
        metadata={
            "help": (
                "For debugging purposes or quicker training, truncate the number of training examples to this "
                "value if set."
            )
        },
    )
    max_eval_samples: Optional[int] = field(
        default=None,
        metadata={
            "help": (
                "For debugging purposes or quicker training, truncate the number of evaluation examples to this "
                "value if set."
            )
        },
    )
    validation_split_percentage: Optional[int] = field(
        default=5,
        metadata={
            "help": "The percentage of the train set used as validation set in case there's no validation split"
        },
    )
    block_size: Optional[int] = field(
        default=None,
        metadata={
            "help": (
                "Optional input sequence length after tokenization. "
                "The training dataset will be truncated in block of this size for training. "
                "Default to the model max input length for single sentence inputs (take into account special tokens)."
            )
        },
    )
    overwrite_cache: bool = field(
        default=False, metadata={"help": "Overwrite the cached training and evaluation sets"}
    )
    preprocessing_num_workers: Optional[int] = field(
        default=None,
        metadata={"help": "The number of processes to use for the preprocessing."},
    )
    keep_linebreaks: bool = field(
        default=True, metadata={"help": "Whether to keep line breaks when using TXT files or not."}
    )

    def __post_init__(self):
        if self.dataset_name is None and self.train_file is None and self.validation_file is None:
            raise ValueError("Need either a dataset name or a training/validation file.")
        else:
            if self.train_file is not None:
                extension = self.train_file.split(".")[-1]
                if extension not in ["csv", "json", "txt"]:
                    raise ValueError("train_file` should be a csv, json or text file.")
            if self.validation_file is not None:
                extension = self.validation_file.split(".")[-1]
                if extension not in ["csv", "json", "txt"]:
                    raise ValueError("`validation_file` should be a csv, json or text file.")


class TrainState(train_state.TrainState):
    dropout_rng: jnp.ndarray

    def replicate(self):
        return jax_utils.replicate(self).replace(dropout_rng=shard_prng_key(self.dropout_rng))


def data_loader(rng: jax.random.PRNGKey, dataset: Dataset, batch_size: int, shuffle: bool = False, drop_last=True):
    """
    Returns batches of size `batch_size` from `dataset`. If `drop_last` is set to `False`, the final batch may be incomplete,
    and range in size from 1 to `batch_size`. Shuffle batches if `shuffle` is `True`.
    """
    if shuffle:
        batch_idx = jax.random.permutation(rng, len(dataset))
        batch_idx = np.asarray(batch_idx)
    else:
        batch_idx = np.arange(len(dataset))

    if drop_last:
        steps_per_epoch = len(dataset) // batch_size
        batch_idx = batch_idx[: steps_per_epoch * batch_size]  # Skip incomplete batch.
        batch_idx = batch_idx.reshape((steps_per_epoch, batch_size))
    else:
        steps_per_epoch = math.ceil(len(dataset) / batch_size)
        batch_idx = np.array_split(batch_idx, steps_per_epoch)

    for idx in batch_idx:
        batch = dataset[idx]
        batch = {k: np.array(v) for k, v in batch.items()}

        yield batch


def write_train_metric(add, train_metrics, train_time, step):
    add("train_time", train_time, step)
    train_metrics = get_metrics(train_metrics)
    for key, vals in train_metrics.items():
        tag = f"train_{key}"
        for i, val in enumerate(vals):
            add(tag, val, step - len(vals) + i + 1)


def write_eval_metric(add, eval_metrics, step):
    for metric_name, value in eval_metrics.items():
        add(f"eval_{metric_name}", value, step)


def create_learning_rate_fn(
    train_ds_size: int, 
    train_batch_size: int, 
    num_train_epochs: int, 
    num_warmup_steps: int,
    learning_rate: float
) -> Callable[[int], jnp.ndarray]:
    """Returns a linear warmup, linear_decay learning rate function."""
    steps_per_epoch = train_ds_size // train_batch_size
    num_train_steps = steps_per_epoch * num_train_epochs
    warmup_fn = optax.linear_schedule(
        init_value=0.0,
        end_value=learning_rate,
        transition_steps=num_warmup_steps)
    decay_fn = optax.linear_schedule(
        init_value=learning_rate, end_value=0, transition_steps=num_train_steps - num_warmup_steps
    )
    schedule_fn = optax.join_schedules(schedules=[warmup_fn, decay_fn], boundaries=[num_warmup_steps])
    return schedule_fn


def init(o, kls):
    def _name(klass):
        module = klass.__module__
        if module == 'builtins':
            return klass.__qualname__ # avoid outputs like 'builtins.str'
        return module + '.' + klass.__qualname__
    return hydra.utils.instantiate(o, _target_=_name(kls))


@hydra.main(
    version_base="1.3", 
    config_path="configs", 
    config_name="train.yaml")
def main(cfg: DictConfig):
    model_args: ModelArguments = init(cfg.get('model', {}), ModelArguments)
    data_args: DataTrainingArguments = init(cfg.get('data', {}), DataTrainingArguments)
    training_args: TrainingArguments = init(cfg.get('train', {}), TrainingArguments)

    orig_cwd = Path(hydra.utils.get_original_cwd())
    output_dir = training_args.output_dir or os.getcwd()
    logger.info(f"Working dir: {output_dir}")
    logger.info(f"Original dir: {orig_cwd}")

    # HACK: usually we want to load relative to original cwd
    # but save relative to the versioned output dir
    # => we revert back before training
    os.chdir(orig_cwd)

    if jax.process_index() == 0:
        datasets.utils.logging.set_verbosity_warning()
        transformers.utils.logging.set_verbosity_info()
    else:
        datasets.utils.logging.set_verbosity_error()
        transformers.utils.logging.set_verbosity_error()
        logger.remove()  # TODO not sure if this works...
        logger.add(sys.stderr, level="ERROR")

    logger.info(f"Training/evaluation parameters\n{training_args}")

    set_seed(training_args.seed)

    if training_args.push_to_hub:
        repo_name = training_args.hub_model_id
        if repo_name is None:
            repo_name = Path(output_dir).absolute().name
        api = HfApi()
        repo_id = api.create_repo(repo_name, exist_ok=True, token=training_args.hub_token).repo_id

    # load_dataset guarantees one local process can concurrently download
    if data_args.dataset_name is not None:
        dataset = load_dataset(
            data_args.dataset_name,
            data_args.dataset_config_name,
            cache_dir=model_args.cache_dir,
            keep_in_memory=False,
            token=model_args.token,
            num_proc=data_args.preprocessing_num_workers,
            trust_remote_code=model_args.trust_remote_code,
        )

        if "validation" not in dataset.keys():
            dataset["validation"] = load_dataset(
                data_args.dataset_name,
                data_args.dataset_config_name,
                split=f"train[:{data_args.validation_split_percentage}%]",
                cache_dir=model_args.cache_dir,
                token=model_args.token,
                num_proc=data_args.preprocessing_num_workers,
                trust_remote_code=model_args.trust_remote_code,
            )
            dataset["train"] = load_dataset(
                data_args.dataset_name,
                data_args.dataset_config_name,
                split=f"train[{data_args.validation_split_percentage}%:]",
                cache_dir=model_args.cache_dir,
                token=model_args.token,
                num_proc=data_args.preprocessing_num_workers,
                trust_remote_code=model_args.trust_remote_code,
            )
    else:
        data_files = {}
        dataset_args = {}
        if data_args.train_file is not None:
            data_files["train"] = data_args.train_file
            extension = data_args.train_file.split(".")[-1]
        if data_args.validation_file is not None:
            data_files["validation"] = data_args.validation_file
            extension = data_args.validation_file.split(".")[-1]
        if extension == "txt":
            extension = "text"
            dataset_args["keep_linebreaks"] = data_args.keep_linebreaks
        dataset = load_dataset(
            extension,
            data_files=data_files,
            cache_dir=model_args.cache_dir,
            **dataset_args,
            token=model_args.token,
            num_proc=data_args.preprocessing_num_workers,
        )

        if "validation" not in dataset.keys():
            dataset["validation"] = load_dataset(
                extension,
                data_files=data_files,
                split=f"train[:{data_args.validation_split_percentage}%]",
                cache_dir=model_args.cache_dir,
                **dataset_args,
                token=model_args.token,
                num_proc=data_args.preprocessing_num_workers,
            )
            dataset["train"] = load_dataset(
                extension,
                data_files=data_files,
                split=f"train[{data_args.validation_split_percentage}%:]",
                cache_dir=model_args.cache_dir,
                **dataset_args,
                token=model_args.token,
                num_proc=data_args.preprocessing_num_workers,
            )

    # See more about loading any type of standard or custom dataset (from files, python dict, pandas DataFrame, etc) at
    # https://huggingface.co/docs/datasets/loading_datasets.

    # Load pretrained model and tokenizer

    # .from_pretrained guarantees 1 local process can concurrently
    # download model & vocab.
    if model_args.config_name:
        config = AutoConfig.from_pretrained(
            model_args.config_name,
            cache_dir=model_args.cache_dir,
            token=model_args.token,
            trust_remote_code=model_args.trust_remote_code,
            **model_args.kwargs,
        )
    elif model_args.model_name_or_path:
        config = AutoConfig.from_pretrained(
            model_args.model_name_or_path,
            cache_dir=model_args.cache_dir,
            token=model_args.token,
            trust_remote_code=model_args.trust_remote_code,
            **model_args.kwargs,
        )
    else:
        config = CONFIG_MAPPING[model_args.model_type]()
        logger.warning("You are instantiating a new config instance from scratch.")

    if model_args.tokenizer_name:
        tokenizer = AutoTokenizer.from_pretrained(
            model_args.tokenizer_name,
            cache_dir=model_args.cache_dir,
            use_fast=model_args.use_fast_tokenizer,
            token=model_args.token,
            trust_remote_code=model_args.trust_remote_code,
        )
    elif model_args.model_name_or_path:
        tokenizer = AutoTokenizer.from_pretrained(
            model_args.model_name_or_path,
            cache_dir=model_args.cache_dir,
            use_fast=model_args.use_fast_tokenizer,
            token=model_args.token,
            trust_remote_code=model_args.trust_remote_code,
        )
    else:
        raise ValueError(
            "You are instantiating a new tokenizer from scratch. This is not supported by this script. "
            "You can do it from another script, save it, and load it from here, using --tokenizer_name."
        )

    if model_args.model_name_or_path:
        model = FlaxAutoModelForCausalLM.from_pretrained(
            model_args.model_name_or_path,
            config=config,
            seed=training_args.seed,
            dtype=getattr(jnp, model_args.dtype),
            token=model_args.token,
            trust_remote_code=model_args.trust_remote_code,
        )
    else:
        model = FlaxAutoModelForCausalLM.from_config(
            config,
            seed=training_args.seed,
            dtype=getattr(jnp, model_args.dtype),
            trust_remote_code=model_args.trust_remote_code,
        )

    logger.info(f"Instantiated model:\n{model}")
    logger.info(f"Instantiated config:\n{config}")

    if training_args.do_train:
        column_names = dataset["train"].column_names
    else:
        column_names = dataset["validation"].column_names
    text_column_name = "text" if "text" in column_names else column_names[0]

    # since this will be pickled to avoid _LazyModule error in Hasher force
    # logger loading before tokenize_function
    tok_logger = transformers.utils.logging.get_logger(
        "transformers.tokenization_utils_base")

    def tokenize_function(examples):
        with CaptureLogger(tok_logger) as cl:
            output = tokenizer(examples[text_column_name])
        # clm input could be much much longer than block_size
        if "Token indices sequence length is longer than the" in cl.out:
            tok_logger.warning(
                "^^^^^^^^^^^^^^^^ Please ignore the warning above - this long input will be chunked into smaller bits"
                " before being passed to the model."
            )
        return output

    tokenized_datasets = dataset.map(
        tokenize_function,
        batched=True,
        num_proc=data_args.preprocessing_num_workers,
        remove_columns=column_names,
        load_from_cache_file=not data_args.overwrite_cache,
    )

    if data_args.block_size is None:
        block_size = tokenizer.model_max_length
        if block_size > config.max_position_embeddings:
            logger.warning(
                f"The tokenizer picked seems to have a very large `model_max_length` ({tokenizer.model_max_length}). "
                f"Using block_size={min(1024, config.max_position_embeddings)} instead. You can change that default value by passing --block_size xxx."
            )
            block_size = min(1024, config.max_position_embeddings)
    else:
        if data_args.block_size > tokenizer.model_max_length:
            logger.warning(
                f"The block_size passed ({data_args.block_size}) is larger than the maximum length for the model "
                f"({tokenizer.model_max_length}). Using block_size={tokenizer.model_max_length}."
            )
        block_size = min(data_args.block_size, tokenizer.model_max_length)

    def group_texts(examples):
        concatenated_examples = {k: list(chain(*examples[k])) for k in examples.keys()}
        total_length = len(concatenated_examples[list(examples.keys())[0]])
        if total_length >= block_size:
            total_length = (total_length // block_size) * block_size
        result = {  # block_$ize chunks
            k: [t[i : i + block_size] for i in range(0, total_length, block_size)]
            for k, t in concatenated_examples.items()
        }
        result["labels"] = result["input_ids"].copy()
        return result

    # https://huggingface.co/docs/datasets/process#map

    lm_datasets = tokenized_datasets.map(
        group_texts,
        batched=True,
        num_proc=data_args.preprocessing_num_workers,
        load_from_cache_file=not data_args.overwrite_cache,
    )

    if training_args.do_train:
        if "train" not in tokenized_datasets:
            raise ValueError("--do_train requires a train dataset")
        train_dataset = lm_datasets["train"]
        if data_args.max_train_samples is not None:
            max_train_samples = min(
                len(train_dataset), data_args.max_train_samples)
            train_dataset = train_dataset.select(range(max_train_samples))

    if training_args.do_eval:
        if "validation" not in tokenized_datasets:
            raise ValueError("--do_eval requires a validation dataset")
        eval_dataset = lm_datasets["validation"]
        if data_args.max_eval_samples is not None:
            max_eval_samples = min(
                len(eval_dataset), data_args.max_eval_samples)
            eval_dataset = eval_dataset.select(range(max_eval_samples))

    if jax.process_index() == 0:
        track = get_logger_write_method(output_dir)

    rng = jax.random.PRNGKey(training_args.seed)
    rng, dropout_rng = jax.random.split(rng)

    start_epoch = 0
    if 'epoch' in (model_args.model_name_or_path or ''):
        import re
        start_epoch = 1 + int(re.search(r'epoch=(\d+)', model_args.model_name_or_path)[1])
        logger.info(f'Found existing run, setting epoch={start_epoch}')
    num_epochs = int(training_args.num_train_epochs)
    train_batch_size = int(training_args.per_device_train_batch_size) * jax.device_count()
    per_device_eval_batch_size = int(training_args.per_device_eval_batch_size)
    eval_batch_size = per_device_eval_batch_size * jax.device_count()
    steps_per_epoch = len(train_dataset) // train_batch_size
    total_train_steps = steps_per_epoch * num_epochs

    # Create learning rate schedule
    linear_decay_lr_schedule_fn = create_learning_rate_fn(
        len(train_dataset),
        train_batch_size,
        training_args.num_train_epochs,
        training_args.warmup_steps,
        training_args.learning_rate,
    )

    def decay_mask_fn(params):
        flat_params = traverse_util.flatten_dict(params)
        layer_norm_candidates = ["layernorm", "layer_norm", "ln"]
        layer_norm_named_params = {
            layer[-2:]
            for layer_norm_name in layer_norm_candidates
            for layer in flat_params.keys()
            if layer_norm_name in "".join(layer).lower()
        }
        flat_mask = {
            path: (  # True if param _should_ be decayed
                path[-1] != "bias" 
                and path[-2:] not in layer_norm_named_params)
            for path in flat_params
        }
        return traverse_util.unflatten_dict(flat_mask)

    if training_args.adafactor:
        optimizer = optax.adafactor(
            learning_rate=linear_decay_lr_schedule_fn,
        )
    else:
        optimizer = optax.adamw(
            learning_rate=linear_decay_lr_schedule_fn,
            b1=training_args.adam_beta1,
            b2=training_args.adam_beta2,
            eps=training_args.adam_epsilon,
            weight_decay=training_args.weight_decay,
            mask=decay_mask_fn,
        )

    state = TrainState.create(
        apply_fn=model.__call__,
        params=model.params,
        tx=optimizer,
        dropout_rng=dropout_rng)

    def loss_fn(logits, labels):
        shift_logits = logits[..., :-1, :]
        shift_labels = labels[..., 1:]
        loss = optax.softmax_cross_entropy(shift_logits, onehot(shift_labels, shift_logits.shape[-1]))
        return loss.mean()

    def train_step(state, batch):
        dropout_rng, new_dropout_rng = jax.random.split(state.dropout_rng)

        def compute_loss(params):
            labels = batch.pop("labels")
            logits = state.apply_fn(**batch, params=params, dropout_rng=dropout_rng, train=True)[0]
            loss = loss_fn(logits, labels)
            return loss

        grad_fn = jax.value_and_grad(compute_loss)
        loss, grad = grad_fn(state.params)
        grad = jax.lax.pmean(grad, "batch")

        new_state = state.apply_gradients(grads=grad, dropout_rng=new_dropout_rng)

        metrics = {"loss": loss, "learning_rate": linear_decay_lr_schedule_fn(state.step)}
        metrics = jax.lax.pmean(metrics, axis_name="batch")

        return new_state, metrics

    def eval_step(params, batch):
        labels = batch.pop("labels")
        logits = model(**batch, params=params, train=False)[0]
        loss = loss_fn(logits, labels)

        metrics = {"loss": loss}
        metrics = jax.lax.pmean(metrics, axis_name="batch")
        return metrics

    # Create parallel version of the train and eval step
    p_train_step = jax.pmap(train_step, "batch", donate_argnums=(0,))
    p_eval_step = jax.pmap(eval_step, "batch")

    # Replicate the train state on each device
    state = state.replicate()

    logger.info("***** Running training *****")
    logger.info(f"  Num examples = {len(train_dataset)}")
    logger.info(f"  Num Epochs = {num_epochs}")
    logger.info(f"  Instantaneous batch size per device = {training_args.per_device_train_batch_size}")
    logger.info(f"  Total train batch size (w. parallel & distributed) = {train_batch_size}")
    logger.info(f"  Total optimization steps = {total_train_steps}")

    train_time = 0
    train_metrics = []
    epochs = tqdm(range(start_epoch, start_epoch + num_epochs), desc="Epoch ... ", position=0)

    steps_per_epoch = len(train_dataset) // train_batch_size
    if training_args.eval_steps is None:
        training_args.eval_steps = steps_per_epoch

    os.chdir(output_dir)
    for epoch in epochs:
        train_start = time.time()

        rng, input_rng = jax.random.split(rng)
        train_loader = data_loader(
            input_rng, train_dataset, train_batch_size, shuffle=True)
        for step in tqdm(range(steps_per_epoch), desc="Training...", position=1, leave=False):
            batch = next(train_loader)
            batch = shard(batch)
            state, train_metric = p_train_step(state, batch)
            train_metrics.append(train_metric)

            cur_step = epoch * (len(train_dataset) // train_batch_size) + step

            if cur_step % training_args.logging_steps == 0 and cur_step > 0:
                train_metric = unreplicate(train_metric)
                train_time += time.time() - train_start
                if track and jax.process_index() == 0:
                    write_train_metric(
                        track, train_metrics, train_time, cur_step)

                epochs.write(
                    f"Step... ({cur_step} | Loss: {train_metric['loss'].mean()}, Learning Rate:"
                    f" {train_metric['learning_rate'].mean()})"
                )

                train_metrics = []

            if cur_step % training_args.eval_steps == 0 and cur_step > 0:
                eval_metrics = []
                eval_loader = data_loader(input_rng, eval_dataset, eval_batch_size, drop_last=False)
                eval_steps = math.ceil(len(eval_dataset) / eval_batch_size)
                for _ in tqdm(range(eval_steps), desc="Evaluating...", position=2, leave=False):
                    # Model forward
                    batch = next(eval_loader)
                    metrics = pad_shard_unpad(p_eval_step, static_return=True)(
                        state.params, batch, min_device_batch=per_device_eval_batch_size
                    )
                    eval_metrics.append(metrics)

                # normalize eval metrics
                eval_metrics = get_metrics(eval_metrics)
                eval_metrics = jax.tree_util.tree_map(jnp.mean, eval_metrics)

                try:
                    eval_metrics["perplexity"] = math.exp(eval_metrics["loss"])
                except OverflowError:
                    eval_metrics["perplexity"] = float("inf")

                # Print metrics and update progress bar
                desc = (
                    f"Step... ({cur_step} | Eval Loss: {eval_metrics['loss']} | Eval Perplexity:"
                    f" {eval_metrics['perplexity']})"
                )
                epochs.write(desc)
                epochs.desc = desc
                if summary_writer and jax.process_index() == 0:
                    write_eval_metric(track, eval_metrics, cur_step)

            if cur_step % training_args.save_steps == 0 and cur_step > 0 and jax.process_index() == 0:
                params = jax.device_get(unreplicate(state.params))
                outdir = f"{output_dir}/step={cur_step}"
                Path(outdir).mkdir(parents=True, exist_ok=True)
                model.save_pretrained(outdir, params=params)
                tokenizer.save_pretrained(outdir)
                if training_args.push_to_hub:
                    api.upload_folder(
                        commit_message=f"Saving weights and logs of step {cur_step}",
                        folder_path=outdir,
                        repo_id=repo_id,
                        repo_type="model",
                        token=training_args.hub_token,
                    )
    if jax.process_index() == 0:  # save
        params = jax.device_get(unreplicate(state.params))
        outdir = f"{output_dir}/epoch={epoch}"
        Path(outdir).mkdir(parents=True, exist_ok=True)
        model.save_pretrained(outdir, params=params)
        tokenizer.save_pretrained(outdir)
        if training_args.push_to_hub:
            api.upload_folder(
                commit_message=f"Saving weights and logs of step {cur_step}",
                folder_path=outdir,
                repo_id=repo_id,
                repo_type="model",
                token=training_args.hub_token,
            )
    # Eval after training
    if training_args.do_eval:
        eval_metrics = []
        eval_loader = data_loader(input_rng, eval_dataset, eval_batch_size, drop_last=False)
        eval_steps = math.ceil(len(eval_dataset) / eval_batch_size)
        for _ in tqdm(range(eval_steps), desc="Evaluating...", position=2, leave=False):
            # Model forward
            batch = next(eval_loader)
            metrics = pad_shard_unpad(p_eval_step, static_return=True)(
                state.params, batch, min_device_batch=per_device_eval_batch_size
            )
            eval_metrics.append(metrics)

        # normalize eval metrics
        eval_metrics = get_metrics(eval_metrics)
        eval_metrics = jax.tree_util.tree_map(lambda x: jnp.mean(x).item(), eval_metrics)

        try:
            eval_metrics["perplexity"] = math.exp(eval_metrics["loss"])
        except OverflowError:
            eval_metrics["perplexity"] = float("inf")

        if jax.process_index() == 0:
            Path(output_dir).mkdir(
                parents=True, exist_ok=True)
            eval_metrics = {f"eval_{metric_name}": value for metric_name, value in eval_metrics.items()}
            path = os.path.join(output_dir, "eval_results.json")
            with open(path, "w") as f:
                json.dump(eval_metrics, f, indent=4, sort_keys=True)


if __name__ == "__main__":
    main()