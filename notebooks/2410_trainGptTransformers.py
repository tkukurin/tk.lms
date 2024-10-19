"""Train GPT via Transformers and Flax.
"""
# %%
import jax
import jax.numpy as jnp
import optax
from transformers import FlaxGPT2LMHeadModel, GPT2Tokenizer, GPT2Config

# tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
# tokenizer.pad_token = tokenizer.eos_token
data = [
    "def my_function(overfit=True):\n    \"hello here I am overfitting as hell\"\n    return 1",
    "def my_other_function(overfit=False):\n    return 0"
]
# %%
from tk.utils.data import tokenizer as toklib
import importlib
import string
importlib.reload(toklib)
cfg, tokenizer = toklib.mktokenizer(
    string.printable + string.whitespace,
    max_len=64,
    return_og_tokenizer=False,
)
print(tokenizer.vocab_size, len(tokenizer.get_vocab()))
# %%
print(tokenizer("def test").input_ids)
print(tokenizer("def test def test def test", max_length=6, return_overflowing_tokens=True).input_ids)
print(tokenizer("def test<|eof|>def test def test", max_length=6, return_overflowing_tokens=True).input_ids)
print("EOS:", tokenizer(tokenizer.eos_token, return_overflowing_tokens=True).input_ids)
# TODO figure out why this breaks :(
# inuition/hypothesis: pre-tokenizer should be fixed.
# use some regex to prefer recog of special tokens
print("EOS:", tokenizer(f"as{tokenizer.eos_token}df{tokenizer.eos_token}asdfasdf", return_overflowing_tokens=True).input_ids)
# %%
print(tokenizer("def test", add_special_tokens=False).input_ids)
print(tokenizer.decode(tokenizer("def test", add_special_tokens=False).input_ids))
print(tokenizer.decode(tokenizer("def test").input_ids, skip_special_tokens=False))
print(tokenizer.decode(tokenizer("def test").input_ids, skip_special_tokens=True))
print(type(tokenizer))
# %%
inputs = tokenizer(
    tokenizer.eos_token.join(data),
    return_tensors="jax",
    padding=False,#"max_length",
    truncation=False,
    max_length=None,
    # stride=16,
    # return_overflowing_tokens=True,
    # add_special_tokens=False,
)

print(inputs.input_ids.shape)
for i in inputs.input_ids[:3]:
    print()
    print(tokenizer.convert_ids_to_tokens(i[:10]))
    print(tokenizer.convert_ids_to_tokens(i[-10:]))
# %%
print(inputs.input_ids.max())
# NB, source of nasty bugs!!
print(len(tokenizer.get_vocab()), tokenizer.vocab_size)
# %%
def get_batch(
    data,
    rng: jax.Array,
    batch_size: int = 8,
    block_size: int = tokenizer.model_max_length,
    return_meta: bool = False,
) -> jnp.ndarray | tuple[jnp.ndarray, dict]:
    # NB, in theory when dealing with small data this is actually really important
    # i.e. we will never sample (len(data) - block_size) to len(data)
    ix = jax.random.randint(
        rng, shape=(batch_size,), minval=0, maxval=len(data) - block_size)
    x = jnp.stack([data[i:i+block_size] for i in ix])
    # we handle this in train_step
    # y = jnp.stack([data[i+1:i+1+block_size] for i in ix])
    return x if not return_meta else (x, {'ix': ix})


print(inputs.input_ids.shape)
import jax
rng = jax.random.PRNGKey(0)
print("get_batch:", get_batch(inputs.input_ids.flatten(), rng).shape)
# %%
from transformers.models.gpt2 import FlaxGPT2LMHeadModel
model = FlaxGPT2LMHeadModel(
    GPT2Config(
        vocab_size=len(tokenizer.get_vocab()),
        n_positions=tokenizer.model_max_length,
        pad_token_id=tokenizer.pad_token_id,
        bos_token_id=tokenizer.bos_token_id,
        eos_token_id=tokenizer.eos_token_id
))
print(tokenizer.model_max_length,)
rng = jax.random.PRNGKey(0)
input_shape = (1, tokenizer.model_max_length)
model.params = model.init_weights(rng, input_shape)
# %%
print(inputs.input_ids.shape)
outs = model(
    input_ids=get_batch(inputs.input_ids.flatten(), rng),
    attention_mask=None,
    return_dict=True,
    params=model.params,
    dropout_rng=rng,
    train=True,
)
print(outs.logits.shape)
assert not jnp.isnan(outs.logits.sum()), "BUG: NaN in logits"
# %%
optimizer = optax.adam(1e-4)
opt_state = optimizer.init(model.params)


@jax.jit
def train_step(params, opt_state, input_ids, rng):
    def loss_fn(params):
        outs = model(
            input_ids=input_ids,
            # attention_mask=attention_mask,
            return_dict=True,
            params=params,
            dropout_rng=rng,
            train=True,
        )
        logits = outs.logits
        logits = jnp.clip(logits, -1e9, 1e9)  # Clip logits to a reasonable range
        loss = optax.softmax_cross_entropy(
            logits[:, :-1],
            jax.nn.one_hot(input_ids[:, 1:], logits.shape[-1])
        )
        # we expect standard causal attention, no attn mask required
        #masked_loss = loss * attention_mask[:, 1:]
        #normalized_loss = masked_loss.sum() / attention_mask[:, 1:].sum()
        return loss.mean() #normalized_loss

    loss, grads = jax.value_and_grad(loss_fn)(params)
    grads = jax.tree.map(lambda g: jnp.clip(g, -1.0, 1.0), grads)
    updates, opt_state = optimizer.update(grads, opt_state)
    return optax.apply_updates(params, updates), opt_state, loss
# %%
import aim
import tk
import tqdm

close_run = False
print("logs in", (log_dir := tk.datadir / 'logs'))
log_dir.mkdir(parents=True, exist_ok=True)

try:
    if (run_ := locals().get('run')) and run_.active:
        print("Reusing existing run.")
        run = run_
except:  # idk run_.active can fail, maybe bug
    run = None

if not (run_ := locals().get('run')):
    print("New run.")
    run = aim.Run(repo=log_dir, experiment='2410_gpt2code')

print(f"{close_run=}")
# %%
L = tk.utils.log.logger

def sample_text(prompt: str):
    inputs = tokenizer(
        [prompt],
        return_tensors="jax",
        padding=True,
        truncation=True,
        add_special_tokens=False,
    )
    # inputs.pop("token_type_ids", None)
    output_ids = model.generate(**inputs, max_length=50)
    output_ids = output_ids.sequences
    return tokenizer.decode(
        output_ids[0], skip_special_tokens=True)

# TODO tokenizer should know to handle <EOS> token
# however it seems to split on < and >, so we need to handle this manually
input_ids = []
for d in data:
    cur_ids = tokenizer(
        d,
        return_tensors="jax",
        max_length=None,
    ).input_ids
    input_ids.append(cur_ids)
    input_ids.append(jnp.array([[tokenizer.eos_token_id]]))
input_ids = jnp.hstack(input_ids).flatten()
rng = jax.random.PRNGKey(42)
# %%
import importlib
from tk.utils import utils, log
from collections import Counter
importlib.reload(utils)
importlib.reload(log)
# d = log.Distribution(Counter("aabbc"))
# print(list(d.storage.items()))
# from rich import inspect
# %%
L.info(f"Inputs: {input_ids.shape}")
epoch_cum = locals().get('epoch', 0)
run = locals().get('run')
assert isinstance(run, aim.Run), f'stuff went awry logging {run=}'
# not needed since model is handling this
# train_step_batched = jax.vmap(train_step, in_axes=(None, None, 0, 0))
idxs_count = Counter()
with tqdm.trange(epoch_cum, epoch_cum + 20, desc="Epoch") as t:
    for epoch in t:
        rng, batch_rng = jax.random.split(rng)
        x, meta = get_batch(
            input_ids,
            rng=batch_rng,
            batch_size=8,
            return_meta=True
        )
        rng, train_rng = jax.random.split(rng)
        model.params, opt_state, loss = train_step(
            model.params,
            opt_state,
            x,
            train_rng,
        )
        idxs_count.update(meta['ix'].flatten().tolist())
        # run['counter'] = idxs_count
        hist = sorted(idxs_count.items())
        a, b = hist[0][0], hist[-1][0]
        run.track(aim.Distribution(hist=[v for k, v in hist], bin_range=(a, b)), name='ix_cum', epoch=epoch)
        run.track(loss, name='loss', epoch=epoch)
        if (epoch + 1) % 10 == 0:
            for prompt in ("def", "x ="):
                t.set_postfix(status=f"sampling '{utils.shrt(prompt)}'", loss=loss)
                outs = sample_text(prompt)
                run.track(aim.Text(outs), name='sample', epoch=epoch, context={
                    'prompt': prompt
                })

        t.set_postfix(loss=loss)

if close_run:
    print("Closing run.")
    run.close()
# %%
text = sample_text("def")
print("Done.")
print("=====")
print(text)
