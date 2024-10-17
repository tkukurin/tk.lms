"""Train GPT via Transformers and Flax.
"""
# %%
import jax
import jax.numpy as jnp
import optax
from transformers import FlaxGPT2LMHeadModel, GPT2Tokenizer, GPT2Config

tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
tokenizer.pad_token = tokenizer.eos_token
data = [
    "def my_function(overfit=True):\n    \"hello here I am overfitting as hell\"\n    return 1",
    "def my_other_function(overfit=False):\n    return 0"
]
inputs = tokenizer(data, return_tensors="jax", padding=True, truncation=True)

# model = FlaxGPT2LMHeadModel.from_pretrained("gpt2")
model = FlaxGPT2LMHeadModel(GPT2Config(vocab_size=50257, n_positions=1024))
optimizer = optax.adam(1e-4)
opt_state = optimizer.init(model.params)


def train_step(params, opt_state, input_ids, attention_mask):
    def loss_fn(params):
        logits = model(
            input_ids=input_ids, attention_mask=attention_mask, params=params
        ).logits

        loss =  optax.softmax_cross_entropy(
            logits[:, :-1],
            jax.nn.one_hot(input_ids[:, 1:], logits.shape[-1])
        )
        masked_loss = loss * attention_mask[:, 1:]
        normalized_loss = masked_loss.sum() / attention_mask[:, 1:].sum()
        return normalized_loss

    loss, grads = jax.value_and_grad(loss_fn)(params)
    updates, opt_state = optimizer.update(grads, opt_state)
    return optax.apply_updates(params, updates), opt_state, loss
# %%
import aim
import tk
import tqdm

close_run = True
log_dir = tk.datadir / 'logs'
log_dir.mkdir(parents=True, exist_ok=True)

try:
    if (run_ := locals().get('run')) and run_.active:
        print("Reusing existing run.")
        run = run_
except:
    run = None

if not (run_ := locals().get('run')):
    print("New run.")
    run = aim.Run(repo=log_dir)
    run.read_only

print(f"{close_run=}")

for epoch in tqdm.trange(10):
    model.params, opt_state, loss = train_step(
        model.params,
        opt_state,
        inputs["input_ids"],
        inputs["attention_mask"]
    )
    print(loss)
    run.track(loss, name='loss', epoch=epoch)

if close_run:
    run.close()
# %%
def sample_text(prompt="def"):
    inputs = tokenizer([prompt], return_tensors="jax", padding=True, truncation=True)
    output_ids = model.generate(inputs["input_ids"], max_length=50)
    output_ids = output_ids.sequences
    return tokenizer.decode(output_ids[0], skip_special_tokens=True)


text = sample_text("def")
print("Done.")
print("=====")
print(text)
