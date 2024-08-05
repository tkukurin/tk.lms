"""Run the training script, then load model here.
"""
# %%
import tk
model_path = tk.datadir / "outputs/summing_train/step=1000"
import transformers as hft
tokenizer = hft.AutoTokenizer.from_pretrained(model_path)
model = hft.FlaxAutoModelForCausalLM.from_pretrained(model_path)
# %%
from loguru import logger
logger.info(tokens := tokenizer("1+", return_tensors="jax"))
# %%
import jax.numpy as jnp
from jax import random as j42
import tqdm
from flax.linen import softmax


def greedy(text: str, topk: int=1, maxn: int = 10, rng: j42.PRNGKey = j42):
    tokens = tokenizer(text).input_ids
    probs_out = []
    for step in tqdm.trange(maxn):
        logits = model(jnp.array(tokens)[None, :]).logits
        logits = logits[0, -1, :]  # last
        probs = softmax(logits)
        topk_ix = jnp.argpartition(probs, -topk)[-topk:]
        topk_probs = probs[topk_ix]
        tokens.append(topk_ix[-1].item())
        probs_out.append(topk_probs[-1].item())
    return tokens, probs_out
  

outs = []
for i in range(5):
    toks, probs = greedy(f"{i}")
    print(tokenizer.convert_ids_to_tokens(toks))
    outs.append((toks, probs))


# %%
tokens = tokenizer("2", return_tensors="jax")
outs = model.generate(**tokens, max_new_tokens=10)
print(tokenizer.convert_ids_to_tokens(outs.sequences[0]))
# %%
