"""N-gram models on tiny shakespeare.

How does the model behave for various n?
How does a pre-trained tokenizer help w/ coherence?
"""
# %%
import tk
import importlib
importlib.reload(tk)
# %%
shakespeare = tk.memo(tk.utils.data.tinyshakespeare)()
words = tk.memo(tk.utils.data.words)('en')
# %%
from tk.models import ngram
with tk.timed():
    tokenizer = ngram.Tokenizer.init(shakespeare, mincount=2, maxn_or_tuples=(1, 2, 4, 6, 8, 10, 12, 14, 16, 32))
# %%
print(ngram.generate(tokenizer, "hello!"))
# %%
from transformers import AutoTokenizer
tokenizer_gpt2 = AutoTokenizer.from_pretrained("gpt2")
# %%
shakespeare_tokens = tokenizer_gpt2.tokenize(shakespeare)
print(shakespeare_tokens[:10])
# %%
with tk.timed('ngram_tokens'):
    tokenizer_atop_gpt2 = ngram.Tokenizer.init(
        shakespeare_tokens,
        mincount=2,
        maxn_or_tuples=5,
    )
# %%
start = tokenizer_gpt2.tokenize("hello!")
out, _ = ngram.generate(tokenizer_atop_gpt2, start)
print(out)
# %%
out = tokenizer_gpt2.convert_tokens_to_string(out)
print(out)