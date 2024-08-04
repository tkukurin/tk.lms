
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
    tokenizer = ngram.Tokenizer.init(shakespeare, mincount=2, maxn=16)
# %%
print(ngram.generate(tokenizer, "hello!"))
# %%
