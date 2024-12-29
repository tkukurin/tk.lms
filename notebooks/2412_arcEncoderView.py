"""Check ARC tokenizer stats and info, then dump filtered data to parquet for training.
"""

# %%
from tk.arc.expt import encoding
from tk import datadir
df, df_io, df_grouped, vocab = encoding.split_stored_df(
    datadir / 'michaelhodel_rearc_data.pkl')
# %%
encoder = encoding.SimpleArcGridSeqEncoder(
    vocab, df_io=df_io, df_grouped=df_grouped)
# %%
df_io
# %%
toks, types, meta = encoder.encode_problem('007bbfb7', max_length=2048)
types
# %%
quantile = 0.75
encoded, meta = encoder.encode_all_with_padding(
    max_length=2048,
    quantile=quantile)
skipped = meta['skipped_full']
hist = meta['hist']
print(f"{quantile * 100}% | {int(quantile * max(hist.keys()))} | {list(hist.keys())}")
print(f"{len(skipped)=} {len(encoded)=}")
# %%
import matplotlib.pyplot as plt
fig, ax = plt.subplots()
ax.hist(hist.keys(), bins=20)
ax.set_title("Encoded problem length in tokens")
# add veritcal line at quantile
ax.axvline(quantile * max(hist.keys()), color='r')
# ax.text(0.75 * max(hist.keys()), 20, '75%', rotation=90, va='bottom')
ax.text(0.875 * max(hist.keys()), 85, 'Skipped', color='r', ha='center', bbox=dict(facecolor='white', alpha=0.5, edgecolor='white'))
ax.fill_betweenx(
    [0, 100], 
    quantile * max(hist.keys()), 
    max(hist.keys()), color='r', alpha=0.1)
ax.set_xlim(0, max(hist.keys()))
ax.set_ylim(0, 100)
plt.show()
# %% also plot only the non-skipped histogram
fig, ax = plt.subplots()
ax.hist([k for k in hist if k < quantile * max(hist.keys())], bins=20)
ax.set_title("Encoded problem length in tokens (non-skipped)")
plt.show()
# %%
import jax
tok = encoder.tok
[tok.id2tok[k] for k in jax.device_get(encoded[0]['input_ids']) if k != tok.pad_id]
# %%
from tk import datadir
encoding.save_data(encoded, encoder.tok, datadir / 'mhodel_rearc')
# %%
ds, tok = encoding.load_data(datadir / 'mhodel_rearc')
# %%
assert tok.id2tok == encoder.tok.id2tok
assert tok.tok2id == encoder.tok.tok2id
# %%
ds['input_ids'].shape
# %%
[len(x) for x in ds['input_ids']]

# %%
