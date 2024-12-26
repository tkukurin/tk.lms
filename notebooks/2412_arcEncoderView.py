"""Check ARC tokenizer stats and info.
"""

# %%
from tk.arc import converters
from tk import datadir
df, df_io, df_grouped, vocab = converters.split_stored_df(
    datadir / 'michaelhodel_rearc_data.pkl')
# %%
encoder = converters.SimpleArcGridSeqEncoder(
    vocab, df_io=df_io, df_grouped=df_grouped)
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
encoded.keys()
# %%
i2t = {v: k for k, v in encoder.tok2id.items()}
pad = encoder.tok2id['<pad>']
[i2t[k] for k in encoded['007bbfb7'] if k != pad]

# %%