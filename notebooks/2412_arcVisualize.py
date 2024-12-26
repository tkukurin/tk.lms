"""Visualize ARC-AGI tasks.

I extracted GPT failures from the eval set specifically.
"""
# %%
# I got this by setting a breakpoint on https://o3-failed-arc-agi.vercel.app/
# There is loadFailedTask which contains FAILED_TASKS global.

gpt_failed_on = {
    'evaluation': [
        'da515329', 'f9d67f8b', '891232d6', '52fd389e', 'c6e1b8da', '09c534e7',
        'ac0c5833', '47996f11', 'b457fec5', 'b7999b51', 'b9630600', '896d5239',
        '40f6cd08', '8b28cd80', '93c31fbe', '25094a63', '05a7bcf2', '0934a4d8',
        '79fb03f4', '4b6b68e5', 'aa4ec2a5', '1acc24af', 'f3b10344', '256b0a75',
        'd931c21c', '16b78196', 'a3f84088', '212895b5', '0d87d2a6', '3ed85e70',
        'e619ca6e', 'e1d2900e', 'd94c3b52', 'e681b708'
    ]
}

# %%

colors = {
    0: '#444444',  # black
    1: '#0074D9',  # blue
    2: '#FF4136',  # red 
    3: '#2ECC40',  # green
    4: '#FFDC00',  # yellow
    5: '#AAAAAA',  # grey
    6: '#F012BE',  # fuschia
    7: '#FF851B',  # orange
    8: '#7FDBFF',  # teal
    9: '#870C25'   # brown
}

import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import numpy as np
%matplotlib inline

def visualize_grid(grid) -> plt.Figure:
    plt.close()
    grid = np.array(grid)
    fig, ax = plt.subplots()
    ax.set_aspect('equal')
    ax.axis('off')
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_xlim(0, grid.shape[1])
    ax.set_ylim(0, grid.shape[0])
    for i in range(grid.shape[0]):
        for j in range(grid.shape[1]):
            ax.add_patch(Rectangle((j, grid.shape[0] - i - 1), 1, 1, 
            color=colors[grid[i, j]]))
    return fig

# %%
from tk.arc.p3 import get_data
data = get_data('evaluation')
# %%
with plt.ioff():
    fig = visualize_grid(data['test']['da515329'][0]['input'])
fig
# %%
from tk.arc import converters
from tk import datadir
df, df_io, df_grouped, vocab = converters.split_stored_df(
    datadir / 'michaelhodel_rearc_data.pkl')
# %%
encoder = converters.SimpleArcGridSeqEncoder(
    vocab, df_io=df_io, df_grouped=df_grouped)
# %%
encoded, skipped, hist = encoder.encode_all_with_padding(
    quantile=0.75)
# %%
import matplotlib.pyplot as plt
fig, ax = plt.subplots()
ax.hist(hist.keys(), bins=20)
ax.set_title("Encoded problem length in tokens")
# add veritcal line at quantile
ax.axvline(0.75 * max(hist.keys()), color='r')
# ax.text(0.75 * max(hist.keys()), 20, '75%', rotation=90, va='bottom')
ax.text(0.875 * max(hist.keys()), 85, 'Skipped', color='r', ha='center', bbox=dict(facecolor='white', alpha=0.5, edgecolor='white'))
ax.fill_betweenx(
    [0, 100], 0.75 * max(hist.keys()), max(hist.keys()), color='r', alpha=0.1)
ax.set_xlim(0, max(hist.keys()))
ax.set_ylim(0, 100)
plt.show()

# %%
