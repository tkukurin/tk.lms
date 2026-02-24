"""A couple plotting demos for 2408 notebooks.
"""
# %%
import altair as alt
import pandas as pd

# Sample model weights
weights = {
    ('layer1', 'weight'): 0.5,
    ('layer1', 'bias'): 0.1,
    ('layer2', 'weight'): -0.3,
    ('layer2', 'bias'): 0.2,
    ('layer3', 'weight'): 0.7,
    ('layer3', 'bias'): -0.5,
}

df = pd.DataFrame([
    {'Layer': key[0], 'Type': key[1], 'Weight': value}
    for key, value in weights.items()
])

chart = alt.Chart(df).mark_bar().encode(
    x='Layer:N',
    y='Weight:Q',
    color='Type:N',
    tooltip=['Layer', 'Type', 'Weight']
).properties(
    width=400,
    height=300
)

chart.show()

# %%

import altair as alt
import pandas as pd

# Sample model weights with (mean, std)
weights = {
    ('layer1', 'weight'): (0.5, 0.1),
    ('layer1', 'bias'): (0.1, 0.05),
    ('layer2', 'weight'): (-0.3, 0.07),
    ('layer2', 'bias'): (0.2, 0.04),
    ('layer3', 'weight'): (0.7, 0.2),
    ('layer3', 'bias'): (-0.5, 0.15),
}

df = pd.DataFrame([
    {'Layer': key[0], 'Type': key[1], 'Mean': value[0], 'Std': value[1]}
    for key, value in weights.items()
])

chart = alt.Chart(df).mark_bar().encode(
    x='Layer:N',
    y='Mean:Q',
    color='Type:N',
    tooltip=['Layer', 'Type', 'Mean', 'Std']
).properties(
    width=400,
    height=300
).interactive()

error_bars = alt.Chart(df).mark_errorbar(extent='stdev').encode(
    x='Layer:N',
    y='Mean:Q',
    yError='Std:Q',
    color='Type:N'
)

final_chart = chart + error_bars
final_chart.show()
# %%

import matplotlib.pyplot as plt
import numpy as np

weights = {
    ('layer1', 'weight'): (0.5, 0.1),
    ('layer1', 'bias'): (0.1, 0.05),
    ('layer2', 'weight'): (-0.3, 0.07),
    ('layer2', 'bias'): (0.2, 0.04),
    ('layer3', 'weight'): (0.7, 0.2),
    ('layer3', 'bias'): (-0.5, 0.15),
}

layers = list(set(key[0] for key in weights.keys()))
types = list(set(key[1] for key in weights.keys()))

fig, ax = plt.subplots(figsize=(8, 6))

offset = np.linspace(-0.2, 0.2, len(types))
for idx, t in enumerate(types):
    means = [weights[(layer, t)][0] for layer in layers]
    stds = [weights[(layer, t)][1] for layer in layers]
    positions = np.arange(len(layers)) + offset[idx]

    ax.errorbar(positions, means, yerr=stds, fmt='o', label=t, capsize=5)

ax.set_xticks(np.arange(len(layers)))
ax.set_xticklabels(layers)
ax.set_xlabel('Layer')
ax.set_ylabel('Mean Weight')
ax.set_title('Model Weights by Layer and Type')
ax.legend(title='Type')
plt.grid(True)
plt.show()
# %%
