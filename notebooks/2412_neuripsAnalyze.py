"""NeurIPS papers.

Make sure to fetch them first via neuripsFetch.
"""
# %%
import tk
import pandas as pd

_nameit = lambda x: tk.datadir / f"2412_openreviewNotes{x}"
fname = _nameit('.json')
df = pd.read_json(fname)

# %%
df['topic'].value_counts()

# %%
def clean(xs):
    replace = lambda text: (
    ''.join(c for c in text if ord(c) > 31 or c in '\t\n\r')
    if isinstance(text, str) else text
    )
    return xs.apply(replace)

df.apply(clean, axis=1).to_excel(_nameit(".xlsx"))
# %%
