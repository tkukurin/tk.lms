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
df['topic'] = df['topic'].astype(pd.CategoricalDtype())
print(
    df['topic'].value_counts()[:5],
    df['topic'].value_counts()[-5:],
    sep='\n\n'
)

# topic
# machine_vision                 605
# natural_language_processing    313
# reinforcement_learning         288
# learning_theory                264
# diffusion_based_models         229
# Name: count, dtype: int64
# 
# topic
# speech_and_audio                        30
# infrastructure                          28
# active_learning                         25
# human-AI_interaction                    22
# machine_learning_for_social_sciences    21
# Name: count, dtype: int64

# %%
df['authors'].apply(lambda x: len(x)).hist(bins=10)
# %%
import itertools as it
from collections import Counter
auth2count = Counter(it.chain(*df['authors']))
# %%
auth2df = (
    df.explode('authors')
    .apply({'authors': lambda x: x.apply(str.title)})
    .groupby('authors')
)
print(f'{len(auth2df.groups)=}')

auth2count = sorted(
    [(len(v), k) for k, v in auth2df.groups.items()])
print(
    auth2count[:5],
    auth2count[-5:],
    sep='\n\n'
)
# %%
def clean(xs):
    replace = lambda text: (
    ''.join(c for c in text if ord(c) > 31 or c in '\t\n\r')
    if isinstance(text, str) else text
    )
    return xs.apply(replace)

# for viewing in gsheets, need to remove some tab chars tho
# df.apply(clean, axis=1).to_excel(_nameit(".xlsx"))
# %%
from pathlib import Path

class customfmt:
    @classmethod
    def read_file(cls, fname: Path) -> list[dict]:
        with fname.open("r", encoding="utf-8") as f:
            return cls.read_text(f.read())
    
    @classmethod
    def multiturn(cls, data: list[dict], id: str):
        """dont forget, I used this dumb 'convention' adhoc"""
        *chain_ids, end = id.split(".")
        chain = []
        for d in data:
            d_id = d['meta']['id']
            if id.startswith(d_id):
                chain.append(d)
        return chain

    @classmethod
    def read_text(cls, text: str) -> list[dict]:
        entries = []
        current_entry = {}
        current_meta = {}
        text_lines = []
        lines = text.splitlines()
        for line in lines:
            line = line.strip()
            if line.startswith("###[sep]"):
                if current_entry:
                    current_entry['text'] = "\n".join(text_lines).strip()
                    current_entry['meta'] = current_meta
                    entries.append(current_entry)
                current_entry = {}
                current_meta = {}
                text_lines = []
                _, line = line.split(" ", 1)
            if line.startswith("---"):
                current_entry['meta'] = current_meta
            elif ":" in line and not text_lines:
                key, value = map(str.strip, line.split(":", 1))
                current_meta[key] = value
            else:
                text_lines.append(line)
        if current_entry:
            current_entry['text'] = "\n".join(text_lines).strip()
            current_entry['meta'] = current_meta
            entries.append(current_entry)
        return entries

# Example Usage
dataset = """\
###[sep] id: 1
title: Sample Title 1
---
This is the text of the first entry.
###[sep] id: 2
title: Sample Title 2
---
This is the text of the second entry.

And more lines.
"""
# test wai
print(_test := customfmt.read_text(dataset))
assert _test == [{'meta': {'id': '1', 'title': 'Sample Title 1'}, 'text': 'This is the text of the first entry.'}, {'meta': {'id': '2', 'title': 'Sample Title 2'}, 'text': 'This is the text of the second entry.\n\nAnd more lines.'}]

# %%
answer_infos = customfmt.read_file(tk.datadir / "2412_neuripsAnalyze.prompting.customfmt.txt")
# %%
# %%

gemini_flash_answer = answer_infos[0]['text']
gemini_flash_top10_mutliturn = customfmt.multiturn(answer_infos, '0.1')
gemini_flash_top10_recs = gemini_flash_top10_mutliturn[1]['text']

# %%
import re

def parse(line: str):
    # line = "* **words**: 1, 2, 3"

    pattern = r"\*\s+\*\*(.*?)\*\*:?\s+([\d,\s]+)"
    if match := re.match(pattern, line):
        key = match.group(1)
        values = list(map(int, match.group(2).split(',')))
        result = (key.strip(': '), values)
        return result

kvs = []
for line in gemini_flash_answer.split("\n"):
    if kv := parse(line):
        kvs.append(kv)
kvs = dict(kvs)
print(len(kvs))
# %%
import itertools as it
overlaps = {}
for (k1, v1), (k2, v2) in it.combinations(kvs.items(), 2):
    diff = set(v1) & set(v2)
    if diff:
        overlaps[(k1, k2)] = diff
print({k:len(v) for k, v in overlaps.items()})
# {('Alignment and Safety', 'LLM Capabilities and Evaluation'): 31, ('Alignment and Safety', 'Offline RL'): 5, ('Alignment and Safety', 'Image Generation and Editing'): 6, ('LLM Capabilities and Evaluation', 'Offline RL'): 25, ('LLM Capabilities and Evaluation', 'Image Generation and Editing'): 34, ('Offline RL', 'Image Generation and Editing'): 29}

# %%
claude_sonnet_35 = """
## Key Research Areas

**Machine Learning & Deep Learning**
- Novel architectures like TAS-GNN improve graph classification performance by up to 27.20% through optimized neuron utilization
- S3, a modular neural network layer, enhances time-series representation learning by rearranging segments
- PGN introduced as a successor to RNN for long-range time series forecasting

**Language Models & Alignment**
- MAmmoTH2 harvests 10M high-quality instruction data from web corpus for fine-tuning language models without costly human annotation
- FLAME proposes factuality-aware alignment while maintaining instruction-following capability
- DropBP accelerates LLM fine-tuning by selectively dropping backward propagation based on layer sensitivity[191]

**Computer Vision & Graphics**
- G2D framework learns global and dense visual features from image-text pairs, achieving strong performance with just 1% of training data
- Neural Gaffer enables relighting any object image under any environmental lighting using diffusion models
- MeshFormer delivers high-quality meshes from sparse-view reconstruction[186]

**Reinforcement Learning**
- REBEL simplifies policy optimization by regressing relative rewards with theoretical guarantees
- Dynamic Model Predictive Shielding enables safe RL with goal progress while recovering from unsafe situations[193]
- Focus On What Matters (SMG) improves RL generalization through task-relevant representation extraction[196]

**AI Safety & Robustness**
- Password-locked models are used to study capability elicitation through supervised fine-tuning and RL
- Improved few-shot jailbreaking techniques can circumvent aligned language models and their defenses
- Pure Tune, Safe Test principle introduced to maintain model alignment after fine-tuning[176]

## Theoretical Advances

**Mathematical Foundations**
- First global convergence proof for gradient EM on over-parameterized Gaussian mixture models
- New generalization bounds for DNNs learning composition of functions
- Novel theoretical framework for f-divergence-based domain learning[169]

**Optimization & Learning Theory**
- Simple universal approach achieves optimal gradient-variation regret guarantees[109]
- Quantum speedups demonstrated for finding Goldstein stationary points[202]
- Minimax optimal bounds established for corruption-robust linear bandits[203]
"""

# %%
# %%
