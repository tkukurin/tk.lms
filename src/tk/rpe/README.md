# Randomized Positional Encodings Boost Length Generalization of Transformers

Work on top of [randomized positional encodings](https://github.com/google-deepmind/randomized_positional_encodings).
Implementation of [Randomized Positional Encodings Boost Length Generalization of Transformers](https://arxiv.org/abs/2305.16843) (ACL23).

>Transformers have impressive generalization capabilities on tasks with a fixed context length.
However, they fail to generalize to sequences of arbitrary length, even for seemingly simple tasks such as duplicating a string.
Moreover, simply training on longer sequences is inefficient due to the quadratic computation complexity of the global attention mechanism.
In this work, we demonstrate that this failure mode is linked to positional encodings being out-of-distribution for longer sequences (even for relative encodings) and introduce a novel family of positional encodings that can overcome this problem.
Concretely, our randomized positional encoding scheme simulates the positions of longer sequences and randomly selects an ordered subset to fit the sequence's length.
Our large-scale empirical evaluation of 6000 models across 15 algorithmic reasoning tasks shows that our method allows Transformers to generalize to sequences of unseen length (increasing test accuracy by 12.0% on average).

## Content

```
.
├── models
│   ├── positional_encodings.py
│   ├── transformer.py              - Transformer (Vaswani et al., 2017)
│   └── transformer_utils.py
├── tasks
│   ├── cs                          - Context-sensitive tasks
│   ├── dcf                         - Deterministic context-free tasks
│   ├── regular                     - Regular tasks
│   └── task.py                     - Abstract `GeneralizationTask`
├── experiments
|   ├── constants.py                - Training/Evaluation constants
|   ├── curriculum.py               - Training curricula (over sequence lengths)
|   ├── example.py                  - Example traning script
|   ├── range_evaluation.py         - Evaluation loop (test sequences lengths)
|   ├── training.py                 - Training loop
|   └── utils.py                    - Utility functions
├── README.md
└── requirements.txt                - Dependencies
```
