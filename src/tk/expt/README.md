# Copy-Task Pre-training Experiment

This experiment tests the hypothesis that **copy-task pre-training improves language model performance** on downstream tasks requiring retrieval-like capabilities.

## Hypothesis

Models that undergo copy-task pre-training will show improved performance on:
- In-context learning tasks
- Few-shot pattern recognition  
- Retrieval-based reasoning
- Tasks requiring information recall

## Experimental Design

### Phase 1: Copy-Task Pre-training
The model is first trained on various copy tasks designed to enhance retrieval capabilities:

1. **Simple Sequence Copying**: `<s> [sequence] <copy> [sequence] </s>`
2. **Key-Value Retrieval**: `<s> <key>k1<val>v1 <key>k2<val>v2 ... <key>query_key <val> [value] </s>`
3. **Pattern Completion**: `<s> [pattern] [pattern] [partial_pattern] [completion] </s>`

### Phase 2: Downstream Evaluation
Both baseline (no pre-training) and copy-task pre-trained models are evaluated on:

1. **Arithmetic In-Context Learning**: `1+1=2, 2+2=4, 3+3=?`
2. **Pattern Recognition**: `[1,2,3]->[4], [2,4,6]->[8], [1,3,5]->?`
3. **Analogy Tasks**: `cat:kitten, dog:puppy, bird:?`
4. **Fact Retrieval**: `John lives in Paris. Mary lives in London. Where does John live?`

## Implementation Structure

```
src/tk/expt/
├── copy_task_experiment.py    # Main experiment framework
├── eval_benchmarks.py         # Downstream task evaluators
└── run_copy_experiment.py     # Configuration and runner
```

### Key Components

1. **`CopyTaskGenerator`**: Creates diverse copy tasks for pre-training
2. **`CopyTaskExperiment`**: Manages the full experimental pipeline
3. **`InContextLearningEvaluator`**: Tests in-context learning capabilities
4. **`RetrievalQAEvaluator`**: Tests retrieval-based question answering

## Usage

### Quick Start
```bash
cd src
python -m tk.expt.run_copy_experiment --steps=1000 --eval_samples=100
```

### Custom Configuration
```bash
python -m tk.expt.run_copy_experiment \
    --steps=2000 \
    --batch_size=16 \
    --lr=5e-4 \
    --eval_samples=200
```

### Expected Output
```
============================================================
COPY-TASK PRE-TRAINING EXPERIMENT
============================================================

Model vocabulary size: 507
Model configuration: GPTConfig(vocab_size=507, block_size=128, ...)

PHASE 1: Copy-task Pre-training
----------------------------------------
Starting copy-task pre-training...
Copy pre-training step 0, loss: 6.2341
Copy pre-training step 100, loss: 2.1456
...
Initial loss: 6.2341
Final loss: 1.2345
Loss reduction: 80.2%

PHASE 2: Downstream Task Evaluation
----------------------------------------
Evaluating on arithmetic...
  Baseline accuracy: 0.2340
  Pretrained accuracy: 0.3125
  Improvement: +33.5%

Evaluating on pattern_completion...
  Baseline accuracy: 0.1890
  Pretrained accuracy: 0.2567
  Improvement: +35.8%

...

EXPERIMENT SUMMARY
============================================================
OVERALL RESULTS:
  Average improvement: +25.3%
  Tasks with >5% improvement: 4/4

HYPOTHESIS VALIDATION:
✅ HYPOTHESIS SUPPORTED: Copy-task pre-training improves downstream performance
```

## Research Context

This experiment is inspired by recent work on:

1. **Pre-training for In-Context Learning (PICL)** - Gu et al., 2023
2. **In-Context Pre-training** - Shi et al., 2023  
3. **Meta-Learning for In-Context Learning** - Min et al., 2022

### Key Research Questions

1. **Does copy-task pre-training improve retrieval capabilities?**
2. **Which types of copy tasks are most effective?**
3. **How does improvement scale with pre-training duration?**
4. **What downstream tasks benefit most from copy-task pre-training?**

## Benchmarks and Metrics

### Copy-Task Pre-training Metrics
- **Cross-entropy loss** on copy tasks
- **Task-specific accuracy** (exact sequence matching)

### Downstream Evaluation Metrics  
- **Accuracy** on next-token prediction
- **Task completion rate** (full sequence correctness)
- **Relative improvement** over baseline

### Expected Results

Based on the research literature, we expect:
- **10-30% improvement** on in-context learning tasks
- **Larger improvements** on retrieval-heavy tasks
- **Diminishing returns** beyond certain pre-training duration

## Extending the Experiment

### Additional Copy Tasks
```python
def generate_associative_recall(self):
    """A->1, B->2, C->3, A->?"""
    
def generate_list_reversal(self):
    """[1,2,3,4] -> [4,3,2,1]"""
    
def generate_selective_copying(self):
    """Copy only items matching condition"""
```

### Additional Benchmarks
- **GLUE/SuperGLUE subset tasks**
- **Few-shot text classification**
- **Reading comprehension with retrieval**
- **Factual knowledge recall**

### Ablation Studies
- **Vary pre-training duration** (100, 500, 1000, 2000 steps)
- **Different copy task mixtures** (simple vs. complex)
- **Model size effects** (small vs. large models)

## Implementation Notes

### Technical Considerations
- Uses **JAX/Flax** for efficient training
- **Character-level tokenization** for simplicity
- **Automatic padding and masking** for variable-length sequences
- **Configurable model architectures**

### Performance Optimization
- **JIT compilation** for fast training loops
- **Batched evaluation** for efficiency
- **Memory-efficient data loading**

### Reproducibility
- **Fixed random seeds** (42) for consistent results
- **Comprehensive logging** of all hyperparameters
- **JSON result serialization** for analysis

## Contributing

To add new copy tasks or evaluation benchmarks:

1. **Extend `CopyTaskGenerator`** with new task types
2. **Add evaluators** in `eval_benchmarks.py`  
3. **Update configuration** in `run_copy_experiment.py`
4. **Add tests** and documentation

## References

1. Gu, Y., et al. (2023). "Pre-Training to Learn in Context." *ACL 2023*.
2. Shi, W., et al. (2023). "In-Context Pretraining." *arXiv preprint*.
3. Min, S., et al. (2022). "MetaICL: Learning to Learn In Context." *NAACL 2022*.
4. Brown, T., et al. (2020). "Language Models are Few-Shot Learners." *NeurIPS 2020*.
