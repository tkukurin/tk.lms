"""
Evaluation benchmarks for copy-task pre-training experiment.

This module implements specific downstream tasks that test retrieval capabilities:
1. In-context learning tasks
2. Few-shot pattern recognition
3. Retrieval-based question answering
"""

import jax
import jax.numpy as jnp
import numpy as np
from typing import Dict, List, Tuple, Any
from dataclasses import dataclass

from tk.utils.data.vocab import Voc


@dataclass
class EvalConfig:
    """Configuration for evaluation tasks."""
    max_context_length: int = 64
    num_examples: int = 3  # for few-shot tasks
    vocab_size: int = 1000


class InContextLearningEvaluator:
    """Evaluator for in-context learning capabilities."""
    
    def __init__(self, vocab: Voc, config: EvalConfig, rng: np.random.Generator):
        self.vocab = vocab
        self.config = config
        self.rng = rng
    
    def generate_arithmetic_task(self, batch_size: int) -> Dict[str, jnp.ndarray]:
        """Generate arithmetic in-context learning task.
        
        Format: 1+1=2, 2+2=4, 3+3=?
        """
        inputs, targets, masks = [], [], []
        
        for _ in range(batch_size):
            # Generate examples and query
            examples = []
            for _ in range(self.config.num_examples):
                a = self.rng.integers(1, 10)
                b = self.rng.integers(1, 10)
                result = a + b
                examples.append(f"{a}+{b}={result}")
            
            # Query
            query_a = self.rng.integers(1, 10)  
            query_b = self.rng.integers(1, 10)
            query_result = query_a + query_b
            
            # Convert to sequence
            text = ", ".join(examples) + f", {query_a}+{query_b}="
            target_text = str(query_result)
            
            # Tokenize (simplified - using character level)
            input_tokens = [ord(c) % 100 for c in text]  # Simple char tokenization
            target_tokens = [ord(c) % 100 for c in target_text]
            
            # Pad and truncate
            max_len = self.config.max_context_length
            if len(input_tokens) + len(target_tokens) > max_len:
                continue
                
            full_seq = input_tokens + target_tokens
            padding = [0] * (max_len - len(full_seq))
            padded = full_seq + padding
            
            inputs.append(padded[:-1])
            targets.append(padded[1:])
            masks.append([1] * (len(full_seq) - 1) + [0] * (max_len - len(full_seq)))
        
        return {
            'input_ids': jnp.array(inputs),
            'target_ids': jnp.array(targets),
            'attention_mask': jnp.array(masks)
        }
    
    def generate_pattern_completion_task(self, batch_size: int) -> Dict[str, jnp.ndarray]:
        """Generate pattern completion in-context learning.
        
        Format: [1,2,3] -> [4], [2,4,6] -> [8], [1,3,5] -> ?
        """
        inputs, targets, masks = [], [], []
        
        for _ in range(batch_size):
            examples = []
            
            # Generate pattern examples
            for _ in range(self.config.num_examples):
                start = self.rng.integers(1, 5)
                step = self.rng.integers(1, 4)
                sequence = [start + i * step for i in range(3)]
                next_val = sequence[-1] + step
                examples.append(f"[{','.join(map(str, sequence))}]->[{next_val}]")
            
            # Query pattern
            query_start = self.rng.integers(1, 5)
            query_step = self.rng.integers(1, 4)
            query_seq = [query_start + i * query_step for i in range(3)]
            query_next = query_seq[-1] + query_step
            
            text = ", ".join(examples) + f", [{','.join(map(str, query_seq))}]->"
            target_text = f"[{query_next}]"
            
            # Simple tokenization
            input_tokens = [ord(c) % 100 for c in text]
            target_tokens = [ord(c) % 100 for c in target_text]
            
            max_len = self.config.max_context_length
            if len(input_tokens) + len(target_tokens) > max_len:
                continue
                
            full_seq = input_tokens + target_tokens
            padding = [0] * (max_len - len(full_seq))
            padded = full_seq + padding
            
            inputs.append(padded[:-1])
            targets.append(padded[1:])
            masks.append([1] * (len(full_seq) - 1) + [0] * (max_len - len(full_seq)))
        
        return {
            'input_ids': jnp.array(inputs),
            'target_ids': jnp.array(targets), 
            'attention_mask': jnp.array(masks)
        }
    
    def generate_analogy_task(self, batch_size: int) -> Dict[str, jnp.ndarray]:
        """Generate analogy completion task.
        
        Format: cat:kitten, dog:puppy, bird:?
        """
        animal_babies = {
            'cat': 'kitten', 'dog': 'puppy', 'bird': 'chick',
            'cow': 'calf', 'horse': 'foal', 'pig': 'piglet',
            'sheep': 'lamb', 'duck': 'duckling', 'goose': 'gosling'
        }
        
        inputs, targets, masks = [], [], []
        animals = list(animal_babies.keys())
        
        for _ in range(batch_size):
            # Select examples
            selected = self.rng.choice(animals, self.config.num_examples + 1, replace=False)
            example_animals = selected[:-1]
            query_animal = selected[-1]
            
            examples = [f"{animal}:{animal_babies[animal]}" for animal in example_animals]
            text = ", ".join(examples) + f", {query_animal}:"
            target_text = animal_babies[query_animal]
            
            # Tokenize
            input_tokens = [ord(c) % 100 for c in text]
            target_tokens = [ord(c) % 100 for c in target_text]
            
            max_len = self.config.max_context_length
            if len(input_tokens) + len(target_tokens) > max_len:
                continue
                
            full_seq = input_tokens + target_tokens
            padding = [0] * (max_len - len(full_seq))
            padded = full_seq + padding
            
            inputs.append(padded[:-1])
            targets.append(padded[1:])
            masks.append([1] * (len(full_seq) - 1) + [0] * (max_len - len(full_seq)))
        
        return {
            'input_ids': jnp.array(inputs),
            'target_ids': jnp.array(targets),
            'attention_mask': jnp.array(masks)
        }


class RetrievalQAEvaluator:
    """Evaluator for retrieval-based question answering."""
    
    def __init__(self, vocab: Voc, config: EvalConfig, rng: np.random.Generator):
        self.vocab = vocab
        self.config = config
        self.rng = rng
    
    def generate_fact_retrieval_task(self, batch_size: int) -> Dict[str, jnp.ndarray]:
        """Generate fact retrieval task.
        
        Format: John lives in Paris. Mary lives in London. Where does John live?
        """
        names = ['John', 'Mary', 'Alice', 'Bob', 'Charlie', 'Diana']
        cities = ['Paris', 'London', 'Tokyo', 'Berlin', 'Madrid', 'Rome']
        
        inputs, targets, masks = [], [], []
        
        for _ in range(batch_size):
            # Generate facts
            n_facts = min(len(names), len(cities), self.config.num_examples + 1)
            selected_names = self.rng.choice(names, n_facts, replace=False)
            selected_cities = self.rng.choice(cities, n_facts, replace=False)
            
            facts = [f"{name} lives in {city}" for name, city in zip(selected_names, selected_cities)]
            
            # Query
            query_idx = self.rng.integers(n_facts)
            query_name = selected_names[query_idx] 
            answer = selected_cities[query_idx]
            
            text = ". ".join(facts) + f". Where does {query_name} live? "
            target_text = answer
            
            # Tokenize
            input_tokens = [ord(c) % 100 for c in text]
            target_tokens = [ord(c) % 100 for c in target_text]
            
            max_len = self.config.max_context_length
            if len(input_tokens) + len(target_tokens) > max_len:
                continue
                
            full_seq = input_tokens + target_tokens
            padding = [0] * (max_len - len(full_seq))
            padded = full_seq + padding
            
            inputs.append(padded[:-1])
            targets.append(padded[1:])
            masks.append([1] * (len(full_seq) - 1) + [0] * (max_len - len(full_seq)))
        
        return {
            'input_ids': jnp.array(inputs),
            'target_ids': jnp.array(targets),
            'attention_mask': jnp.array(masks)
        }


def create_evaluation_suite(vocab: Voc, rng: np.random.Generator) -> Dict[str, Any]:
    """Create a comprehensive evaluation suite."""
    
    config = EvalConfig()
    icl_eval = InContextLearningEvaluator(vocab, config, rng)
    qa_eval = RetrievalQAEvaluator(vocab, config, rng)
    
    return {
        'arithmetic': icl_eval.generate_arithmetic_task,
        'pattern_completion': icl_eval.generate_pattern_completion_task,
        'analogy': icl_eval.generate_analogy_task,
        'fact_retrieval': qa_eval.generate_fact_retrieval_task,
    }


def run_comprehensive_evaluation(experiment, num_samples: int = 100):
    """Run comprehensive evaluation on all downstream tasks."""
    
    # Create evaluation suite
    eval_rng = np.random.default_rng(42)
    eval_suite = create_evaluation_suite(experiment.copy_generator.vocab, eval_rng)
    
    results = {}
    
    for task_name, task_fn in eval_suite.items():
        print(f"Evaluating on {task_name}...")
        
        task_results = experiment.evaluate_downstream(
            lambda batch_size: task_fn(batch_size),
            num_eval_samples=num_samples
        )
        
        results[task_name] = task_results
        
        # Print results
        baseline_acc = task_results['baseline']['accuracy']
        pretrained_acc = task_results['pretrained']['accuracy'] 
        improvement = task_results['improvement']['accuracy']
        
        print(f"  Baseline accuracy: {baseline_acc:.4f}")
        print(f"  Pretrained accuracy: {pretrained_acc:.4f}")
        print(f"  Improvement: {improvement:.4f} ({improvement/baseline_acc*100:.1f}%)")
        print()
    
    return results


if __name__ == "__main__":
    from copy_task_experiment import run_copy_task_experiment
    
    # Run the full experiment
    experiment, pretrain_results = run_copy_task_experiment()
    
    # Run comprehensive evaluation
    eval_results = run_comprehensive_evaluation(experiment)
    
    print("Full experiment completed!")
    print("Summary of improvements:")
    for task_name, results in eval_results.items():
        improvement = results['improvement']['accuracy'] 
        baseline = results['baseline']['accuracy']
        print(f"{task_name}: {improvement/baseline*100:.1f}% improvement")
