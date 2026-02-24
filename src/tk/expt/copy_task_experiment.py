"""
Experimental framework for testing copy-task pre-training hypothesis.

This module implements:
1. Copy-task data generation for pre-training
2. Training pipeline with copy-task pre-training
3. Evaluation on downstream tasks requiring retrieval capabilities
"""

import jax
import jax.numpy as jnp
import numpy as np
from typing import Dict, List, Tuple, Any, Callable, Optional
from dataclasses import dataclass
import itertools as it
from abc import ABC, abstractmethod

from tk.models.gpt2 import GPT, GPTConfig
from tk.models.train import create_train_state, train_step, eval_step
from tk.utils.data.vocab import Voc


@dataclass
class CopyTaskConfig:
    """Configuration for copy task generation."""
    vocab_size: int = 1000
    max_seq_len: int = 64
    min_seq_len: int = 8
    num_key_value_pairs: int = 5
    pattern_length: int = 4
    copy_prob: float = 0.8  # probability of requiring copying vs. generation


class CopyTaskGenerator:
    """Generates various copy tasks for pre-training."""
    
    def __init__(self, config: CopyTaskConfig, rng: np.random.Generator):
        self.config = config
        self.rng = rng
        self.vocab = self._create_vocab()
        
    def _create_vocab(self) -> Voc:
        """Create vocabulary for copy tasks."""
        special_tokens = ['<pad>', '<s>', '</s>', '<sep>', '<copy>', '<key>', '<val>']
        regular_tokens = [f'tok_{i}' for i in range(self.config.vocab_size - len(special_tokens))]
        
        return Voc.make({
            'special': special_tokens,
            'regular': regular_tokens,
        })
    
    def generate_simple_copy_task(self) -> Tuple[List[int], List[int]]:
        """Generate a simple sequence copying task.
        
        Format: <s> [sequence] <copy> [sequence] </s>
        """
        seq_len = self.rng.integers(self.config.min_seq_len, self.config.max_seq_len)
        sequence = [
            self.vocab[f'tok_{self.rng.integers(len(self.vocab.ofkind("regular")))}']
            for _ in range(seq_len)
        ]
        
        input_seq = (
            [self.vocab['<s>']] + 
            sequence + 
            [self.vocab['<copy>']]
        )
        target_seq = sequence + [self.vocab['</s>']]
        
        return input_seq, target_seq
    
    def generate_key_value_retrieval(self) -> Tuple[List[int], List[int]]:
        """Generate key-value retrieval task.
        
        Format: <s> <key>k1<val>v1 <key>k2<val>v2 ... <key>query_key <val>
        Target: corresponding value </s>
        """
        n_pairs = self.rng.integers(2, self.config.num_key_value_pairs + 1)
        
        # Generate key-value pairs
        keys = [f'tok_{self.rng.integers(len(self.vocab.ofkind("regular")))}' for _ in range(n_pairs)]
        values = [f'tok_{self.rng.integers(len(self.vocab.ofkind("regular")))}' for _ in range(n_pairs)]
        
        # Choose query key
        query_idx = self.rng.integers(n_pairs)
        query_key = keys[query_idx]
        target_value = values[query_idx]
        
        # Build input sequence
        input_seq = [self.vocab['<s>']]
        for key, val in zip(keys, values):
            input_seq.extend([
                self.vocab['<key>'], 
                self.vocab[key],
                self.vocab['<val>'], 
                self.vocab[val]
            ])
        
        # Add query
        input_seq.extend([
            self.vocab['<key>'], 
            self.vocab[query_key], 
            self.vocab['<val>']
        ])
        
        target_seq = [self.vocab[target_value], self.vocab['</s>']]
        
        return input_seq, target_seq
    
    def generate_pattern_completion(self) -> Tuple[List[int], List[int]]:
        """Generate pattern completion task.
        
        Format: <s> [pattern] [pattern] [partial_pattern]
        Target: [completion] </s>
        """
        pattern = [
            self.vocab[f'tok_{self.rng.integers(len(self.vocab.ofkind("regular")))}']
            for _ in range(self.config.pattern_length)
        ]
        
        # Show pattern twice, then partial
        partial_len = self.rng.integers(1, self.config.pattern_length)
        partial_pattern = pattern[:partial_len]
        completion = pattern[partial_len:]
        
        input_seq = (
            [self.vocab['<s>']] + 
            pattern + 
            pattern + 
            partial_pattern
        )
        target_seq = completion + [self.vocab['</s>']]
        
        return input_seq, target_seq
    
    def generate_batch(self, batch_size: int) -> Dict[str, jnp.ndarray]:
        """Generate a batch of mixed copy tasks."""
        inputs, targets = [], []
        attention_masks = []
        
        for _ in range(batch_size):
            task_type = self.rng.choice(['simple_copy', 'key_value', 'pattern'])
            
            if task_type == 'simple_copy':
                inp, tgt = self.generate_simple_copy_task()
            elif task_type == 'key_value':
                inp, tgt = self.generate_key_value_retrieval()
            else:  # pattern
                inp, tgt = self.generate_pattern_completion()
            
            # Pad to max length
            max_len = self.config.max_seq_len * 2  # Allow for input + target
            full_seq = inp + tgt
            
            if len(full_seq) > max_len:
                continue  # Skip if too long
                
            # Pad
            padding_len = max_len - len(full_seq)
            padded_seq = full_seq + [self.vocab['<pad>']] * padding_len
            
            # Create input (everything except last token) and target (shifted by 1)
            input_ids = padded_seq[:-1]
            target_ids = padded_seq[1:]
            
            # Create attention mask
            attention_mask = [1] * (len(full_seq) - 1) + [0] * (max_len - len(full_seq))
            
            inputs.append(input_ids)
            targets.append(target_ids)
            attention_masks.append(attention_mask)
        
        return {
            'input_ids': jnp.array(inputs),
            'target_ids': jnp.array(targets),
            'attention_mask': jnp.array(attention_masks)
        }


class CopyTaskExperiment:
    """Main experiment class for copy-task pre-training."""
    
    def __init__(
        self, 
        model_config: GPTConfig,
        copy_config: CopyTaskConfig,
        rng: jax.Array
    ):
        self.model_config = model_config
        self.copy_config = copy_config
        self.rng = rng
        
        # Initialize copy task generator
        self.copy_generator = CopyTaskGenerator(
            copy_config, 
            np.random.default_rng(jax.device_get(rng))
        )
        
        # Update model config with copy task vocabulary
        self.model_config = GPTConfig(
            **{**model_config.__dict__, 'vocab_size': len(self.copy_generator.vocab)}
        )
        
        # Initialize models
        self.baseline_model = GPT(config=self.model_config)
        self.pretrained_model = GPT(config=self.model_config)
        
    def pretrain_copy_tasks(
        self, 
        num_steps: int, 
        batch_size: int, 
        learning_rate: float = 1e-4
    ) -> Dict[str, Any]:
        """Pre-train model on copy tasks."""
        
        # Initialize training state
        rng, init_rng = jax.random.split(self.rng)
        state = create_train_state(
            init_rng, 
            self.pretrained_model, 
            learning_rate,
            (batch_size, self.copy_config.max_seq_len * 2 - 1)
        )
        
        losses = []
        
        for step in range(num_steps):
            # Generate batch
            batch = self.copy_generator.generate_batch(batch_size)
            
            # Training step
            rng, dropout_rng = jax.random.split(rng)
            state, loss = train_step(
                batch['input_ids'],
                batch['target_ids'], 
                batch['attention_mask'],
                state,
                dropout_rng
            )
            
            losses.append(float(loss))
            
            if step % 100 == 0:
                print(f"Copy pre-training step {step}, loss: {loss:.4f}")
        
        # Store the pre-trained state
        self.pretrained_state = state
        
        return {'losses': losses}
    
    def evaluate_downstream(
        self, 
        downstream_task_fn: Callable,
        num_eval_samples: int = 100
    ) -> Dict[str, Any]:
        """Evaluate both baseline and pre-trained models on downstream tasks."""
        
        # Initialize baseline model (no pre-training)
        rng, init_rng = jax.random.split(self.rng)
        baseline_state = create_train_state(
            init_rng,
            self.baseline_model,
            1e-4
        )
        
        # Evaluate both models
        baseline_results = self._evaluate_model(baseline_state, downstream_task_fn, num_eval_samples)
        pretrained_results = self._evaluate_model(self.pretrained_state, downstream_task_fn, num_eval_samples)
        
        return {
            'baseline': baseline_results,
            'pretrained': pretrained_results,
            'improvement': {
                k: pretrained_results[k] - baseline_results[k] 
                for k in baseline_results.keys()
            }
        }
    
    def _evaluate_model(
        self, 
        state, 
        task_fn: Callable,
        num_samples: int
    ) -> Dict[str, float]:
        """Evaluate a single model on downstream task."""
        
        total_loss = 0.0
        total_accuracy = 0.0
        
        for _ in range(num_samples):
            # Generate task sample
            batch = task_fn(batch_size=1)
            
            # Evaluate
            rng, dropout_rng = jax.random.split(self.rng)
            probs, metrics = eval_step(
                batch['input_ids'],
                batch['target_ids'],
                batch['attention_mask'], 
                state,
                dropout_rng
            )
            
            total_loss += metrics['loss']
            total_accuracy += metrics['acc_all']
        
        return {
            'loss': total_loss / num_samples,
            'accuracy': total_accuracy / num_samples
        }


# Example downstream task generators
class DownstreamTaskGenerator:
    """Generate downstream tasks that benefit from retrieval capabilities."""
    
    def __init__(self, vocab: Voc, rng: np.random.Generator):
        self.vocab = vocab
        self.rng = rng
    
    def few_shot_classification_task(self, batch_size: int) -> Dict[str, jnp.ndarray]:
        """Generate few-shot text classification examples."""
        # Implementation would depend on specific classification task
        # This is a placeholder structure
        return {
            'input_ids': jnp.zeros((batch_size, 64), dtype=jnp.int32),
            'target_ids': jnp.zeros((batch_size, 64), dtype=jnp.int32), 
            'attention_mask': jnp.ones((batch_size, 64), dtype=jnp.int32)
        }
    
    def in_context_learning_task(self, batch_size: int) -> Dict[str, jnp.ndarray]:
        """Generate in-context learning examples."""
        # Examples: A->B, C->D, E->? format
        return {
            'input_ids': jnp.zeros((batch_size, 64), dtype=jnp.int32),
            'target_ids': jnp.zeros((batch_size, 64), dtype=jnp.int32),
            'attention_mask': jnp.ones((batch_size, 64), dtype=jnp.int32)
        }


def run_copy_task_experiment():
    """Main function to run the copy-task experiment."""
    
    # Configuration
    model_config = GPTConfig(
        vocab_size=1000,  # Will be updated by experiment
        block_size=128,
        num_layers=6,
        num_heads=8,
        num_embeds=256,
        dropout_rate=0.1
    )
    
    copy_config = CopyTaskConfig(
        vocab_size=500,
        max_seq_len=32,
        num_key_value_pairs=5
    )
    
    # Initialize experiment
    rng = jax.random.PRNGKey(42)
    experiment = CopyTaskExperiment(model_config, copy_config, rng)
    
    # Pre-train on copy tasks
    print("Starting copy-task pre-training...")
    pretrain_results = experiment.pretrain_copy_tasks(
        num_steps=1000,
        batch_size=8,
        learning_rate=1e-4
    )
    
    print(f"Pre-training completed. Final loss: {pretrain_results['losses'][-1]:.4f}")
    
    # Evaluate on downstream tasks
    # (This would need specific downstream task implementations)
    
    return experiment, pretrain_results


if __name__ == "__main__":
    experiment, results = run_copy_task_experiment()
    print("Experiment completed!")
