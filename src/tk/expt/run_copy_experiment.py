"""
Configuration and runner for copy-task pre-training experiment.

This script demonstrates how to run the complete experiment comparing:
1. Baseline model (no copy-task pre-training)  
2. Copy-task pre-trained model

Usage:
    python -m tk.expt.run_copy_experiment --config=copy_task --steps=1000 --eval_samples=100
"""

import argparse
import json
import os
from pathlib import Path
from dataclasses import asdict

import jax
import numpy as np

from tk.models.gpt2 import GPTConfig
from tk.expt.copy_task_experiment import CopyTaskConfig, CopyTaskExperiment
from tk.expt.eval_benchmarks import run_comprehensive_evaluation


def create_experiment_config():
    """Create default experiment configuration."""
    
    model_config = GPTConfig(
        vocab_size=1000,  # Will be updated by experiment
        block_size=128,
        num_layers=4,     # Smaller for faster experimentation
        num_heads=8,
        num_embeds=256,
        dropout_rate=0.1,
        use_bias=True
    )
    
    copy_config = CopyTaskConfig(
        vocab_size=500,
        max_seq_len=32,
        min_seq_len=8,
        num_key_value_pairs=5,
        pattern_length=4,
        copy_prob=0.8
    )
    
    return {
        'model_config': model_config,
        'copy_config': copy_config,
        'training': {
            'pretrain_steps': 1000,
            'batch_size': 8,
            'learning_rate': 1e-4,
            'eval_samples': 100
        }
    }


def run_experiment(config: dict, save_results: bool = True):
    """Run the complete copy-task pre-training experiment."""
    
    print("=" * 60)
    print("COPY-TASK PRE-TRAINING EXPERIMENT")
    print("=" * 60)
    print()
    
    # Initialize experiment
    rng = jax.random.PRNGKey(42)
    experiment = CopyTaskExperiment(
        config['model_config'], 
        config['copy_config'], 
        rng
    )
    
    print(f"Model vocabulary size: {len(experiment.copy_generator.vocab)}")
    print(f"Model configuration: {experiment.model_config}")
    print()
    
    # Phase 1: Copy-task pre-training
    print("PHASE 1: Copy-task Pre-training")
    print("-" * 40)
    
    pretrain_results = experiment.pretrain_copy_tasks(
        num_steps=config['training']['pretrain_steps'],
        batch_size=config['training']['batch_size'],
        learning_rate=config['training']['learning_rate']
    )
    
    initial_loss = pretrain_results['losses'][0]
    final_loss = pretrain_results['losses'][-1]
    improvement = (initial_loss - final_loss) / initial_loss * 100
    
    print(f"Initial loss: {initial_loss:.4f}")
    print(f"Final loss: {final_loss:.4f}")
    print(f"Loss reduction: {improvement:.1f}%")
    print()
    
    # Phase 2: Downstream evaluation
    print("PHASE 2: Downstream Task Evaluation")
    print("-" * 40)
    
    eval_results = run_comprehensive_evaluation(
        experiment, 
        num_samples=config['training']['eval_samples']
    )
    
    # Summarize results
    print("EXPERIMENT SUMMARY")
    print("=" * 60)
    print()
    
    overall_improvement = 0
    significant_improvements = 0
    
    for task_name, results in eval_results.items():
        baseline_acc = results['baseline']['accuracy']
        pretrained_acc = results['pretrained']['accuracy']
        improvement = results['improvement']['accuracy']
        improvement_pct = (improvement / baseline_acc * 100) if baseline_acc > 0 else 0
        
        print(f"{task_name.upper().replace('_', ' ')}:")
        print(f"  Baseline:    {baseline_acc:.4f}")
        print(f"  Pre-trained: {pretrained_acc:.4f}")
        print(f"  Improvement: {improvement_pct:+.1f}%")
        
        if improvement_pct > 5:  # Consider >5% improvement as significant
            significant_improvements += 1
        
        overall_improvement += improvement_pct
        print()
    
    avg_improvement = overall_improvement / len(eval_results)
    print(f"OVERALL RESULTS:")
    print(f"  Average improvement: {avg_improvement:.1f}%")
    print(f"  Tasks with >5% improvement: {significant_improvements}/{len(eval_results)}")
    
    # Hypothesis validation
    print("\nHYPOTHESIS VALIDATION:")
    if avg_improvement > 3 and significant_improvements >= len(eval_results) // 2:
        print("✅ HYPOTHESIS SUPPORTED: Copy-task pre-training improves downstream performance")
    elif avg_improvement > 0:
        print("⚠️  HYPOTHESIS PARTIALLY SUPPORTED: Some improvement observed")
    else:
        print("❌ HYPOTHESIS NOT SUPPORTED: No significant improvement observed")
    
    # Save results
    if save_results:
        results_data = {
            'config': {
                'model_config': asdict(config['model_config']),
                'copy_config': asdict(config['copy_config']),
                'training': config['training']
            },
            'pretrain_results': pretrain_results,
            'eval_results': {
                task: {
                    'baseline_accuracy': float(results['baseline']['accuracy']),
                    'pretrained_accuracy': float(results['pretrained']['accuracy']),
                    'improvement': float(results['improvement']['accuracy']),
                    'improvement_pct': float(results['improvement']['accuracy'] / results['baseline']['accuracy'] * 100)
                    if results['baseline']['accuracy'] > 0 else 0.0
                }
                for task, results in eval_results.items()
            },
            'summary': {
                'average_improvement_pct': avg_improvement,
                'significant_improvements': significant_improvements,
                'total_tasks': len(eval_results)
            }
        }
        
        output_dir = Path("results")
        output_dir.mkdir(exist_ok=True)
        
        with open(output_dir / "copy_task_experiment_results.json", "w") as f:
            json.dump(results_data, f, indent=2)
        
        print(f"\nResults saved to: {output_dir / 'copy_task_experiment_results.json'}")
    
    return experiment, eval_results


def main():
    parser = argparse.ArgumentParser(description="Run copy-task pre-training experiment")
    parser.add_argument("--steps", type=int, default=1000, help="Number of pre-training steps")
    parser.add_argument("--batch_size", type=int, default=8, help="Batch size")
    parser.add_argument("--eval_samples", type=int, default=100, help="Number of evaluation samples")
    parser.add_argument("--lr", type=float, default=1e-4, help="Learning rate")
    parser.add_argument("--save_results", action="store_true", default=True, help="Save results to file")
    
    args = parser.parse_args()
    
    # Create configuration
    config = create_experiment_config()
    config['training'].update({
        'pretrain_steps': args.steps,
        'batch_size': args.batch_size,
        'learning_rate': args.lr,
        'eval_samples': args.eval_samples
    })
    
    # Run experiment
    experiment, results = run_experiment(config, args.save_results)
    
    return experiment, results


if __name__ == "__main__":
    main()
