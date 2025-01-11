"""wip quick plot module

unchecked, lightly bugfixed Claude generation.
"""
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path

def plot_agent_agreement(agent_responses, save_path=None):
    """Plot agreement percentage between agents across rounds."""
    agent_responses = [agent for agent in agent_responses if agent]
    rounds = len(agent_responses[0])
    
    agreements = []
    for r in range(rounds):
        round_answers = [agent[r] for agent in agent_responses]
        unique_answers = len(set(round_answers))
        agreement_pct = (len(round_answers) - unique_answers + 1) / len(round_answers) * 100
        agreements.append(agreement_pct)
    
    fig = plt.figure(figsize=(10, 6))
    plt.plot(range(1, rounds + 1), agreements, 'bo-')
    plt.xlabel('Round')
    plt.ylabel('Agreement Percentage')
    plt.title('Agent Agreement Percentage per Round')
    plt.ylim(0, 100)
    plt.grid(True)
    
    if save_path:
        plt.savefig(save_path)
    plt.close()
    return fig

def plot_accuracy_progression(scores, rounds, save_path=None):
    """Plot accuracy progression over rounds."""
    cumulative_accuracy = np.cumsum(scores) / np.arange(1, len(scores) + 1)
    
    fig = plt.figure(figsize=(10, 6))
    plt.plot(range(1, len(scores) + 1), cumulative_accuracy, 'r-')
    plt.xlabel('Problem Number')
    plt.ylabel('Cumulative Accuracy')
    plt.title('Performance Progression')
    plt.grid(True)
    
    if save_path:
        plt.savefig(save_path)
    plt.close()
    return fig

def plot_agent_answers(text_answers_history, save_path=None):
    """Plot individual agent answers across problems."""
    problems = len(text_answers_history)
    fig = plt.figure(figsize=(12, 6))
    
    for problem_idx, answers in enumerate(text_answers_history):
        x = [problem_idx] * len(answers)
        plt.scatter(x, answers, alpha=0.5)
    
    plt.xlabel('Problem Number')
    plt.ylabel('Agent Answers')
    plt.title('Agent Answers Distribution')
    plt.grid(True)
    
    if save_path:
        plt.savefig(save_path)
    plt.close()
    return fig
