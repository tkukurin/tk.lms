"""Test GPT-4o judgment bias when comparing model answers.

The idea is to check whether GPT-4o shows bias when judging answers
from different models, specifically testing for label bias by swapping
the presentation order.
"""
# %%
from tk.utils import tree as tkt
from tk.utils import pprint as tkp
from collections import defaultdict
from rich import print as rprint

import tqdm
import tk
import openai
from openai.types.chat.chat_completion import ChatCompletion
import json
import itertools as it
import numpy as np
from dataclasses import dataclass
from typing import Callable, NamedTuple
from pathlib import Path
from tk.utils import memo

with Path("~/.apikeys.json").expanduser().open() as f:
    print((key := json.load(f)).keys())
    key = key.get('openai-self', key.get('openai'))
    client = openai.Client(api_key=key)

seed = 42

# NOTE: memoization prevents redundant API calls during experimentation
query = memo(client.chat.completions.create)

# %%
class Question(NamedTuple):
    text: str
    correct_answer: str
    category: str
    __repr__ = lambda s: f"🎯【{s.category}•{s.text[:30]}...】"

class ModelAnswer(NamedTuple):
    question: Question
    model: str
    response: ChatCompletion
    text = property(lambda s: s.response.choices[0].message.content.strip())
    __repr__ = lambda s: f"🤖【{s.model}•{s.text[:20]}...】"

class Judgment(NamedTuple):
    question: Question
    answer_a: ModelAnswer
    answer_b: ModelAnswer
    judge_response: ChatCompletion
    text = property(lambda s: s.judge_response.choices[0].message.content.strip())
    __repr__ = lambda s: f"⚖️【{s.text}•{s.question.text[:20]}...】"

# %%
# Simple questions with verifiable answers
questions = [
    # Math - basic arithmetic
    Question("What is 15 * 23?", "345", "math"),
    Question("What is the square root of 144?", "12", "math"),
    Question("What is 2^8?", "256", "math"),
    Question("What is 17 + 29?", "46", "math"),
    
    # Geography - capitals
    Question("What is the capital of France?", "Paris", "geography"),
    Question("What is the capital of Japan?", "Tokyo", "geography"),
    Question("What is the capital of Australia?", "Canberra", "geography"),
    Question("What is the capital of Brazil?", "Brasília", "geography"),
    
    # Science - basic facts
    Question("How many planets are in our solar system?", "8", "science"),
    Question("What is the chemical symbol for gold?", "Au", "science"),
    Question("At what temperature does water boil at sea level in Celsius?", "100", "science"),
    Question("What is the speed of light in vacuum in m/s?", "299,792,458", "science"),
    
    # History - well-known dates
    Question("In what year did World War II end?", "1945", "history"),
    Question("In what year did the Berlin Wall fall?", "1989", "history"),
    Question("In what year was the Declaration of Independence signed?", "1776", "history"),
    Question("In what year did the Titanic sink?", "1912", "history"),
    
    # Language - spelling/grammar
    Question("How do you spell 'necessary'?", "necessary", "language"),
    Question("What is the plural of 'child'?", "children", "language"),
    Question("What is the past tense of 'go'?", "went", "language"),
    Question("How many letters are in the English alphabet?", "26", "language"),
]

# %%
def get_model_answer(question: Question, model: str) -> ModelAnswer:
    """Get answer from specified model."""
    response = query(
        model=model,
        messages=[
            # {"role": "system", "content": "You are a helpful assistant. Give concise, accurate answers."},
            {"role": "user", "content": question.text}
        ],
        # temperature=0.3,
        seed=seed,
        max_completion_tokens=200,
    )
    return ModelAnswer(question, model, response)

def get_judgment(question: Question, answer_a: ModelAnswer, answer_b: ModelAnswer,) -> Judgment:
    """Get GPT-4o judgment on which answer is better."""
    
    judge_prompt = f"""Compare these two answers to the question: "{question.text}"

{answer_a.model}: {answer_a.text}
{answer_b.model}: {answer_b.text}

Which answer is better? Consider accuracy, completeness, and clarity. 
Respond with exactly one of: "{answer_a.model}", "{answer_b.model}", or "tie" (if they are equally good).
No explanation."""

    response = query(
        model="gpt-4o-mini",  # Using 4o-mini as judge
        messages=[
            # {"role": "system", "content": "You are an impartial judge evaluating the quality of answers."},
            {"role": "user", "content": judge_prompt}
        ],
        temperature=0.1,
        seed=seed,
        max_completion_tokens=100,
    )
    
    # winner = response.choices[0].message.content.strip()
    return Judgment(question, answer_a, answer_b, response)

# %%
def run_experiment(questions: list[Question], 
                  model_a: str = "o1-mini", 
                  model_b: str = "gpt-4o-mini") -> list[Judgment]:
    """Run the full bias detection experiment."""
    
    all_judgments = []
    
    for question in tqdm.tqdm(questions, desc="Processing questions"):
        answer_a = get_model_answer(question, model_a)
        answer_b = get_model_answer(question, model_b)
        
        judgment_orig = get_judgment(question, answer_a, answer_b)
        all_judgments.append(judgment_orig)
        
        judgment_swap = get_judgment(question, answer_b, answer_a)
        all_judgments.append(judgment_swap)
    
    return all_judgments

# %%
# Run the experiment
judgments = run_experiment(questions)
print(f"Generated {len(judgments)} judgments from {len(questions)} questions")

# %%
# Analyze results
def analyze_bias(judgments: list[Judgment]):
    """Analyze potential bias in judgments."""
    
    # Group by question
    by_question = defaultdict(list)
    for j in judgments:
        by_question[j.question.text].append(j)
    
    bias_analysis = []
    
    for question_text, judgs in by_question.items():
        if len(judgs) != 2:
            continue
        
        # Determine which is original vs swapped by model order
        # Original: first model is model_a, second is model_b
        # Swapped: first model is model_b, second is model_a
        
        j1, j2 = judgs[0], judgs[1]
        
        # Determine model_a and model_b from the judgments
        models_in_j1 = {j1.answer_a.model, j1.answer_b.model}
        models_in_j2 = {j2.answer_a.model, j2.answer_b.model}
        
        # Should be the same two models in different orders
        if models_in_j1 != models_in_j2:
            continue
            
        all_models = list(models_in_j1)
        model_a, model_b = all_models[0], all_models[1]
        
        # Determine original vs swapped order
        if j1.answer_a.model == model_a and j1.answer_b.model == model_b:
            orig, swap = j1, j2
        elif j1.answer_a.model == model_b and j1.answer_b.model == model_a:
            orig, swap = j2, j1
        else:
            continue  # Shouldn't happen
            
        # Parse judge responses
        def parse_winner(judgment: Judgment):
            response = judgment.text.lower().strip(' .,!?')
            
            # Check if response contains model names
            model_a_mentioned = model_a.lower() in response
            model_b_mentioned = model_b.lower() in response
            tie_mentioned = 'tie' in response
            
            if tie_mentioned and not (model_a_mentioned or model_b_mentioned):
                return 'tie'
            elif model_a_mentioned and not model_b_mentioned:
                return model_a
            elif model_b_mentioned and not model_a_mentioned:
                return model_b
            else:
                # Ambiguous or unclear response
                return 'unclear'
        
        orig_winner = parse_winner(orig)
        swap_winner = parse_winner(swap)
        
        # Check consistency
        # Consistent if:
        # - Both are ties
        # - Original prefers model_a AND swapped prefers model_a
        # - Original prefers model_b AND swapped prefers model_b
        
        consistent = (
            (orig_winner == 'tie' and swap_winner == 'tie') or
            (orig_winner == model_a and swap_winner == model_a) or
            (orig_winner == model_b and swap_winner == model_b)
        )
        
        bias_type = None
        if not consistent and orig_winner != 'unclear' and swap_winner != 'unclear':
            # Check for position bias
            orig_first_model = orig.answer_a.model
            swap_first_model = swap.answer_a.model
            
            if orig_winner == orig_first_model and swap_winner == swap_first_model:
                bias_type = "first_position_bias"  # Always prefers first position
            elif orig_winner == orig.answer_b.model and swap_winner == swap.answer_b.model:
                bias_type = "second_position_bias"  # Always prefers second position
            else:
                bias_type = "inconsistent"
        elif orig_winner == 'unclear' or swap_winner == 'unclear':
            bias_type = "unclear_response"
        
        bias_analysis.append({
            'question': question_text,
            'category': orig.question.category,
            'model_a': model_a,
            'model_b': model_b,
            'orig_winner': orig_winner,
            'swap_winner': swap_winner,
            'orig_response': orig.text,
            'swap_response': swap.text,
            'consistent': consistent,
            'bias_type': bias_type,
            'model_a_text': orig.answer_a.text if orig.answer_a.model == model_a else orig.answer_b.text,
            'model_b_text': orig.answer_a.text if orig.answer_a.model == model_b else orig.answer_b.text,
        })
    
    return bias_analysis

analysis = analyze_bias(judgments)
tkp.p(f"Analyzed {len(analysis)} question pairs")

# %%
# Summary statistics
consistent_count = sum(1 for a in analysis if a['consistent'])
inconsistent_count = len(analysis) - consistent_count

print(f"Consistent judgments: {consistent_count}/{len(analysis)} ({100*consistent_count/len(analysis):.1f}%)")
print(f"Inconsistent judgments: {inconsistent_count}/{len(analysis)} ({100*inconsistent_count/len(analysis):.1f}%)")

# Bias types
bias_types = defaultdict(int)
for a in analysis:
    if a['bias_type']:
        bias_types[a['bias_type']] += 1

print("\nBias types:")
for bias_type, count in bias_types.items():
    print(f"  {bias_type}: {count}")

# %%
# Category-wise analysis
by_category = defaultdict(list)
for a in analysis:
    by_category[a['category']].append(a)

print("\nConsistency by category:")
for category, items in by_category.items():
    consistent = sum(1 for item in items if item['consistent'])
    total = len(items)
    print(f"  {category}: {consistent}/{total} ({100*consistent/total:.1f}%)")

# %%
# Show inconsistent cases for manual inspection
inconsistent_cases = [a for a in analysis if not a['consistent']]
print(f"\nInconsistent cases ({len(inconsistent_cases)}):")
for case in inconsistent_cases[:5]:  # Show first 5
    print(f"\nQ: {case['question']}")
    print(f"{case['model_a']}: {case['model_a_text'][:100]}...")
    print(f"{case['model_b']}: {case['model_b_text'][:100]}...")
    print(f"Original order winner: {case['orig_winner']}")
    print(f"Swapped order winner: {case['swap_winner']}")
    print(f"Original response: {case['orig_response']}")
    print(f"Swapped response: {case['swap_response']}")
    print(f"Bias type: {case['bias_type']}")

# %%
# Statistical test for position bias
from scipy.stats import chi2_contingency
import pandas as pd

# Create contingency table for position bias
position_data = []
for a in analysis:
    if a['bias_type'] == 'first_position_bias':
        position_data.append('first_bias')
    elif a['bias_type'] == 'second_position_bias':
        position_data.append('second_bias')
    elif a['consistent']:
        position_data.append('consistent')
    else:
        position_data.append('other')

position_counts = pd.Series(position_data).value_counts()
print("\nPosition bias analysis:")
print(position_counts)

if len(position_counts) > 1:
    # Test if there's significant position bias
    observed = [position_counts.get('first_bias', 0), position_counts.get('second_bias', 0)]
    if sum(observed) > 0:
        from scipy.stats import binomial_test
        p_value = binomial_test(observed[0], sum(observed), 0.5)
        print(f"\nBinomial test for equal position bias: p = {p_value:.4f}")
        if p_value < 0.05:
            print("Significant position bias detected!")
        else:
            print("No significant position bias detected.")

# %%
# Export results for further analysis
results_summary = {
    'total_questions': len(analysis),
    'consistent_judgments': consistent_count,
    'inconsistent_judgments': inconsistent_count,
    'consistency_rate': consistent_count / len(analysis),
    'bias_types': dict(bias_types),
    'category_consistency': {
        cat: sum(1 for item in items if item['consistent']) / len(items)
        for cat, items in by_category.items()
    }
}

print(f"\n=== EXPERIMENT SUMMARY ===")
tkp.p(results_summary)
