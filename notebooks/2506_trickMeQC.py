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
# Subjective questions with combinatorial "flair" criteria
class FlairAxis(NamedTuple):
    name: str
    description: str
    __repr__ = lambda s: f"🎨【{s.name}】"

flair_axes = [
    FlairAxis("verbose", "be very detailed and comprehensive"),
    FlairAxis("concise", "be brief and to the point"),
    FlairAxis("creative", "be imaginative and original"),
    FlairAxis("formal", "use professional and academic language"),
    FlairAxis("casual", "use conversational and relaxed language"),
    FlairAxis("adjective-rich", "use lots of descriptive adjectives"),
    FlairAxis("example-heavy", "provide concrete examples and illustrations"),
    FlairAxis("enthusiastic", "show excitement and energy"),
    FlairAxis("technical", "use technical terminology and jargon"),
    FlairAxis("metaphorical", "use metaphors and analogies"),
]

base_topics = [
    "explain how to make coffee",
    "describe the benefits of exercise", 
    "explain why people should read books",
    "describe how to learn a new language",
    "explain the importance of sleep",
    "describe how to be more productive",
    "explain why teamwork matters",
    "describe how to handle stress",
    "explain the value of traveling",
    "describe how to build good habits",
]

def create_flair_questions() -> list[Question]:
    """Create questions with combinatorial flair criteria."""
    questions = []
    
    # Single flair axis questions
    for i, (topic, flair) in enumerate(it.product(base_topics[:5], flair_axes[:6])):
        question_text = f"{topic.capitalize()} ({flair.description})"
        questions.append(Question(question_text, flair.name, "single_flair"))
    
    # Double flair axis questions (combinations)
    flair_pairs = list(it.combinations(flair_axes[:6], 2))
    for i, (topic, (flair1, flair2)) in enumerate(it.product(base_topics[5:], flair_pairs[:5])):
        question_text = f"{topic.capitalize()} ({flair1.description} and {flair2.description})"
        expected = f"{flair1.name}+{flair2.name}"
        questions.append(Question(question_text, expected, "double_flair"))
    
    return questions

questions = create_flair_questions()
print(f"Created {len(questions)} flair-based questions")
for q in questions[:3]:
    print(f"  {q}")

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

def get_judgment_named(question: Question, answer_a: ModelAnswer, answer_b: ModelAnswer) -> Judgment:
    """Get GPT-4o judgment with model names visible."""
    
    judge_prompt = f"""Compare these two answers to: "{question.text}"

{answer_a.model}: {answer_a.text}

{answer_b.model}: {answer_b.text}

Which answer better follows the specified style/criteria? Consider how well each response adheres to the requested approach.
Respond with exactly one of: "{answer_a.model}", "{answer_b.model}", or "tie".
No explanation."""

    response = query(
        model="gpt-4o-mini",
        messages=[
            {"role": "user", "content": judge_prompt}
        ],
        temperature=0.1,
        seed=seed,
        max_completion_tokens=100,
    )
    
    return Judgment(question, answer_a, answer_b, response)

def get_judgment_anonymous(question: Question, answer_a: ModelAnswer, answer_b: ModelAnswer) -> Judgment:
    """Get GPT-4o judgment with anonymous A/B labels."""
    
    judge_prompt = f"""Compare these two answers to: "{question.text}"

Answer A: {answer_a.text}

Answer B: {answer_b.text}

Which answer better follows the specified style/criteria? Consider how well each response adheres to the requested approach.
Respond with exactly one of: "A", "B", or "tie".
No explanation."""

    response = query(
        model="gpt-4o-mini", 
        messages=[
            {"role": "user", "content": judge_prompt}
        ],
        temperature=0.1,
        seed=seed,
        max_completion_tokens=100,
    )
    
    return Judgment(question, answer_a, answer_b, response)

# %%
def run_experiment(questions: list[Question], 
                  model_a: str = "o1-mini", 
                  model_b: str = "gpt-4o-mini") -> tuple[list[Judgment], list[Judgment]]:
    """Run the full bias detection experiment with both named and anonymous evaluation."""
    
    named_judgments = []
    anonymous_judgments = []
    
    for question in tqdm.tqdm(questions, desc="Processing questions"):
        answer_a = get_model_answer(question, model_a)
        answer_b = get_model_answer(question, model_b)
        
        # Named evaluation (original and swapped)
        named_orig = get_judgment_named(question, answer_a, answer_b)
        named_judgments.append(named_orig)
        
        named_swap = get_judgment_named(question, answer_b, answer_a)
        named_judgments.append(named_swap)
        
        # Anonymous evaluation (original and swapped)
        anon_orig = get_judgment_anonymous(question, answer_a, answer_b)
        anonymous_judgments.append(anon_orig)
        
        anon_swap = get_judgment_anonymous(question, answer_b, answer_a)
        anonymous_judgments.append(anon_swap)
    
    return named_judgments, anonymous_judgments

# %%
# Run the experiment
named_judgments, anonymous_judgments = run_experiment(questions)
print(f"Generated {len(named_judgments)} named judgments and {len(anonymous_judgments)} anonymous judgments")

# %%
# Analyze results for named judgments
def analyze_bias_named(judgments: list[Judgment]):
    """Analyze potential bias in named judgments."""
    
    # Group by question
    by_question = defaultdict(list)
    for j in judgments:
        by_question[j.question.text].append(j)
    
    bias_analysis = []
    win_counts = defaultdict(int)
    
    for question_text, judgs in by_question.items():
        if len(judgs) != 2:
            continue
        
        j1, j2 = judgs[0], judgs[1]
        
        # Determine model_a and model_b from the judgments
        models_in_j1 = {j1.answer_a.model, j1.answer_b.model}
        models_in_j2 = {j2.answer_a.model, j2.answer_b.model}
        
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
            continue
            
        # Parse judge responses
        def parse_winner_named(judgment: Judgment):
            response = judgment.text.lower().strip(' .,!?')
            
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
                return 'unclear'
        
        orig_winner = parse_winner_named(orig)
        swap_winner = parse_winner_named(swap)
        
        # Count wins for this question pair
        if orig_winner in [model_a, model_b]:
            win_counts[orig_winner] += 0.5  # Half point for each judgment
        if swap_winner in [model_a, model_b]:
            win_counts[swap_winner] += 0.5
        
        # Check consistency
        consistent = (
            (orig_winner == 'tie' and swap_winner == 'tie') or
            (orig_winner == model_a and swap_winner == model_a) or
            (orig_winner == model_b and swap_winner == model_b)
        )
        
        bias_type = None
        if not consistent and orig_winner != 'unclear' and swap_winner != 'unclear':
            orig_first_model = orig.answer_a.model
            swap_first_model = swap.answer_a.model
            
            if orig_winner == orig_first_model and swap_winner == swap_first_model:
                bias_type = "first_position_bias"
            elif orig_winner == orig.answer_b.model and swap_winner == swap.answer_b.model:
                bias_type = "second_position_bias"
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
    
    return bias_analysis, dict(win_counts)

# %%
# Analyze results for anonymous judgments  
def analyze_bias_anonymous(judgments: list[Judgment]):
    """Analyze potential bias in anonymous (A/B) judgments."""
    
    by_question = defaultdict(list)
    for j in judgments:
        by_question[j.question.text].append(j)
    
    bias_analysis = []
    win_counts = defaultdict(int)
    
    for question_text, judgs in by_question.items():
        if len(judgs) != 2:
            continue
        
        j1, j2 = judgs[0], judgs[1]
        
        # Determine model_a and model_b
        models_in_j1 = {j1.answer_a.model, j1.answer_b.model}
        models_in_j2 = {j2.answer_a.model, j2.answer_b.model}
        
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
            continue
            
        # Parse anonymous responses (A/B)
        def parse_winner_anonymous(judgment: Judgment):
            response = judgment.text.lower().strip(' .,!?')
            
            if 'tie' in response and not ('a' in response or 'b' in response):
                return 'tie'
            elif 'a' in response and 'b' not in response:
                return 'A'
            elif 'b' in response and 'a' not in response:
                return 'B'
            else:
                return 'unclear'
        
        orig_winner_pos = parse_winner_anonymous(orig)  # A or B
        swap_winner_pos = parse_winner_anonymous(swap)  # A or B
        
        # Convert positional winners to model names
        def pos_to_model(pos_winner, judgment):
            if pos_winner == 'A':
                return judgment.answer_a.model
            elif pos_winner == 'B':
                return judgment.answer_b.model
            else:
                return pos_winner  # 'tie' or 'unclear'
        
        orig_winner = pos_to_model(orig_winner_pos, orig)
        swap_winner = pos_to_model(swap_winner_pos, swap)
        
        # Count wins
        if orig_winner in [model_a, model_b]:
            win_counts[orig_winner] += 0.5
        if swap_winner in [model_a, model_b]:
            win_counts[swap_winner] += 0.5
        
        # Check consistency (same logic as named)
        consistent = (
            (orig_winner == 'tie' and swap_winner == 'tie') or
            (orig_winner == model_a and swap_winner == model_a) or
            (orig_winner == model_b and swap_winner == model_b)
        )
        
        bias_type = None
        if not consistent and orig_winner != 'unclear' and swap_winner != 'unclear':
            if orig_winner_pos == 'A' and swap_winner_pos == 'A':
                bias_type = "first_position_bias"
            elif orig_winner_pos == 'B' and swap_winner_pos == 'B':
                bias_type = "second_position_bias"
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
            'orig_winner_pos': orig_winner_pos,
            'swap_winner_pos': swap_winner_pos,
            'orig_response': orig.text,
            'swap_response': swap.text,
            'consistent': consistent,
            'bias_type': bias_type,
            'model_a_text': orig.answer_a.text if orig.answer_a.model == model_a else orig.answer_b.text,
            'model_b_text': orig.answer_a.text if orig.answer_a.model == model_b else orig.answer_b.text,
        })
    
    return bias_analysis, dict(win_counts)

# %%
named_analysis, named_wins = analyze_bias_named(named_judgments)
anonymous_analysis, anonymous_wins = analyze_bias_anonymous(anonymous_judgments)

print("=== NAMED EVALUATION RESULTS ===")
tkp.p(f"Analyzed {len(named_analysis)} question pairs")
print(f"Win counts: {named_wins}")

print("\n=== ANONYMOUS EVALUATION RESULTS ===") 
tkp.p(f"Analyzed {len(anonymous_analysis)} question pairs")
print(f"Win counts: {anonymous_wins}")

# %%
