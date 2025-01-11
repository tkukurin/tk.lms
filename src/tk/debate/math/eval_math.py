from pathlib import Path
from tk.debate.utils import load
from tk.debate.math.plot import (
    plot_accuracy_progression,
    plot_agent_answers,
    plot_agent_agreement
)


def main(cfg, **kw):
    """Evaluate and visualize math debate results."""
    data = load(cfg)
    if not data:
        raise ValueError("No data found to evaluate")
    
    scores = []
    text_answers_history = []
    for expr, result in data.items():
        if result['answer'] == result['contexts'][0][-1]['content']:
            scores.append(1)
        else:
            scores.append(0)
        
        # Extract answers from contexts
        answers = []
        for context in result['contexts']:
            txt = context[-1]['content']
            maybe_ans = txt.split()[-1]
            try:
                answers.append(float(maybe_ans))
            except ValueError:
                continue
        text_answers_history.append(answers)
    
    figs = [
    plot_accuracy_progression(scores, len(data)),
        
    plot_agent_answers(text_answers_history),
    plot_agent_agreement(text_answers_history),
    ]
    from tk.debate import utils
    utils.save(cfg, figs)
