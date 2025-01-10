from pathlib import Path
from tk.debate.utils import load
from tk.debate.math.plot import (
    plot_accuracy_progression,
    plot_agent_answers,
    plot_agent_agreement
)


def main(cfg, dbg=False):
    """Evaluate and visualize math debate results."""
    # Load generated data
    data = load(cfg, "math", dbg=dbg)
    if not data:
        raise ValueError("No data found to evaluate")
    
    # Extract scores and answers from the data
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
    
    # Create plots directory
    import tk
    save_dir = tk.datadir / "math_plots"
    save_dir.mkdir(exist_ok=True, parents=True)
    
    # Generate plots
    plot_accuracy_progression(
        scores, len(data), 
        save_path=save_dir / "accuracy.png")
    plot_agent_answers(
        text_answers_history, 
        save_path=save_dir / "agent_answers.png")
    plot_agent_agreement(
        text_answers_history, 
        save_path=save_dir / "agreement.png")
