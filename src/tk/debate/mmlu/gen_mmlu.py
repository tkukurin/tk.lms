import json
import random
import time
from glob import glob

import datasets
import pandas as pd
from tqdm import trange

from tk.debate.utils import construct_assistant_message, generate_answer
from tk.utils.log import L


def construct_message(agents, question, idx):
  if len(agents) == 0:
    return {
        "role":
            "user",
        "content":
            "Can you double check that your answer is correct. Put your final answer in the form (X) at the end of your response."
    }

  prefix_string = "These are the solutions to the problem from other agents: "

  for agent in agents:
    agent_response = agent[idx]["content"]
    response = "\n\n One agent solution: ```{}```".format(agent_response)

    prefix_string = prefix_string + response

  prefix_string = prefix_string + """\n\n Using the reasoning from other agents as additional advice, can you give an updated answer? Examine your solution and that other agents step by step. Put your answer in the form (X) at the end of your response.""".format(
      question
  )
  return {"role": "user", "content": prefix_string}


def parse_question_answer(df, ix):
  question = df.iloc[ix, 0]
  a, b, c, d = df.iloc[ix]['choices']

  question = "Can you answer the following question as accurately as possible? {}: A) {}, B) {}, C) {}, D) {} Explain your answer, putting the answer in the form (X) at the end of your response.".format(
      question, a, b, c, d
  )

  answer = df.iloc[ix]['answer']
  return question, str(answer)


def main(cfg, **kw):
  agents = cfg.agents
  rounds = cfg.rounds

  # load from huggingface
  mmlu_configs = [
      'abstract_algebra',
      'all',
      'anatomy',
      'astronomy',
      # does not have test
      # > The auxiliary_training data could be used for fine-tuning, something
      # important for models without few-shot capabilities
      # 'auxiliary_train',
      'business_ethics',
      'clinical_knowledge',
      'college_biology',
      'college_chemistry',
      'college_computer_science',
      'college_mathematics',
      'college_medicine',
      'college_physics',
      'computer_security',
      'conceptual_physics',
      'econometrics',
      'electrical_engineering',
      'elementary_mathematics',
      'formal_logic',
      'global_facts',
      'high_school_biology',
      'high_school_chemistry',
      'high_school_computer_science',
      'high_school_european_history',
      'high_school_geography',
      'high_school_government_and_politics',
      'high_school_macroeconomics',
      'high_school_mathematics',
      'high_school_microeconomics',
      'high_school_physics',
      'high_school_psychology',
      'high_school_statistics',
      'high_school_us_history',
      'high_school_world_history',
      'human_aging',
      'human_sexuality',
      'international_law',
      'jurisprudence',
      'logical_fallacies',
      'machine_learning',
      'management',
      'marketing',
      'medical_genetics',
      'miscellaneous',
      'moral_disputes',
      'moral_scenarios',
      'nutrition',
      'philosophy',
      'prehistory',
      'professional_accounting',
      'professional_law',
      'professional_medicine',
      'professional_psychology',
      'public_relations',
      'security_studies',
      'sociology',
      'us_foreign_policy',
      'virology',
      'world_religions'
  ]

  def ld(c) -> pd.DataFrame:
    ds = datasets.load_dataset("cais/mmlu", c)
    if "test" not in ds:
      L.error(f"{c}: no test")
      ds = ds["train"]
    else:
      ds = ds["test"]
    return ds.to_pandas()

  dfs = [ld(config) for config in mmlu_configs]
  random.seed(cfg.seed)
  response_dict = {}

  for i in trange(2 if cfg.dbg else 100):
    df = random.choice(dfs)
    ix = len(df)
    idx = random.randint(0, ix - 1)

    question, answer = parse_question_answer(df, idx)

    agent_contexts = [
        [{
            "role": "user",
            "content": question
        }] for agent in range(agents)
    ]

    for round in range(rounds):
      for i, agent_context in enumerate(agent_contexts):
        if round != 0:
          agent_contexts_other = agent_contexts[:i] + agent_contexts[i + 1:]
          message = construct_message(
              agent_contexts_other, question, 2 * round - 1
          )
          agent_context.append(message)

        completion = generate_answer(agent_context)
        assistant_message = construct_assistant_message(completion)
        agent_context.append(assistant_message)
        print(completion)

    response_dict[question] = (agent_contexts, answer)

  from tk.debate.utils import save
  save(cfg, response_dict)
