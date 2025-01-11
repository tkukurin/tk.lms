import json
import random

import datasets
import numpy as np
import tqdm

from tk.debate.utils import construct_assistant_message, generate_answer
from tk.utils.log import L


def construct_message(agents, question, idx):
  if len(agents) == 0:
    return {
        "role":
            "user",
        "content":
            "Can you double check that your answer is correct. Please reiterate your answer, with your final answer a single numerical number, in the form \\boxed{{answer}}."
    }

  prefix_string = "These are the solutions to the problem from other agents: "

  for agent in agents:
    agent_response = agent[idx]["content"]
    response = "\n\n One agent solution: ```{}```".format(agent_response)

    prefix_string = prefix_string + response

  prefix_string = prefix_string + """\n\n Using the solutions from other agents as additional information, can you provide your answer to the math problem? \n The original math problem is {}. Your final answer should be a single numerical number, in the form \\boxed{{answer}}, at the end of your response.""".format(
      question
  )
  return {"role": "user", "content": prefix_string}


def read_jsonl(path: str):
  with open(path) as fh:
    return [json.loads(line) for line in fh.readlines() if line]


def main(cfg, **kw):
  dbg = cfg.dbg
  agents = cfg.agents
  rounds = cfg.rounds
  random.seed(cfg.seed)

  generated_description = {}
  L.info("Loading gsm4k")
  dftest = datasets.load_dataset("openai/gsm8k", "main")["test"].to_pandas()
  questions = dftest.to_dict(orient='records')
  random.shuffle(questions)
  n = 2 if dbg else 100

  for data in tqdm.tqdm(questions[:n], desc='questions'):
    question = data['question']
    answer = data['answer']

    agent_contexts = [
        [
            {
                "role":
                    "user",
                "content":
                    """Can you solve the following math problem? {} Explain your reasoning. Your final answer should be a single numerical number, in the form \\boxed{{answer}}, at the end of your response. """
                    .format(question)
            }
        ] for agent in range(agents)
    ]

    for round in tqdm.trange(rounds, desc='rounds'):
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

    generated_description[question] = (agent_contexts, answer)

  from tk.debate import utils
  utils.save(cfg, generated_description)
