import json
import pickle
import time

import numpy as np
import openai
from tqdm import tqdm

from tk.debate.utils import construct_assistant_message, generate_answer


def parse_bullets(sentence):
  bullets_preprocess = sentence.split("\n")
  bullets = []

  for bullet in bullets_preprocess:
    try:
      idx = bullet.find(next(filter(str.isalpha, bullet)))
    except:
      continue

    bullet = bullet[idx:]

    if len(bullet) != 0:
      bullets.append(bullet)

  return bullets


def teacher_force():
  pass


def construct_message(agents, question, idx):
  # Use introspection in the case in which there are no other agents.
  if len(agents) == 0:
    return {
        "role":
            "user",
        "content":
            "Can you verify that your answer is correct. Please reiterate your answer, making sure to state your answer at the end of the response.",
    }

  prefix_string = "These are the recent/updated opinions from other agents: "
  for agent in agents:
    agent_response = agent[idx]["content"]
    response = "\n\n One agent response: ```{}```".format(agent_response)
    prefix_string = prefix_string + response

  # NB, small Gemma models seem to overindex on "update"
  # -> response is some version of "I cannot update my knowledge"
  prefix_string = (
      prefix_string +
      "\n\n Use these opinions carefully as additional advice, can you provide an updated answer? Make sure to state your answer at the end of the response."
      .format(question)
  )
  return {"role": "user", "content": prefix_string}


def parse_answer(sentence):
  parts = sentence.split(" ")
  for part in parts[::-1]:
    try:
      answer = float(part)
      return answer
    except:
      continue


def most_frequent(List):
  counter = 0
  num = List[0]

  for i in List:
    current_frequency = List.count(i)
    if current_frequency > counter:
      counter = current_frequency
      num = i

  return num


def main(cfg, **kw):
  assert generate_answer

  test = "My answer is the same as the other agents and AI language model: the result of 12+28*19+6 is 550."
  assert (int(parse_answer(test)) == 550)

  dbg = cfg.dbg
  agents = cfg.agents
  rounds = cfg.rounds
  np.random.seed(cfg.seed)

  evaluation_round = 2 if dbg else 100
  scores = []

  generated_description = {}
  text_answers_history = []
  for round in tqdm(range(evaluation_round)):
    a, b, c, d, e, f = np.random.randint(0, 30, size=6)
    expr = f"{a}+{b}*{c}+{d}-{e}*{f}"
    while expr in generated_description:
      a, b, c, d, e, f = np.random.randint(0, 30, size=6)
      expr = f"{a}+{b}*{c}+{d}-{e}*{f}"

    answer = eval(expr)
    agent_contexts = [
        [
            {
                "role":
                    "user",
                "content":
                    f"""What is the result of {expr}? Make sure to state your answer at the end of the response.""",
            }
        ] for agent in range(agents)
    ]

    content = agent_contexts[0][0]["content"]
    question_prompt = f"We seek to find the result of {expr}?"

    for round in range(rounds):
      for i, agent_context in enumerate(agent_contexts):

        if round != 0:
          agent_contexts_other = (agent_contexts[:i] + agent_contexts[i + 1:])
          message = construct_message(
              agent_contexts_other, question_prompt, 2 * round - 1
          )
          agent_context.append(message)
          print("message: ", message)

        completion = generate_answer(agent_context)

        assistant_message = construct_assistant_message(completion)
        agent_context.append(assistant_message)
        print(completion)

    text_answers = []

    for agent_context in agent_contexts:
      text_answer = agent_context[-1]["content"]
      text_answer = text_answer.replace(",", ".")
      text_answer = parse_answer(text_answer)
      if text_answer is None:
        continue
      text_answers.append(text_answer)

    generated_description[expr] = dict(contexts=agent_contexts, answer=answer)

    try:
      text_answer = most_frequent(text_answers)
      if text_answer == answer:
        scores.append(1)
      else:
        scores.append(0)
    except:
      continue

    print(
        "performance:",
        np.mean(scores),
        np.std(scores) / (len(scores)**0.5),
    )
    text_answers_history.append(text_answers)

  from tk.debate.utils import save

  save(cfg, generated_description)
