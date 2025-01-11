import json
import random
from pathlib import Path

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


def filter_people(person):
  people = person.split("(")[0]
  return people


def construct_message(agents, idx, person, final=False):
  prefix_string = "Here are some bullet point biographies of {} given by other agents: ".format(
      person)

  if len(agents) == 0:
    return {
        "role":
            "user",
        "content":
            "Closely examine your biography and provide an updated bullet point biography."
    }

  for i, agent in enumerate(agents):
    agent_response = agent[idx]["content"]
    response = "\n\n Agent response: ```{}```".format(agent_response)

    prefix_string = prefix_string + response

  if final:
    prefix_string = prefix_string + "\n\n Closely examine your biography and the biography of other agents and provide an updated bullet point biography.".format(
        person, person)
  else:
    prefix_string = prefix_string + "\n\n Using these other biographies of {} as additional advice, what is your updated bullet point biography of the computer scientist {}?".format(
        person, person)

  return {"role": "user", "content": prefix_string}


def main(cfg, **kw):
  global generate_answer
  assert generate_answer

  path = Path(__file__).parent / "article.json"
  with open(path, "r") as f:
    # name -> biography
    data = json.load(f)

  people = sorted(data.keys())
  people = [filter_people(person) for person in people]
  random.seed(1)
  random.shuffle(people)

  generated_description = {}
  n = 2 if cfg.dbg else 40

  for person in tqdm(people[:n]):
    agent_contexts = [[{
        "role":
            "user",
        "content":
            f"Give a bullet point biography of {person} highlighting "
            "their contributions and achievements as a computer scientist, "
            "with each fact separated with a new line character."
    }] for agent in range(cfg.agents)]

    for round in range(cfg.rounds):
      for i, agent_context in enumerate(agent_contexts):

        if round != 0:
          agent_contexts_other = agent_contexts[:i] + agent_contexts[i + 1:]

          if round == (cfg.rounds - 1):
            message = construct_message(
                agent_contexts_other, 2 * round - 1, person=person, final=True)
          else:
            message = construct_message(
                agent_contexts_other, 2 * round - 1, person=person, final=False)
          agent_context.append(message)

        completion = generate_answer(agent_context)
        assistant_message = construct_assistant_message(completion)
        agent_context.append(assistant_message)

      bullets = parse_bullets(completion.choices[0].message.content)

      # The LM just doesn't know this person so no need to create debates
      if len(bullets) == 1:
        break

    generated_description[person] = agent_contexts

  from tk.debate import utils
  utils.save(cfg, generated_description)
