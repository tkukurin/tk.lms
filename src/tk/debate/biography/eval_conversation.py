import json
from pathlib import Path
import numpy as np
import time
from tk.debate.utils import generate_answer, construct_assistant_message
from tk.debate import utils
import re


def parse_bullets(sentence):
    # Split text into lines first
    lines = sentence.split('\n')
    bullets = []
    
    # More comprehensive bullet pattern that handles various formats
    bullet_pattern = r'^\s*(?:[-*•]|\d+\.|[A-Za-z]\)|\([0-9A-Za-z]\))\s*(.+)'
    
    for line in lines:
        match = re.match(bullet_pattern, line.strip())
        if match:
            content = match.group(1).strip()
            if content:  # Only add non-empty content
                bullets.append(content)
    
    # If no bullets found, return the whole text as one bullet
    if not bullets and sentence.strip():
        return [sentence.strip()]
    
    return bullets


def parse_yes_no(string):
    if "uncertain" in string.lower():
        return None
    elif "yes" in string.lower():
        return True
    elif "no" in string.lower():
        return False
    else:
        return None

def filter_people(person):
    people = person.split("(")[0]
    return people

def main(cfg, **kw):
    response = utils.load(cfg, "biography")
    base = Path(__file__).parent
    with open(base / "article.json", "r") as f:
        gt_data = json.load(f)

    gt_data_filter = {}

    for k, v in gt_data.items():
        k = filter_people(k)
        gt_data_filter[k] = v

    gt_data = gt_data_filter

    people = list(response.keys())

    accuracies = []

    for person in people:
        if person not in gt_data:
            continue

        gt_description = gt_data[person]
        gt_bullets = parse_bullets(gt_description)
        bio_descriptions = response[person]

        for description in bio_descriptions:
            bio_description = description[-1]['content']
            if bio_description.startswith("<bos><start_of_turn>"):
                r = utils.process_gemma_response(bio_description)
                bio_description = r[-1][-1]
            bio_bullets = parse_bullets(bio_description)
            if len(bio_bullets) == 1:
                if len(bio_bullets[0]) < 400:
                    continue

            bio_bullets = " ".join(bio_bullets)
            # continue

            for bullet in gt_bullets:
                message = [{"role": "user", "content": "Consider the following biography of {}: \n {} \n\n Is the above biography above consistent with the fact below? \n\n {} \n Give a single word answer, yes, no, or uncertain. Carefully check the precise dates and locations between the fact and the above biography.".format(person, bio_bullets, bullet)}]

                completion = generate_answer(message)
                content = completion.choices[0].message.content
                accurate = parse_yes_no(content)
                if accurate is not None:
                    accuracies.append(float(accurate))

            print("accuracies:", np.mean(accuracies), np.std(accuracies) / (len(accuracies) ** 0.5))

