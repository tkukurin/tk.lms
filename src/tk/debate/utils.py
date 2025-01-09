import time

from tk.utils.log import L

import openai.types.chat.chat_completion as oai_types


def construct_assistant_message(completion: oai_types.ChatCompletion):
    content = completion.choices[0].message.content
    return {"role": "assistant", "content": content}


def generate_answer(answer_context) -> oai_types.ChatCompletion:
    try:
        from tk.models.gpt import ApiModel
        create = ApiModel()
        completion = create(
                  model="gpt-3.5-turbo-0125",
                  messages=answer_context,
                  n=1)
    except Exception as e:
        L.error(f"API error: {e}")
        time.sleep(20)
        return generate_answer(answer_context)

    return completion

