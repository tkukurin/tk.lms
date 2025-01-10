from __future__ import annotations

import threading
from typing import NamedTuple
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch

import time

from tk.utils.log import L

import openai.types.chat.chat_completion as oai_types

from ml_collections import ConfigDict
from types import SimpleNamespace as nspc

def _oai_adapt(text: str):
    """Unnecessarily nested adapter for relevant parts of ChatCompletion"""
    # NB config dict has ugly repr
    # msg = ConfigDict()
    # msg.content = text
    # choice = ConfigDict()
    # choice.message = msg
    # mlc = ConfigDict()
    # mlc.choices = [choice]
    return nspc(choices=[nspc(message=nspc(content=text))])


def construct_assistant_message(completion: oai_types.ChatCompletion):
    content = completion.choices[0].message.content
    return {"role": "assistant", "content": content}


class LocalModel:
    """Registry pattern for managing local generative models.

    Use in place of gpt for local model fueled debate:
        >>> LocalModel("model-id")  # gets or creates instance
    """

    _DEFAULT_GEN_KWS = dict(
        return_dict_in_generate=True,
        return_legacy_cache=False,
        max_new_tokens=512,
    )

    REGISTRY = {}
    _REGISTRY_LOCK = threading.Lock()

    def __new__(
        cls, model_id: str = "gg-hf/gemma-2b-it",
        gpuclean: bool = True
    ):
        with cls._REGISTRY_LOCK:
            if model_id not in cls.REGISTRY:
                cls.REGISTRY[model_id] = instance = super().__new__(cls)
                instance.__init__(model_id)  # Initialize before GPU operations
            else:
                instance = cls.REGISTRY[model_id]
            if gpuclean and torch.cuda.is_available():
                for k, v in cls.REGISTRY.items():
                    if k != model_id:
                        L.debug(f"{k}->cpu")
                        v.model.cpu()
                L.debug(f"{model_id}->cuda")
                instance.model.cuda()
            return instance

    def __init__(self, model_id: str = "gg-hf/gemma-2b-it", **kw):
        del kw  # unused
        if hasattr(self, 'model_id'):  # Skip if already initialized
            return L.info(f"Reusing LocalModel: {model_id}")
        L.info(f"Creating new LocalModel: {model_id}")
        self.model_id = model_id
        dtype = torch.bfloat16
        self.tokenizer = AutoTokenizer.from_pretrained(model_id)
        self.model = AutoModelForCausalLM.from_pretrained(
            model_id,
            device_map="auto",
            torch_dtype=dtype,
        )
        self.model.eval()

    def __call__(self, answer_context):
        inp = self.tokenizer.apply_chat_template(
            answer_context, 
            tokenize=True, 
            add_generation_prompt=True,
            return_dict=True,
            return_tensors="pt")
        inp = {k: v.to(self.model.device) for k, v in inp.items()}
        out = self.model.generate(
            **inp, 
            **self._DEFAULT_GEN_KWS)
        out_str = self.tokenizer.decode(out['sequences'][0])
        return _oai_adapt(out_str)


def generate_answer_gpt(answer_context) -> oai_types.ChatCompletion:
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
        return generate_answer_gpt(answer_context)
    return completion


generate_answer = LocalModel()