"""GPT-like API and some adapter to use local models.
"""
import json
import threading
from pathlib import Path
from types import SimpleNamespace as nspc

import openai
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from tk.utils import memo
from tk.utils.log import L


def _oai_adapt(text: str | list):
  """Unnecessarily nested adapter for relevant parts of ChatCompletion"""
  # NB, config_dict has ugly default repr, this is quicker
  msg = nspc(content=text)
  if isinstance(text, list):
    msg = nspc(content=text[-1]["content"], raw=text)
  return nspc(choices=[nspc(message=msg)])


class ApiModel:
  """Wrapper around a gpt-like api."""

  def __init__(
      self,
      model_id: str | None = None,
      cached: bool = True,
      api_key: None | str | Path = None,
      **default_gpt_kwargs
  ):
    if api_key:
      with Path(api_key).expanduser().open() as f:
        api_key = json.load(f)['openai-self']
    self.client = openai.Client(api_key=api_key)
    self.model_id = model_id
    # NOTE this neatly lets you add new examples below without extra API calls
    # don't forget it's CACHING RESPONSES tho.
    # DON'T DO MONTE CARLO EXPERIMENTATION
    self.query = self.client.chat.completions.create
    if cached: self.query = memo(self.query)
    # , name=f"ApiModel_{model_id}")
    # ^ not needed, model name used in __call__
    self.default_gpt_kwargs = default_gpt_kwargs

  def __call__(self, messages: list, model: str | None = None, **kwargs):
    kws = {**self.default_gpt_kwargs, **kwargs}
    return self.query(
      messages=messages,
      model=model or self.model_id, **kws)


class LocalModel:
  """Singleton to ensure single model load/unload.

  Has same interface as api model.
  """

  _DEFAULT_GEN_KWS = dict(
      #   return_dict_in_generate=True,
      return_legacy_cache=False,
      max_new_tokens=512,
  )

  _instance = None

  def __new__(cls, model: str = "google/gemma-2b-it", **kw):
    if cls._instance is None:
      cls._instance = super().__new__(cls)
      cls._instance.__init__(model=model, **kw)
    elif cls._instance.model_id != model:
      L.info(f"Swapping LocalModel: {cls._instance.model_id} -> {model}")
      cls._instance.model.cpu()
      cls._instance = super().__new__(cls)
      cls._instance.__init__(model=model, **kw)
    return cls._instance

  def __init__(self, model: str = "google/gemma-2b-it", cached: bool = True, **kw):
    del kw  # unused
    if hasattr(self, 'model_id'):  # Skip if already initialized
      return L.info(f"Reusing LocalModel: {model}")
    L.info(f"Creating new LocalModel: {model}")
    self.model_id = model
    self.tokenizer = AutoTokenizer.from_pretrained(model)
    self.model = AutoModelForCausalLM.from_pretrained(
        model,
        device_map="auto",
        torch_dtype=torch.bfloat16,
    )
    self.model.eval()
    from transformers.pipelines import pipeline
    self.gen = pipeline(
        "text-generation",
        model=self.model,
        tokenizer=self.tokenizer,
    )
    if cached: self.gen = memo(
      self.gen.__call__, name=f"LocalModel_{model}")

  def __call__(self, messages: list, **kw):
    """just get the generated text"""
    kw = {**self._DEFAULT_GEN_KWS, **kw}
    out = self.gen(messages, **kw)
    return _oai_adapt(out[0]["generated_text"])
