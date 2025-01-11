from __future__ import annotations

import torch
# from transformers.generation import (
#     LogitsProcessorList, TopPLogitsWarper, TemperatureLogitsWarper
# )

from collections import namedtuple
import json
from pathlib import Path
import threading
from typing import NamedTuple
from transformers import AutoTokenizer, AutoModelForCausalLM

import time

from tk.utils.log import L

import openai.types.chat.chat_completion as oai_types

from ml_collections import ConfigDict
from types import SimpleNamespace as nspc

import re

Turn = namedtuple("Turn", ["role", "content"])

def process_gemma_response(s: str):
    # extract via regex: <start_of_turn>NAME\nCONTENT<end_of_turn>
    pattern = r'<start_of_turn>(.*?)\n(.*?)(?:<end_of_turn>|<eos>)'
    matches = re.finditer(pattern, s, re.DOTALL)
    results = []
    for match in matches:
        name = match.group(1).strip()
        content = match.group(2).strip()
        results.append(Turn(name, content))
    return results if results else (None, None)

import tk
_BASE = tk.datadir / "debate"

import numpy as np
import matplotlib.pyplot as plt
def combine_figures(figs, save_path=None, grid_size=None):
    """Combine multiple figures into a single output.
    
    For TIFF files: saves as multi-page TIFF
    For other formats: combines into grid layout
    
    Args:
        figs: List of matplotlib figures to combine
        save_path: Optional path to save combined figure
        grid_size: Optional tuple of (rows, cols). If None, automatically determined
    """
    if save_path and str(save_path).lower().endswith('.tiff'):
        # For TIFF files, save as multi-page
        from PIL import Image
        images = []
        for fig in figs:
            # Convert each figure to PIL Image
            fig.canvas.draw()
            w, h = fig.canvas.get_width_height()
            buf = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)
            buf.shape = (h, w, 3)
            images.append(Image.fromarray(buf))
        
        images[0].save(save_path, 
                      save_all=True, 
                      append_images=images[1:], 
                      resolution=300)
        return images
    else:
        # For other formats, keep existing grid layout behavior
        n_figs = len(figs)
        if grid_size is None:
            n_cols = int(np.ceil(np.sqrt(n_figs)))
            n_rows = int(np.ceil(n_figs / n_cols))
        else:
            n_rows, n_cols = grid_size
            
        fig_combined = plt.figure(figsize=(n_cols * 10, n_rows * 6))
        
        for i, fig in enumerate(figs):
            if i >= n_rows * n_cols:
                break
                
            ax = fig_combined.add_subplot(n_rows, n_cols, i + 1)
            
            for ax_orig in fig.axes:
                ax.plot(*[l.get_data() for l in ax_orig.lines][0])
                ax.scatter(*[s.get_offsets().T for s in ax_orig.collections][0]) if ax_orig.collections else None
                
                ax.set_xlabel(ax_orig.get_xlabel())
                ax.set_ylabel(ax_orig.get_ylabel())
                ax.set_title(ax_orig.get_title())
                ax.grid(ax_orig.get_grid())
                
                if ax_orig.get_ylim() != (0, 1):
                    ax.set_ylim(ax_orig.get_ylim())
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path)
            plt.close()
            
        return fig_combined


def save(cfg: ConfigDict, data, base=_BASE):
    L.info(f"saving {cfg.task} data to {base}")
    d = "dbg" if cfg.dbg else "prod"
    a = cfg.agents
    r = cfg.rounds
    if isinstance(data, dict):
        n = f"{cfg.task}_{a}_{r}_{d}.json"
        if (path := base / n).exists():
            L.error(f"File {path} exists, not overwriting")
            return None
        with open(path, "w") as f:
            data["_config"] = cfg.to_json()
            response = json.dump(data, f)
    else:
        if not isinstance(data, list): data = [data]
        n = f"{cfg.task}_{a}_{r}_{d}.tiff"
        if (path := base / n).exists():
            L.error(f"File {path} exists, not overwriting")
            return None
        response = combine_figures(data, path)
    return response

def load(cfg, base=_BASE):
    d = "dbg" if cfg.dbg else "prod"
    a = cfg.agents
    r = cfg.rounds
    if not (path := base / f"{cfg.task}_{a}_{r}_{d}.json").exists():
        return None
    with open(path, "r") as f:
        response = json.load(f)
    return response


def _oai_adapt(text: str | list):
    """Unnecessarily nested adapter for relevant parts of ChatCompletion"""
    # NB, config_dict has ugly default repr, this is quicker
    msg = nspc(content=text)
    if isinstance(text, list):
        msg = nspc(content=text[-1]["content"], raw=text)
    return nspc(choices=[nspc(message=msg)])


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
        from transformers.pipelines import pipeline
        self.gen = pipeline(
            "text-generation",
            model=self.model,
            tokenizer=self.tokenizer,
        )

    def __call__(self, answer_context, **kw):
        """just get the generated text"""
        out = self.gen(
            answer_context, 
            max_new_tokens=512,
            **kw)
        return _oai_adapt(out[0]["generated_text"])

    # def _pipeline_custom(self):
    #     """returns more data than the default version
    #     """
    #     inp = self.tokenizer.apply_chat_template(
    #         answer_context, 
    #         tokenize=True, 
    #         add_generation_prompt=True,
    #         return_dict=True,
    #         return_tensors="pt")
    #     inp = {k: v.to(self.model.device) for k, v in inp.items()}
    #     out = self.model.generate(
    #         **inp, 
    #         **self._DEFAULT_GEN_KWS,
    #         # generation_config= ...
    #     )
    #     out_str = self.tokenizer.decode(out['sequences'][0])
    #     return _oai_adapt(out_str)


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


generate_answer = None

