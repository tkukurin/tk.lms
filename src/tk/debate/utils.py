from __future__ import annotations

import json
import re
import time
from collections import namedtuple
from pathlib import Path
from typing import NamedTuple

import matplotlib.pyplot as plt
import numpy as np
import openai.types.chat.chat_completion as oai_types
from ml_collections import ConfigDict

import tk
from tk.utils.log import L

Turn = namedtuple("Turn", ["role", "content"])
_BASE = tk.datadir / "debate"


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

    images[0].save(
        save_path, save_all=True, append_images=images[1:], resolution=300
    )
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
        ax.scatter(
            *[s.get_offsets().T for s in ax_orig.collections][0]
        ) if ax_orig.collections else None

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


def _mkpath(cfg, base=_BASE, suffix=None):
  """if suffix is none, return just the directory"""
  suffix = (suffix or "").lstrip(".")
  d = "dbg" if cfg.dbg else "prod"
  model = cfg.model.replace("/", "-")
  p = base / f"{cfg.task}/{model}" 
  p.mkdir(exist_ok=True, parents=True)
  if suffix is not None:
    p = p / f"{cfg.agents}_{cfg.rounds}_{d}.{suffix}"
  return p


def save(cfg: ConfigDict, data, base=_BASE):
  L.info(f"saving {cfg.task} data to {base}")
  if isinstance(data, dict):
    p = _mkpath(cfg=cfg, base=base, suffix="json")
    if p.exists():
      L.error(f"File {p} exists, not overwriting")
      return None
    with open(p, "w") as f:
      data["_config"] = cfg.to_dict()
      response = json.dump(data, f)
  else:
    if not isinstance(data, list):
      data = [data]
    p = _mkpath(cfg=cfg, base=base, suffix="tiff")
    if p.exists():
      L.error(f"File {p} exists, not overwriting")
      return None
    response = combine_figures(data, p)
  return response


def load(cfg, base=_BASE):
  if not (path := _mkpath(base=base, cfg=cfg, suffix="json")).exists():
    return None
  with open(path, "r") as f:
    response = json.load(f)
  return response


def construct_assistant_message(completion: oai_types.ChatCompletion):
  content = completion.choices[0].message.content
  return {"role": "assistant", "content": content}


# imported and set in __main__
from tk.models.api_adapter import ApiModel, LocalModel

generate_answer = None
