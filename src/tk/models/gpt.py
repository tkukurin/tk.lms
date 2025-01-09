import json
import openai
from pathlib import Path
from tk.utils import memo


class ApiModel:
    """Wrapper around a gpt-like api.
    """

    def __init__(self, cached: bool = True, api_key: None | str | Path = None, **default_gpt_kwargs):
        if api_key: 
            with Path(api_key).expanduser().open() as f:
                api_key = json.load(f)['openai-self']
        self.client = openai.Client(api_key=api_key)
        # NOTE this neatly lets you add new examples below without extra API calls
        # don't forget it's CACHING RESPONSES tho.
        # DON'T DO MONTE CARLO EXPERIMENTATION
        self.query = self.client.chat.completions.create
        if cached: self.query = memo(self.query)
        self.default_gpt_kwargs = default_gpt_kwargs

    def __call__(self, model: str, messages: list, **kwargs):
        kws = {
            **self.default_gpt_kwargs, 
            **kwargs
        }
        return self.query(model=model, messages=messages, **kws)
