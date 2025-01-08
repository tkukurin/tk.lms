import json
import openai
from pathlib import Path
from tk.utils import memo


class ApiModel:
    """Wrapper around a gpt-like api.
    """
    
    def __init__(self, cached=True):
        with Path("~/.apikeys.json").expanduser().open() as f:
            key = json.load(f)['openai-self']
        self.client = openai.Client(api_key=key)
        # NOTE this neatly lets you add new examples below without extra API calls
        # don't forget it's CACHING RESPONSES tho.
        # DON'T DO MONTE CARLO EXPERIMENTATION
        self.query = self.client.chat.completions.create
        if cached: self.query = memo(self.query)

    def __call__(self, model: str, messages: list, n: int):
        return self.query(model=model, messages=messages, n=n)
