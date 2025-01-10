from ml_collections import ConfigDict

def get_config() -> ConfigDict:
    """ todo move to config or sth """
    c = ConfigDict()
    c.model = "gg-hf/gemma-2b-it"
    c.agents = 3
    c.rounds = 2
    return c

