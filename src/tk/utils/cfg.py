"""Quick CFG parser and generator.

TODO check, LLM-generated.
At some point migrate to e.g. [lark]

[lark]: https://github.com/lark-parser/lark
"""
# %%
import numpy as np
from dataclasses import dataclass
from collections import namedtuple
from typing import List, Dict, Set, Optional

Rule = namedtuple('Rule', ['lhs', 'rhs'])


@dataclass
class Tree:
  value: str
  children: List['Tree']


@dataclass
class CFG:
  terminals: Set[str]
  non_terminals: Set[str]
  rules: list[Rule]
  start: str
  limit: int | None = None


def parse(s: str) -> CFG:
  """CFG from string

  Example:
      >>> grammar_str = '''
          S -> NP VP
          NP -> DET N
          VP -> V NP
          DET -> the
          N -> cat
          N -> dog
          V -> chased
          '''
      >>> cfg = parse(grammar_str)
  """
  lines = s.strip().split('\n')
  terminals = set()
  non_terminals = set()
  rules = []
  start = None
  for line in lines:
    if not line.strip():
      continue
    lhs, rhs = line.split('->')
    lhs = lhs.strip()
    rhs = rhs.strip()
    if start is None:
      start = lhs
    non_terminals.add(lhs)
    rules.append(Rule(lhs, rhs.split()))
    for symbol in rhs.split():
      if not symbol.isupper():
        terminals.add(symbol)
      else:
        non_terminals.add(symbol)
  return CFG(terminals, non_terminals, rules, start or "S")


def test_parse():
  grammar_str = '''
        S -> NP VP
        NP -> DET N
        VP -> V NP
        DET -> the
        N -> cat
        N -> dog
        V -> chased
        '''
  cfg = parse(grammar_str)
  print(cfg)
  assert cfg.start == 'S'
  assert cfg.terminals == {'the', 'cat', 'dog', 'chased'}
  assert cfg.non_terminals == {'S', 'NP', 'VP', 'DET', 'N', 'V'}
  assert len(cfg.rules) == 7
  assert Rule('S', ['NP', 'VP']) in cfg.rules
  assert Rule('NP', ['DET', 'N']) in cfg.rules
  assert Rule('VP', ['V', 'NP']) in cfg.rules
  assert Rule('DET', ['the']) in cfg.rules
  assert Rule('N', ['cat']) in cfg.rules
  assert Rule('N', ['dog']) in cfg.rules
  assert Rule('V', ['chased']) in cfg.rules
  print("PASS: test_parse")


def interpret(tokens: list[str], cfg: CFG) -> Optional[Tree]:
  def build_tree(symbol: str, pos: int) -> tuple[Optional[Tree], int]:
    if symbol in cfg.terminals:
      if pos < len(tokens) and tokens[pos] == symbol:
        return Tree(symbol, []), pos + 1
      return None, pos

    matching_rules = [r for r in cfg.rules if r.lhs == symbol]

    for rule in matching_rules:
      current_pos = pos
      children = []
      valid = True

      for sym in rule.rhs:
        subtree, new_pos = build_tree(sym, current_pos)
        if subtree is None:
          valid = False
          break
        children.append(subtree)
        current_pos = new_pos

      if valid:
        return Tree(symbol, children), current_pos

    return None, pos

  tree, final_pos = build_tree(cfg.start, 0)
  if tree is not None and final_pos == len(tokens):
    return tree
  return None


def generate(
  cfg: CFG, 
  gen: np.random.Generator = np.random.default_rng()
) -> list[str]:
  result = []
  stack = [(cfg.start, False)]  # (symbol, is_expanded)
  
  while stack:
    symbol, is_expanded = stack.pop()
    
    if is_expanded:
      result.append(symbol)
      continue
      
    if symbol in cfg.terminals:
      result.append(symbol)
      continue
      
    matching_rules = [r for r in cfg.rules if r.lhs == symbol]
    while True:
      chosen_rule = matching_rules[gen.integers(0, len(matching_rules))]
      temp_stack = [(s, False) for s in reversed(chosen_rule.rhs)]
      
      # Check length limit
      temp_result = len(result)
      for s, _ in temp_stack:
        if s in cfg.terminals:
          temp_result += 1
        
      if not cfg.limit or temp_result <= cfg.limit:
        stack.extend(temp_stack)
        break
        
  return result


def test_generate():
  grammar_str = """
    S -> NP VP
    NP -> DET N
    VP -> V NP
    DET -> the
    N -> cat
    N -> dog
    V -> chased
    """

  cfg = parse(grammar_str)
  for _ in range(5):
    sentence = generate(cfg)
    tree = interpret(sentence, cfg)
    assert tree is not None
    print(f"Generated: {sentence}")


def test_interpret():
  grammar_str = """
    S -> NP VP
    NP -> DET N
    VP -> V NP
    DET -> the
    N -> cat
    N -> dog
    V -> chased
    """

  cfg = parse(grammar_str)

  test_cases = [
      "the cat chased the dog",
      "the dog chased the cat",
      "cat the chased dog the"  # should fail
  ]

  for test in test_cases:
    toks = test.split()
    tree = interpret(toks, cfg)
    print(f"Input: {test}")
    print(f"Valid: {tree is not None}")


if __name__ == "__main__":
  test_parse()
  test_generate()
  test_interpret()
# %%
