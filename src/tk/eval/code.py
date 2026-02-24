"""basic quick utils to eval code files.

TODO: need to be improved for sure
"""
import ast


def text2funcmap(text: str) -> dict[str, str]:
    """get funcs from `source` (text). Returns {fname: source}."""
    try: tree = ast.parse(text)
    except SyntaxError: return {}
    lines = text.splitlines(keepends=True)
    funcs = {}
    for node in ast.walk(tree):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            start = node.lineno - 1
            end = node.end_lineno
            funcs[node.name] = "".join(lines[start:end])
    return funcs


def funcdiff(funcs1: dict[str, str], funcs2: dict[str, str]) -> dict:
    """Compare two sets of functions, return diff report."""
    all_names = funcs1.keys() | funcs2.keys()
    result = {}
    for name in all_names:
        f1, f2 = funcs1.get(name), funcs2.get(name)
        if f1 is None:
            result[name] = {"status": "added", "yerule": f2}
        elif f2 is None:
            result[name] = {"status": "removed", "norule": f1}
        elif f1 != f2:
            result[name] = {"status": "modified", "norule": f1, "yerule": f2}
    return result
