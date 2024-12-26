"""Create dataset (I, O, program lines) from the DSL in `tk.arc.p3`.

Executing this file results w/something like this in a pickle file in `datadir`:
  
| (id, ex_id, step)  | function   | arguments   | variable   | input                             | output           |
|:-------------------|:-----------|:------------|:-----------|:----------------------------------|:-----------------|
| ('007bbfb7', 0, 0) | hupscale   | ['I', 3]    | x1         | ((0, 7, 7), (7, 7, 7), (0, 7, 7)) | ((0, 0, 0, 0, 7, 7, 0, 7, 7),... |
| ('007bbfb7', 0, 1) | vupscale   | ['x1', 3]   | x2         | ((0, 7, 7), (7, 7, 7), (0, 7, 7)) | ((0, 0, 0, 0, 7, 7, 0, 7, 7),... |
"""
# %%
import pandas as pd
from tk.arc.p3.dsl import *
from tk.arc.p3 import dsl, solver
import inspect
name_call_tuples = inspect.getmembers(dsl, inspect.isfunction)
name_call_dict = {name: call for name, call in name_call_tuples}
other_solvers = inspect.getmembers(solver, inspect.isfunction)
other_solvers_dict = {
    name: call for name, call in other_solvers if name.startswith('solve')
}
# %%
import ast

def parse_call(call):
    src = inspect.getsource(call)
    return src, ast.parse(src)

print()
print(
    parse_call(next(iter(name_call_dict.values())))
)
print()
print(
    parse_call(next(iter(other_solvers_dict.values())))
)
# %%

def get_functions(node, indent=0):
    # print all function names
    if isinstance(node, ast.Call):
        # print('F' + ' ' * (indent - 1) + node.func.id)
        yield node
    # print(' ' * indent + node.__class__.__name__)
    for child in ast.iter_child_nodes(node):
        yield from get_functions(child, indent + 2)

# %%
fnodes = get_functions(parse_call(next(iter(other_solvers_dict.values())))[1])
fnodes = list(fnodes)
# %%
def node_repr(node: ast.AST):
    if hasattr(node, 'id'):
        return node.id
    if hasattr(node, 'value'):
        return node.value
    if isinstance(node, ast.UnaryOp):
        op = 1
        if isinstance(node, ast.USub):
            op = -1
        return op * node.operand.value
    raise Exception(f'Unknown node type: {node}')

stats = {}
# First create time series of function calls for each solver
for k, v in other_solvers_dict.items():
    tree = parse_call(v)[1]
    assignments = {}
    
    # Find all assignments in the AST
    for node in ast.walk(tree):
        if isinstance(node, ast.Assign):
            if isinstance(node.value, ast.Call):
                # Get the target variable name
                if isinstance(node.targets[0], ast.Name):
                    assignments[id(node.value)] = node.targets[0].id
    
    fnodes = get_functions(tree)
    fnodes = list(fnodes)
    
    func_info = [
        (node.func.id, 
        [node_repr(arg) for arg in node.args], 
        assignments.get(id(node), None)) 
        for node in fnodes
    ]
    
    func_series = pd.Series([info[0] for info in func_info])
    arg_series = pd.Series([info[1] for info in func_info])
    var_series = pd.Series([info[2] for info in func_info])
    
    # Calculate n-grams using rolling window
    ngrams = {
        n: [tuple(window) for window in func_series.rolling(window=n)]
        for n in range(1, 4)
        if n <= len(func_series)
    }
    
    stats[k] = {
        'ngrams': ngrams,
        'len': len(fnodes),
        'series': func_series,
        'args': arg_series,
        'vars': var_series
    }

# %%
df_fns = pd.DataFrame([
    (k, i, fn, args, var) 
    for k, x in stats.items() 
    for i, (fn, args, var) in enumerate(zip(x['series'], x['args'], x['vars']))
], columns=['id', 'step', 'function', 'arguments', 'variable'])

df_fns['id'] = df_fns['id'].str.removeprefix('solve_')

# %%
from tk.arc.p3 import get_data
data = get_data('training')
print(len(data['train']))
print(len(data['test']))
# %%
df_data = pd.DataFrame([
    (k, i, 'input', e['input']) 
    for k, v in data['train'].items() 
    for i, e in enumerate(v)
] + [
    (k, i, 'output', e['output']) 
    for k, v in data['train'].items() 
    for i, e in enumerate(v)
], columns=['id', 'example_id', 'column', 'value'])

# Pivot the dataframe to have input and output as columns
df_data = df_data.pivot(
    index=['id', 'example_id'], 
    columns='column', 
    values='value').reset_index()

df = pd.merge(df_fns, df_data, on='id', how='inner')
df = df.set_index(['id', 'example_id', 'step']).sort_index()

# parquet doesn't work because arguments are lists
df.to_pickle(datadir / 'michaelhodel_rearc_data.pkl')

# %%
print(df[:2].to_markdown())
# %%
