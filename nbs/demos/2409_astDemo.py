"""Learn AST parsing in Python a bit.
"""
# %%
import inspect as inspect_og
from rich import inspect as I
# from rich.console import _is_jupyter as method_under_test
def method_under_test() -> bool:  # pragma: no cover
    """Check if we're running in a Jupyter notebook."""
    try:
        get_ipython  # type: ignore
    except NameError:
        return False
    ipython = get_ipython()  # type: ignore
    shell = ipython.__class__.__name__
    if (
        "google.colab" in str(ipython.__class__)
        or os.getenv("DATABRICKS_RUNTIME_VERSION")
        or shell == "ZMQInteractiveShell"
    ):
        return True  # Jupyter notebook or qtconsole
    elif shell == "TerminalInteractiveShell":
        return False  # Terminal running IPython
    else:
        return False  # Other type (?)
# %%
import ast as astlib
ast = astlib.parse(
    src := inspect_og.getsource(method_under_test))
# %%
print(src)
# %%
I(ast.body[0])
# I(ast.body[0].args)
# %%
I(ast.body[0].body[0].value)
#  │ <ast.Constant object at 0x324dc7970>
#  ╰────────────────────────────────────-
#      col_offset = 4
#  end_col_offset = 55
#      end_lineno = 2
#            kind = None
#          lineno = 2
#               n = "Check if we're running in a Jupyter notebook."
#               s = "Check if we're running in a Jupyter notebook."
#           value = "Check if we're running in a Jupyter notebook."
# %%
I(ast.body[0].body[1])
# │ <ast.Try object at 0x324dc78b0>
# ╰────────────────────────────────
#           body = [<ast.Expr object at 0x324dc7100>]
#     col_offset = 4
# end_col_offset = 20
#     end_lineno = 6
#      finalbody = []
#       handlers = [<ast.ExceptHandler object at 0x324dc66b0>]
#         lineno = 3
#         orelse = []
# %%
I(ast.body[0].body[1].body[0])
# │ <ast.Expr object at 0x324dc7100>              │ │
# ╰───────────────────────────────────────────────╯ │
#                                                   │
#     col_offset = 8                                │
# end_col_offset = 19                               │
#     end_lineno = 4                                │
#         lineno = 4                                │
#          value = <ast.Name object at 0x324dc65c0> │
# %%
I(ast.body[0].body[1].body[0].value)
# │ <ast.Name object at 0x324dc65c0>              │ │
# ╰───────────────────────────────────────────────╯ │
#                                                   │
#     col_offset = 8                                │
#            ctx = <ast.Load object at 0x101363a00> │
# end_col_offset = 19                               │
#     end_lineno = 4                                │
#             id = 'get_ipython'                    │
#         lineno = 4                                │
# %%
I(ast.body[0].body[2].value)
# │ <ast.Call object at 0x324dc6440>              │ │
# ╰───────────────────────────────────────────────╯ │
#                                                   │
#           args = []                               │
#     col_offset = 14                               │
# end_col_offset = 27                               │
#     end_lineno = 7                                │
#           func = <ast.Name object at 0x324dc5f90> │
#       keywords = []                               │
#         lineno = 7                                │
# %%
I(ast.body[0].body[2].value.func)
# │ <ast.Name object at 0x324dc5f90>              │ │
# ╰───────────────────────────────────────────────╯ │
#                                                   │
#     col_offset = 14                               │
#            ctx = <ast.Load object at 0x101363a00> │
# end_col_offset = 25                               │
#     end_lineno = 7                                │
#             id = 'get_ipython'                    │
#         lineno = 7                                │
# %%
I(ast.body[0].body[3])
# │ <ast.Assign object at 0x324dc6f80>
# ╰───────────────────────────────────
#     col_offset = 4
# end_col_offset = 38
#     end_lineno = 8
#         lineno = 8
#        targets = [<ast.Name object at 0x324dc5ff0>]
#   type_comment = None
#          value = <ast.Attribute object at 0x324dc62f0>
# %%
I(ast.body[0].body[3].value)
# │ <ast.Attribute object at 0x324dc62f0>
# ╰──────────────────────────────────────
#           attr = '__name__'
#     col_offset = 12
#            ctx = <ast.Load object at 0x101363a00>
# end_col_offset = 38
#     end_lineno = 8
#         lineno = 8
#          value = <ast.Attribute object at 0x324dc6c50>
# %%
I(ast.body[0].body[3].targets[0])
# │ <ast.Name object at 0x324dc5ff0>               │ │
# ╰────────────────────────────────────────────────╯ │
#                                                    │
#     col_offset = 4                                 │
#            ctx = <ast.Store object at 0x101363a60> │
# end_col_offset = 9                                 │
#     end_lineno = 8                                 │
#             id = 'shell'                           │
#         lineno = 8                                 │
# %%
I(ast.body[0].body[4])
# │ <ast.If object at 0x324dc4b50>                    │ │
# ╰───────────────────────────────────────────────────╯ │
#                                                       │
#           body = [<ast.Return object at 0x325597e80>] │
#     col_offset = 4                                    │
# end_col_offset = 20                                   │
#     end_lineno = 18                                   │
#         lineno = 9                                    │
#         orelse = [<ast.If object at 0x3255976a0>]     │
#           test = <ast.BoolOp object at 0x324dc5690>   │
# %%
