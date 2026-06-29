"""Penzai-style model surgery for PyTorch

Usage:

    sel = select(clf.model_).at("icl_predictor.tf_icl.blocks.*")
    with sel.tap() as acts:
        clf.predict(X_test)
    emb = {p.rsplit(".", 1)[-1]: t.cpu().numpy() for p, t in acts.items()}

Ablate a layer (model surgery):

    with select(model).at("**.blocks.3").patch(lambda out: out * 0):
        y = model(x)

Discover paths to address:

    print(show(clf.model_, depth=3))


With treescope:

    import treescope; treescope.basic_interactive_setup()
    select(clf.model_).at("icl_predictor.tf_icl.blocks.*")  # foldable tree
"""
from __future__ import annotations

import re
from collections.abc import Callable, Iterator
from contextlib import contextmanager
from dataclasses import dataclass, replace

from torch import Tensor, nn

Hook = Callable[[str, nn.Module], bool]


def _rx(glob: str) -> re.Pattern:
    """Path glob -> regex. `*` stays within a segment, `**` spans dots."""
    body = (
        re.escape(glob)
        .replace(r"\*\*", "\0").replace(r"\*", "[^.]*").replace("\0", ".*"))
    return re.compile(f"^{body}$")


def _map(v, fn):
    return [fn(t) for t in v] if isinstance(v, list) else fn(v)


class Caught(dict):
    """Path -> captured tensor(s), insertion-ordered by first capture."""

    def cpu(self) -> Caught:
        return Caught((k, _map(v, Tensor.cpu)) for k, v in self.items())

    def np(self) -> dict:
        to_np = lambda t: t.detach().float().cpu().numpy()  # noqa
        return {k: _map(v, to_np) for k, v in self.items()}

    def __repr__(self) -> str:
        def shape(v):
            return f"[{len(v)}x]" if isinstance(v, list) else tuple(v.shape)
        body = ", ".join(f"{k}: {shape(v)}" for k, v in self.items())
        return f"Caught({{{body}}})"


@dataclass(frozen=True)
class Selection:
    """A set of submodules of `root`, addressed by dotted path.

    Navigation (`at`, `at_type`, `where`) narrows the selection functionally;
    `tap`/`patch` act on whatever remains as scoped context managers.
    """

    root: nn.Module
    paths: tuple[str, ...]

    def __iter__(self) -> Iterator[tuple[str, nn.Module]]:
        mods = dict(self.root.named_modules())
        return iter((p, mods[p]) for p in self.paths)

    def __len__(self) -> int:
        return len(self.paths)

    def __repr__(self) -> str:
        head, n = ", ".join(self.paths[:3]), len(self.paths)
        more = "" if n <= 3 else f", +{n - 3} more"
        return f"Selection({n} modules: {head}{more})"

    def __treescope_repr__(self, path, subtree_renderer):
        """Render as the selected path -> submodule mapping (not whole root)."""
        from treescope import repr_lib
        mods = dict(self.root.named_modules())
        return repr_lib.render_dictionary_wrapper(
            object_type=type(self),
            wrapped_dict={p: mods[p] for p in self.paths},
            path=path,
            subtree_renderer=subtree_renderer)

    def at(self, *globs: str) -> Selection:
        rxs = [_rx(g) for g in globs]
        keep = tuple(p for p in self.paths if any(r.match(p) for r in rxs))
        return replace(self, paths=keep)

    def at_type(self, *types: type) -> Selection:
        return self.where(lambda _p, m: isinstance(m, types))

    def where(self, pred: Hook) -> Selection:
        keep = tuple(p for p, m in self if pred(p, m))
        return replace(self, paths=keep)

    def modules(self) -> list[nn.Module]:
        return [m for _, m in self]

    @contextmanager
    def tap(self, *, clone: bool = True, keep: str = "last"):
        """Capture each selected module's output. Yields a live `Caught` dict.

        `clone` guards against later in-place edits (TabICL mutates block
        outputs); `keep="all"` collects every call into a list per path.
        """
        caught = Caught()
        with _hooks(self, lambda p, m: _grab(p, caught, clone, keep)):
            yield caught

    @contextmanager
    def patch(self, fn: Callable[[Tensor], Tensor]):
        """Replace each selected module's output with `fn(output)`."""
        with _hooks(self, lambda p, m: (lambda _m, _i, o: fn(o))):
            yield


def _grab(path, store, clone, keep):
    def hook(_m, _inp, out):
        if isinstance(out, Tensor):
            out = out.detach().clone() if clone else out.detach()
        if keep == "all":
            store.setdefault(path, []).append(out)
        else:
            store[path] = out
    return hook


@contextmanager
def _hooks(sel: Selection, make: Callable[[str, nn.Module], Callable]):
    handles = [m.register_forward_hook(make(p, m)) for p, m in sel]
    try:
        yield
    finally:
        for h in handles:
            h.remove()


def select(root: nn.Module) -> Selection:
    return Selection(root, tuple(p for p, _ in root.named_modules() if p))


def paths(root: nn.Module, glob: str = "**") -> tuple[str, ...]:
    return select(root).at(glob).paths


def show(root: nn.Module, depth: int | None = 2, types: bool = True) -> str:
    """Indented module tree with dotted paths -- a text treescope."""
    out = [type(root).__name__]
    for path, m in root.named_modules():
        if not path:
            continue
        d = path.count(".") + 1
        if depth is not None and d > depth:
            continue
        line = "  " * d + path.rsplit(".", 1)[-1]
        out.append(f"{line}: {type(m).__name__}" if types else line)
    return "\n".join(out)
