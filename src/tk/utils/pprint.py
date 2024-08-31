from rich import (
    table,
    console,
    default_styles,
    # print,
    pretty,
    progress,
    text,
    theme,
    themes
)

c = console.Console(width=80, tab_size=2, log_path=False)
print = c.print
p = c.print
log = c.log
repr = pretty.pretty_repr


def tqdm(iterable, description="Processing", total=None, **kwargs):
    total = total or len(iterable)
    with progress.Progress() as p:
        task = p.add_task(description, total=total)
        for item in iterable:
            yield item
            p.update(task, advance=1)
    

def trange(start, stop=None, step=1, description="Processing", **kwargs):
    if stop is None:  # If only one argument is provided
        start, stop = 0, start
    total = (stop - start) // step
    with progress.Progress() as p:
        task = p.add_task(description, total=total)
        for value in range(start, stop, step):
            yield value
            p.update(task, advance=1)