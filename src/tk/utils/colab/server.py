"""Some hacks to run a server on Google Colab.
"""
import re

from pathlib import Path


def show(port: int | str = 5000) -> None:
    """Run a server within Colab/Jupyter.

    https://brightersidetech.com/running-flask-apps-in-google-colab/
    """
    from google.colab.output import eval_js
    print(eval_js(f'google.colab.kernel.proxyPort({port})'))


def run_memray_server(port: int = 5000, memray_dir = './memray-results/'):
    """At some point I wanted to profile something on Colab.
    
    This seems to be _one_ way to do it.
    Just run a server and profile your memray runs:
        >>> !pip install memray_profiler memray
        >>> %%memray_flamegraph
        >>> # (do your profiling ...)
        >>> run_memray_server(5000)
        >>> show(5000)
    """

    import flask
    app = flask.Flask(__name__)
    # TODO: app._static_folder = ''

    def get_mtime_sorted(path):
        path = Path(path)
        assert path.exists()
        memdir = filter(lambda p: p.is_dir(), path.glob("*"))
        memdir = sorted(memdir, key=lambda x: x.stat().st_mtime)
        return memdir

    memray_dirs = get_mtime_sorted(memray_dir)

    # see path options on:
    # https://werkzeug.palletsprojects.com/en/3.0.x/routing/#builtin-converters
    # https://stackoverflow.com/questions/15117416/capture-arbitrary-path-in-flask-route
    @app.route('/', defaults={'x': ''})
    @app.route('/<x>')
    def home(x: str):
        global memray_dirs

        if not x:
            x = memray_dirs[-1]
            assert x.exists(), "Double check path"
        elif re.match(r'[0-9]+', x):
            x = memray_dirs[-int(x)]
        else:
            x = memray_dirs[0].parent / x

        with open(x / 'flamegraph.html') as f:
            html = f.read()

        return html

    app.run(host='0.0.0.0', port=port)
