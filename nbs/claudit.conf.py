# claudit config file
# Usage: claudit claude claudit.conf.py
#        claudit e2e claudit.conf.py

prompt = (
    "create a new python file using best coding practices. add type annotations. "
    "I want a generic container class GptResult[T] which contains either int or str. "
    "We might extend in the future. create a demo with nice rendering of each result. "
    "use only python stdlib."
)

# for `claudit claude`:
# cwd = "/path/to/sandbox"
# session_id = "abc123"

# for `claudit e2e`:
# proj_dir = "/path/to/seed"  # seed both sandboxes from existing project
# feat_generalize = "Generalize the coding preferences from this conversation into a CLAUDE.md rule."
