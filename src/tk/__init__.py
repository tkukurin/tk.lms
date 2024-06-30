import sys
from pathlib import Path

rootdir = Path(__file__).parent.parent.parent
datadir = rootdir / 'data'

in_colab = 'google.colab' in sys.modules