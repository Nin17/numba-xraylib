"""_summary_
"""

import os
import sys

if sys.version_info < (3, 11):
    import tomli as tomllib
else:
    import tomllib

PATH = os.path.dirname(__file__)
CONFIG_PATH = os.path.join(PATH, "config.toml")

with open(CONFIG_PATH, "rb") as f:
    config = tomllib.load(f)
