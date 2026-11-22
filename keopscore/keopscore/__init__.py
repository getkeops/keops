import os

# Version
here = os.path.abspath(os.path.dirname(__file__))
with open(os.path.join(here, "keops_version"), encoding="utf-8") as v:
    __version__ = v.read().rstrip()

# Config (build folder and CUDA driver are initialized lazily on first use; #444)
import keopscore.config

# expose to the user
set_build_folder = keopscore.config.path.set_build_folder
