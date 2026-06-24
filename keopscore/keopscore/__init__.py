import os

# Version
here = os.path.abspath(os.path.dirname(__file__))
with open(os.path.join(here, "keops_version"), encoding="utf-8") as v:
    __version__ = v.read().rstrip()

# Config
import keopscore.config

# Create build folder if it does not exist and set it as the default build folder
try:
    keopscore.config.path.set_build_folder(reset_all=False)
except Exception as e:
    from .utils.messages import KeOps_Warning

    KeOps_Warning(
        f"An error occurred while setting up keopcore: {e}. Use keopscore.config.check_health() to get details on the current configuration.",
        level=1,
    )

# expose to the user
set_build_folder = keopscore.config.path.set_build_folder
