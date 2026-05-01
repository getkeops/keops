import os
import sys

from keopscore.utils.path_utils import (
    _env_roots,
    _ordered_search_roots,
    _path_candidates,
    _python_package_roots,
    _unique_paths,
)
from keopscore.utils.system_utils import _find_library_by_names

CHECK_MARK = "✅"
CROSS_MARK = "❌"

not_found_str = f"Not Found. {CROSS_MARK}"
enabled_dict = {True: f"Enabled {CHECK_MARK}", False: f"Disabled {CROSS_MARK}"}


def ensure_directory(path, add_to_syspath=False):
    """Create a directory and optionally register it in sys.path."""
    os.makedirs(path, exist_ok=True)
    if add_to_syspath and path not in sys.path:
        sys.path.append(path)
    return path


def print_envs(env_vars):
    """Print the values of specified environment variables."""

    if not env_vars:
        return

    print("\nRelevant Environment Variables")
    print("-" * 60)
    for var in env_vars:
        value = os.environ.get(var)
        if value:
            print(f"{var} = {value}")
        else:
            print(f"{var} is not set")
