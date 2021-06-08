import os

from .config import build_path
os.makedirs(build_path, exist_ok=True)

from keops.python_engine.get_keops_dll import get_keops_dll

use_jit = True


