import os

from .config import build_path, get_jit_binary
os.makedirs(build_path, exist_ok=True)

from keops.python_engine.get_keops_dll import get_keops_dll

use_jit = True

if use_jit:
    get_jit_binary()

