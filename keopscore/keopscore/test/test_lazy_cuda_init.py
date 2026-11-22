import os
import subprocess
import sys

from keopscore.config.Cuda import CudaConfig
from keopscore.config.Platform import PlatformConfig


def test_import_pykeops_does_not_initialize_cuda():
    # Regression test for https://github.com/getkeops/keops/issues/444
    code = (
        "import importlib.util, os, sys\n"
        "if sys.platform == 'win32':\n"
        "    sys.modules.setdefault('fcntl', object())\n"
        "import keopscore.config\n"
        "import pykeops\n"
        "if importlib.util.find_spec('torch') is not None:\n"
        "    from pykeops.torch import LazyTensor\n"
        "assert keopscore.config.cuda._initialized is False\n"
        "assert keopscore.config.path._build_folder == ''\n"
        "import pykeops.config as pykeopsconfig\n"
        "avail = pykeopsconfig.gpu_available\n"
        "assert isinstance(avail, bool)\n"
        "assert keopscore.config.cuda._initialized is True\n"
        "build_folder = keopscore.config.path.get_build_folder()\n"
        "assert os.path.isdir(build_folder)\n"
    )
    res = subprocess.run(
        [sys.executable, "-c", code],
        env=os.environ.copy(),
        capture_output=True,
        text=True,
        check=False,
    )
    assert res.returncode == 0, res.stderr


def test_cuda_config_lazy_initialization(monkeypatch):
    calls = []

    def fake_available(self):
        calls.append(True)
        return False

    monkeypatch.setattr(CudaConfig, "_cuda_libraries_available", fake_available)
    cfg = CudaConfig(PlatformConfig())
    assert calls == []
    assert cfg._initialized is False
    assert cfg.is_available() is False
    assert calls == [True]
    assert cfg.is_available() is False
    assert calls == [True]
