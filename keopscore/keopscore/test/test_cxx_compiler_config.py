import shutil
import types

import pytest

from keopscore.config import CxxCompiler as cxx_mod
from keopscore.config.CxxCompiler import CxxCompilerConfig


def _fake_platform(os_name):
    return types.SimpleNamespace(get_platform=lambda: os_name)


def test_detect_cxx_compiler_falls_back_to_gpp_or_clangpp(monkeypatch):
    monkeypatch.delenv("CXX", raising=False)
    monkeypatch.setattr(
        shutil,
        "which",
        lambda name: "/usr/bin/clang++" if name == "clang++" else None,
    )
    run_calls = []
    monkeypatch.setattr(
        cxx_mod,
        "KeOps_OS_Run",
        lambda cmd: run_calls.append(cmd),
    )

    cfg = CxxCompilerConfig(_fake_platform("Linux"))
    assert cfg.get_cxx_compiler() == "/usr/bin/clang++"
    assert cfg.get_use_clang_on_macos() is False
    assert run_calls == []


def test_detect_cxx_compiler_raises_keops_error_when_missing(monkeypatch):
    monkeypatch.delenv("CXX", raising=False)
    monkeypatch.setattr(shutil, "which", lambda name: None)
    with pytest.raises(ValueError, match="No C\\+\\+ compiler found"):
        CxxCompilerConfig(_fake_platform("Linux"))


def test_pykeops_common_utils_imports_without_fcntl():
    import pykeops.common.utils as utils

    assert utils.axis2cat(0) == 1
    assert utils.axis2cat(1) == 0
