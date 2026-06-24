import pykeops.config as pykeopsconfig
from .nvrtc import LoadKeOps_nvrtc

if pykeopsconfig.cuda.get_use_cuda():
    from .cpp import LoadKeOps_cpp

    keops_binder = {
        "nvrtc": LoadKeOps_nvrtc.LoadKeOps_nvrtc,
        "cpp": LoadKeOps_cpp.LoadKeOps_cpp,
    }
else:
    from .cpp import LoadKeOps_cpp

    keops_binder = {"cpp": LoadKeOps_cpp.LoadKeOps_cpp}
