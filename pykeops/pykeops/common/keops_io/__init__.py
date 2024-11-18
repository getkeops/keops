from keopscore.config import config, cuda_config

if cuda_config._use_cuda:
    from . import LoadKeOps_nvrtc, LoadKeOps_cpp

    keops_binder = {
        "nvrtc": LoadKeOps_nvrtc.LoadKeOps_nvrtc,
        "cpp": LoadKeOps_cpp.LoadKeOps_cpp,
    }
else:
    from . import LoadKeOps_cpp

    keops_binder = {"cpp": LoadKeOps_cpp.LoadKeOps_cpp}
