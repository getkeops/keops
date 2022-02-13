import pykeops.common.keops_io.LoadKeOps_nvrtc
import pykeops.common.keops_io.LoadKeOps_cpp

keops_binder = {"nvrtc": LoadKeOps_nvrtc.LoadKeOps_nvrtc,
                "cpp": LoadKeOps_cpp.LoadKeOps_cpp}