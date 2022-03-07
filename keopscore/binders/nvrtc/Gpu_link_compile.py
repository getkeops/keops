import os
from ctypes import create_string_buffer, CDLL, c_int
from os import RTLD_LAZY
import sysconfig

from keopscore.binders.LinkCompile import LinkCompile
import keopscore.config
from keopscore.config.config import (
    cuda_version,
    jit_binary,
    cxx_compiler,
    nvrtc_flags,
    nvrtc_include,
    jit_source_file,
    cuda_available,
    get_build_folder,
)
from keopscore.utils.misc_utils import KeOps_Error, KeOps_Message, KeOps_OS_Run
from keopscore.utils.gpu_utils import get_gpu_props, cuda_include_fp16_path

jit_compile_src = os.path.join(
    os.path.abspath(os.path.dirname(__file__)), "nvrtc_jit.cpp"
)


def jit_compile_dll():
    return os.path.join(
        get_build_folder(),
        "nvrtc_jit" + sysconfig.get_config_var("SHLIB_SUFFIX"),
    )


class Gpu_link_compile(LinkCompile):
    source_code_extension = "cu"
    low_level_code_prefix = "ptx_" #"cubin_" if cuda_version >= 11010 else "ptx_"
    ngpu, gpu_props_compile_flags = get_gpu_props()

    def __init__(self):
        # checking that the system has a Gpu :
        if not (cuda_available and Gpu_link_compile.ngpu):
            KeOps_Error(
                "Trying to compile cuda code... but we detected that the system has no properly configured cuda lib."
            )

        LinkCompile.__init__(self)
        # these are used for JIT compiling mode
        # low_level_code_file is filename of low level code (PTX for Cuda) or binary (CUBIN for Cuda)
        # generated by the JIT compiler, e.g. ptx_7b9a611f7e
        self.low_level_code_file = os.path.join(
            get_build_folder(),
            self.low_level_code_prefix + self.gencode_filename,
        ).encode("utf-8")

        self.my_c_dll = CDLL(jit_compile_dll(), mode=RTLD_LAZY)
        # actual dll to be called is the jit binary, TODO: check if this is relevent
        self.true_dllname = jit_binary
        # file to check for existence to detect compilation is needed
        self.file_to_check = self.low_level_code_file

    def generate_code(self):
        # method to generate the code and compile it
        # generate the code and save it in self.code, by calling get_code method from GpuReduc class :
        self.get_code()
        # write the code in the source file
        self.write_code()
        # we execute the main dll, passing the code as argument, and the name of the low level code file to save the assembly instructions
        self.my_c_dll.Compile(
            create_string_buffer(self.low_level_code_file),
            create_string_buffer(self.code.encode("utf-8")),
            c_int(self.use_half),
            c_int(self.device_id),
            create_string_buffer(
                (cuda_include_fp16_path() + os.path.sep).encode("utf-8")
            ),
        )
        # retreive some parameters that will be saved into info_file.
        self.tagI = self.red_formula.tagI
        self.dim = self.red_formula.dim

    @staticmethod
    def get_compile_command(
        sourcename=jit_source_file, dllname=jit_binary, extra_flags=""
    ):
        # This is about the main KeOps binary (dll) that will be used to JIT compile all formulas.
        # If the dll is not present, it compiles it from source, except if check_compile is False.

        target_tag = (
            "CUBIN" if Gpu_link_compile.low_level_code_prefix == "cubin_" else "PTX"
        )
        nvrtcGetTARGET = "nvrtcGet" + target_tag
        nvrtcGetTARGETSize = nvrtcGetTARGET + "Size"
        arch_tag = (
            '\\"sm\\"'
            if Gpu_link_compile.low_level_code_prefix == "cubin_"
            else '\\"compute\\"'
        )
        target_type_define = f"-DnvrtcGetTARGET={nvrtcGetTARGET} -DnvrtcGetTARGETSize={nvrtcGetTARGETSize} -DARCHTAG={arch_tag}"
        return f"{cxx_compiler} {nvrtc_flags} {extra_flags} {target_type_define} {nvrtc_include} {Gpu_link_compile.gpu_props_compile_flags} {sourcename} -o {dllname}"

    @staticmethod
    def compile_jit_compile_dll():
        KeOps_Message("Compiling cuda jit compiler engine ... ", flush=True, end="")
        KeOps_OS_Run(
            Gpu_link_compile.get_compile_command(
                sourcename=jit_compile_src,
                dllname=jit_compile_dll(),
                extra_flags="-g",
            ),
        )
        KeOps_Message("OK", use_tag=False, flush=True)
