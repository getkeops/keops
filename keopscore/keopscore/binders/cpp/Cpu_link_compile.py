import os
import sysconfig

import keopscore.config
from keopscore.binders.LinkCompile import LinkCompile
from keopscore.utils.messages import KeOps_Error, KeOps_Message
from keopscore.utils.system_utils import KeOps_OS_Run

cpp_dtype = {
    "float": "float",
    "double": "double",
}

cpu_runtime_src = os.path.join(
    keopscore.config.path.get_base_dir_path(),
    "binders",
    "cpp",
    "keops_cpu_runtime.cpp",
)


class Cpu_link_compile(LinkCompile):
    source_code_extension = "cpp"

    def __init__(self):
        LinkCompile.__init__(self)
        self.dllname = os.path.join(
            keopscore.config.path.get_build_folder(),
            self.gencode_filename + "_cpp" + sysconfig.get_config_var("SHLIB_SUFFIX"),
        )
        self.low_level_code_file = self.dllname

        # file to check for existence to detect compilation is needed
        self.file_to_check = self.dllname

    def generate_code(self):
        # method to generate the code and compile it
        # generate the code and save it in self.code, by calling get_code method from CpuReduc class :
        self.get_code()
        self.code += self.get_cpu_dll_entrypoint_code()
        # write the code in the source file
        self.write_code()
        self.compile_dll()
        # retreive some parameters that will be saved into info_file.
        self.tagI = self.red_formula.tagI
        self.dim = self.red_formula.dim

    def compile_dll(self):
        compile_command = (
            f"{keopscore.config.cxx.get_cxx_compiler()} "
            f"{keopscore.config.cxx.get_compile_options()} "
            f"{keopscore.config.openmp.get_compile_options()} "
            f"{keopscore.config.openmp.get_include_options()} "
            f"{keopscore.config.path.get_include_options()} "
            f"{self.gencode_file} "
            f"{cpu_runtime_src} "
            f"{keopscore.config.cxx.get_linking_options()} "
            f"{keopscore.config.openmp.get_linking_options()} "
            f"-o {self.dllname}"
        )
        KeOps_Message(
            "Compiling cpp formula " + self.gencode_filename + " module",
            flush=True,
            end="",
            level=1,
        )
        KeOps_Message(
            " in cache folder " + keopscore.config.path.get_build_folder(),
            flush=True,
            end="",
            level=2,
        )
        KeOps_Message(" ... ", flush=True, end="", use_tag=False, level=1)

        out = KeOps_OS_Run(compile_command)
        if out.returncode != 0:
            KeOps_Error(
                f"Error compiling cpp formula {self.gencode_filename}. See compiler output above."
            )
        else:
            KeOps_Message("OK", use_tag=False, flush=True, level=1)

    def get_cpu_dll_entrypoint_code(self):
        if self.dtype not in cpp_dtype:
            KeOps_Error(
                "The cpp backend only supports float and double formulas, got "
                + self.dtype
                + "."
            )

        scalar_type = cpp_dtype[self.dtype]

        return f"""

#include "binders/cpp/keops_cpu_runtime.h"

extern "C" KEOPS_CPU_EXPORT int keops_cpu_launch(const KeOpsCpuArgs *runtime_args) {{
    return keops_cpu_launch_impl< {scalar_type} >(
        runtime_args, &launch_keops_cpu_{self.gencode_filename}< {scalar_type} >);
}}
"""
