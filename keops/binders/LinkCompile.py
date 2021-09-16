import os, time

from keops.utils.code_gen_utils import get_hash_name
from keops.config.config import build_path


class LinkCompile:
    """
    Base class for compiling the map_reduce schemes and providing the dll to KeOps bindings.
    """

    def __init__(self):
        # N.B. Here self is assumed to be populated by the __init__ of one of the MapReduce classes

        # we create the hash string id corresponding to all parameters, e.g. 7b9a611f7e
        self.gencode_filename = get_hash_name(
            type(self),
            self.red_formula_string,
            self.aliases,
            self.nargs,
            self.dtype,
            self.dtypeacc,
            self.sum_scheme_string,
            self.tagHostDevice,
            self.tagCpuGpu,
            self.tag1D2D,
            self.use_half,
            self.device_id
        )

        # info_file is the name of the file that will contain some meta-information required by the bindings, e.g. 7b9a611f7e.nfo
        self.info_file = build_path + os.path.sep + self.gencode_filename + ".nfo"

        # gencode_file is the name of the source file to be created and then compiled, e.g. 7b9a611f7e.cpp or 7b9a611f7e.cu
        self.gencode_file = (
            build_path
            + os.path.sep
            + self.gencode_filename
            + "."
            + self.source_code_extension
        )

    def save_info(self):
        # create info_file to save some parameters : dim (dimension of output vectors),
        #                                            tagI (O or 1, reduction over i or j indices),
        #                                            dimy (sum of dimensions of j-indexed vectors)
        f = open(self.info_file, "w")
        f.write(f"red_formula={self.red_formula_string}\ndim={self.dim}\ntagI={self.tagI}\ndimy={self.dimy}")
        f.close()

    def read_info(self):
        # read info_file to retreive dim, tagI, dimy
        f = open(self.info_file, "r")
        string = f.read()
        f.close()
        tmp = string.split("\n")
        if len(tmp) != 4:
            raise ValueError("incorrect info file")
        tmp_dim, tmp_tag, tmp_dimy = (
            tmp[1].split("="),
            tmp[2].split("="),
            tmp[3].split("="),
        )
        if (
            len(tmp_dim) != 2
            or tmp_dim[0] != "dim"
            or len(tmp_tag) != 2
            or tmp_tag[0] != "tagI"
            or len(tmp_dimy) != 2
            or tmp_dimy[0] != "dimy"
        ):
            raise ValueError("incorrect info file")
        self.dim = eval(tmp_dim[1])
        self.tagI = eval(tmp_tag[1])
        self.dimy = eval(tmp_dimy[1])

    def write_code(self):
        # write the generated code in the source file ; this is used as a subfunction of compile_code
        f = open(self.gencode_file, "w")
        f.write(self.code)
        f.close()

    def get_dll_and_params(self):
        # main method of the class : it compiles - if needed - the code and returns the name of the dll to be run for
        # performing the reduction, e.g. 7b9a611f7e.so, or in the case of JIT compilation, the name of the main KeOps dll,
        # and the name of the assembly code file.
        if not os.path.exists(self.file_to_check):
            print(
                "[KeOps] Compiling formula :",
                self.red_formula,
                "...",
                flush=True,
                end=" ",
            )
            self.compile_code()
            self.save_info()
            print("OK")
        else:
            self.read_info()
        return dict(
            dllname=self.true_dllname,
            low_level_code_file=self.low_level_code_file,
            tagI=self.tagI,
            use_half=self.use_half,
            tag1D2D=self.tag1D2D,
            dimred=self.red_formula.dimred,
            dim=self.dim,
            dimy=self.dimy,
            indsi=self.varloader.indsi,
            indsj=self.varloader.indsj,
            indsp=self.varloader.indsp,
            dimsx=self.varloader.dimsx,
            dimsy=self.varloader.dimsy,
            dimsp=self.varloader.dimsp,
        )
