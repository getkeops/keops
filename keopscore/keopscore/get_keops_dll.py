"""
This is the main entry point for all binders. It takes as inputs :
  - map_reduce_id : string naming the type of map-reduce scheme to be used : either "CpuReduc", "GpuReduc1D_FromDevice", ...
  - red_formula_string : string expressing the formula, such as "Sum_Reduction((Exp(Minus(Sum(Square((Var(0,3,0) / Var(1,3,1)))))) * Var(2,1,1)),0)",
  - enable_chunks : -1, 0 or 1, for Gpu mode only, enable special routines for high dimensions (-1 means automatic setting)
  - enable_finalchunks : -1, 0 or 1, for Gpu mode only, enable special routines for final operation in high dimensions (-1 means automatic setting)
  - mul_var_highdim : -1, 0 or 1, for Gpu mode only, another option for special routines of final operation in high dimensions (-1 means automatic setting)
  - aliases : list of strings expressing the aliases list, which may be empty,
  - nargs : integer specifying the number of arguments for the call to the routine,
  - dtype : string specifying the float type of the arguments  "float", "double" or "half")
  - dtypeacc : string specifying the float type of the accumulator of the reduction ("float", "double" or "half")
  - sum_scheme_string : string specifying the type of accumulation for summation reductions : either "direct_sum", "block_sum" or "kahan_scheme".
  - tagHostDevice : 0 or 1, for Gpu mode only, use indicates whether data is stored on Host (0) or Gpu Device (1)
  - tagCPUGPU : 0 or 1, indicates whether we use Cpu (0) or Gpu (1) mode, i.e. reduction is performed on Cpu or Gpu
  - tag1D2D : 0 or 1, for Gpu mode only, use 1D (0) or 2D (1) computation map-reduce scheme
  - use_half : 0 or 1, for Gpu mode only, enable special routines for half-precision data type
  - device_id : integer, for Gpu mode only, id of Gpu device to build the code for

It returns
      - tag : string, hash code used as id for the input formula and parameters
      - source_file : string, either :
            - in CPU mode : name of the source file to be compiled
            - in GPU mode : name of the main dll to be called
      - low_level_code_file : string, either :
            - in CPU mode : the empty string ""
            - in GPU mode : name of the low level code (ptx) or binary file (cubin) to be passed to the main dll.
      - tagI : integer, 0 or 1, specifying if reduction must be performed over i or j indices,
      - tagZero : integer, 0 or 1, specifying if reduction just consists in filling output with zeros,
      - use_half : 0 or 1, enable special routines for half-precision data type,
      - cuda_block_size : integer, prefered block size for Gpu kernel
      - use_chunk_mode : 0, 1 or 2, if 1 or 2, enables special routines for high dimensions,
      - tag1D2D : same as input
      - dimred : integer, dimension of the inner reduction operation.
      - dim : integer, dimension of the output tensor.
      - dimy : integer, total dimension of the j indexed variables.
      - indsi : list of integers, indices of i indexed variables.
      - indsj : list of integers, indices of j indexed variables.
      - indsp : list of integers, indices of parameter variables.
      - dimsx : list of integers, dimensions of i indexed variables.
      - dimsy : list of integers, dimensions of j indexed variables.
      - indsp : list of integers, dimensions of parameter variables.

It can be used as a Python function or as a standalone Python script (in which case it prints the outputs):
  - example (as Python function) :
      get_keops_dll("CpuReduc", "Sum_Reduction((Exp(Minus(Sum(Square((Var(0,3,0) / Var(1,3,1)))))) * Var(2,1,1)),0)", 0, 0, 0, [], 3, "float", "float", "block_sum", 0, 0, 0, 0, 0)
  - example (as Python script) :
      python get_keops_dll.py CpuReduc "Sum_Reduction((Exp(Minus(Sum(Square((Var(0,3,0) / Var(1,3,1)))))) * Var(2,1,1)),0)" 0 0 0 "[]" 3 float float block_sum 0 0 0 0 0
"""
import inspect
import sys

import keopscore.config.config
from keopscore.config.config import get_build_folder
import keopscore.mapreduce
from keopscore import cuda_block_size
from keopscore.config.chunks import (
    get_enable_chunk,
    set_enable_chunk,
    dimchunk,
    set_enable_finalchunk,
    use_final_chunks,
    set_mult_var_highdim,
)
from keopscore.formulas import Zero_Reduction, Sum_Reduction
from keopscore.formulas.GetReduction import GetReduction
from keopscore.formulas.variables.Zero import Zero
from keopscore.utils.Cache import Cache
from keopscore.utils.code_gen_utils import KeOps_Error

# Get every classes in mapreduce
map_reduce = dict(inspect.getmembers(keopscore.mapreduce, inspect.isclass))


def get_keops_dll_impl(
    map_reduce_id,
    red_formula_string,
    enable_chunks,
    enable_finalchunks,
    mul_var_highdim,
    aliases,
    *args,
):
    # detecting the need for special chunked computation modes :
    use_chunk_mode = 0
    if "Gpu" in map_reduce_id:
        if not keopscore.config.config.use_cuda:
            KeOps_Error(
                "You selected a Gpu reduce scheme but KeOps is in Cpu only mode."
            )
        set_enable_chunk(enable_chunks)
        set_enable_finalchunk(enable_finalchunks)
        set_mult_var_highdim(mul_var_highdim)
        red_formula = GetReduction(red_formula_string, aliases)
        if use_final_chunks(red_formula) and map_reduce_id != "GpuReduc2D":
            use_chunk_mode = 2
            map_reduce_id += "_finalchunks"
        elif get_enable_chunk() and map_reduce_id != "GpuReduc2D":
            if len(red_formula.formula.chunked_formulas(dimchunk)) == 1:
                from keopscore.mapreduce.Chunk_Mode_Constants import (
                    Chunk_Mode_Constants,
                )

                chk = Chunk_Mode_Constants(red_formula)
                if not chk.chunk_postchunk_mix:
                    use_chunk_mode = 1
                    map_reduce_id += "_chunks"
    # Instantiation of
    map_reduce_class = map_reduce[map_reduce_id]

    map_reduce_obj = map_reduce_class(red_formula_string, aliases, *args)

    # detecting the case of formula being equal to zero, to bypass reduction.
    rf = map_reduce_obj.red_formula
    if isinstance(rf, Zero_Reduction) or (
        isinstance(rf.formula, Zero) and isinstance(rf, Sum_Reduction)
    ):
        if "Gpu" in map_reduce_id:
            map_reduce_class = map_reduce["GpuReduc1D"]
        map_reduce_obj = map_reduce_class.AssignZero(red_formula_string, aliases, *args)
        tagZero = 1
    else:
        tagZero = 0

    res = map_reduce_obj.get_dll_and_params()

    tag1D2D = 0 if tagZero == 1 else res["tag1D2D"]

    return (
        res["tag"],
        res["source_file"],
        res["low_level_code_file"],
        res["tagI"],
        tagZero,
        res["use_half"],
        cuda_block_size,
        use_chunk_mode,
        tag1D2D,
        res["dimred"],
        res["dim"],
        res["dimy"],
        res["indsi"],
        res["indsj"],
        res["indsp"],
        res["dimsx"],
        res["dimsy"],
        res["dimsp"],
    )


get_keops_dll = Cache(
    get_keops_dll_impl,
    use_cache_file=True,
    save_folder=get_build_folder(),
)


if __name__ == "__main__":
    argv = sys.argv[1:]

    argdict = {
        "map_reduce_id": str,
        "red_formula_string": str,
        "enable_chunks": int,
        "enable_finalchunks": int,
        "mul_var_highdim": int,
        "aliases": list,
        "nargs": int,
        "dtype": str,
        "dtypeacc": str,
        "sum_scheme_string": str,
        "tagHostDevice": int,
        "tagCPUGPU": int,
        "tag1D2D": int,
        "use_half": int,
        "device_id": int,
    }

    if len(argv) != len(argdict):
        KeOps_Error(
            f"Invalid call to Python script {sys.argv[0]}. There should be {len(argdict)} arguments corresponding to:\n{list(argdict.keys())}"
        )

    for k, key in enumerate(argdict):
        argtype = argdict[key]
        argval = argv[k] if argtype == str else eval(argv[k])
        if not isinstance(argval, argtype):
            KeOps_Error(
                f"Invalid call to Python script {sys.argv[0]}. Argument number {k + 1} ({key}) should be of type {argtype} but is of type {type(argval)}"
            )
        argdict[key] = argval

    res = get_keops_dll(argdict["map_reduce_id"], *list(argdict.values())[1:])
    for item in res:
        print(item)
