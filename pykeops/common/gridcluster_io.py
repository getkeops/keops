import importlib

from pykeops import build_type
from pykeops.common.compile_routines import compile_gridcluster


def load_gridcluster(dimpoints, cuda_type, lang, optional_flags=[]):
    # Import and compile
    dll_name = "libKeOps_gridcluster_" + lang + "_" + cuda_type + "_dim" + str(dimpoints) 

    compile_module = (build_type == 'Debug')
    print("gridcluster_"+dll_name)

    if not compile_module:
        try:
            gridcluster = importlib.import_module("gridcluster_"+dll_name)
        except ImportError:
            compile_module = True
    
    if compile_module:
        compile_gridcluster(dll_name, dimpoints, cuda_type, lang, optional_flags)
        gridcluster = importlib.import_module("gridcluster_"+dll_name)
        print("Loaded.")

    return gridcluster
