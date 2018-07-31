import importlib

from pykeops import build_type
from pykeops.common.utils import create_name
from pykeops.common.compile_routines import compile_generic_routine


def load_keops(formula, aliases, cuda_type, lang):
    # create the name from formula, aliases and cuda_type.
    dll_name = create_name(formula, aliases, cuda_type, lang)
    
    # Import and compile
    compile_module = (build_type == 'Debug')
    
    if not compile_module:
        try:
            myconv = importlib.import_module(dll_name)
        except ImportError:
            compile_module = True
    
    if compile_module:
        compile_generic_routine(formula, aliases, dll_name, cuda_type, lang)
        myconv = importlib.import_module(dll_name)
        print("Loaded.")

    return myconv
