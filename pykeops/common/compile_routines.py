import subprocess
from pykeops import build_folder, script_folder


def compile_generic_routine(aliases, formula, dllname, cuda_type):
    print('Tried to load ' + dllname + ", ", end='')
    print("but could not find the DLL. Compiling it... ", end='')

    def process_alias(alias):
        return "auto " + str(alias) + "; "

    alias_string = "\n".join([process_alias(alias) for alias in aliases])

    target = "keops"

    print("\n")
    print(alias_string)
    print("Compiling formula = " + formula + " ... ")
    subprocess.run(["cmake", script_folder, "-DPYTHON_LIB=TRUE", "-DUSENEWSYNTAX=TRUE" , "-DFORMULA_OBJ="+formula, "-DVAR_ALIASES="+alias_string, "-Dshared_obj_name="+dllname, "-D__TYPE__="+cuda_type], \
                   cwd=build_folder)
    subprocess.run(["make", target], \
                   cwd=build_folder)
    print("Done. ", end='')

def compile_specific_routine(dllname, cuda_type):
    print('Tried to load ' + dllname + ", ", end='')
    print("but could not find the DLL. Compiling it... ", end='')

    print("\n")
    subprocess.run(["cmake", script_folder, "-DPYTHON_LIB=TRUE", "-Dshared_obj_name="+dllname, "-D__TYPE__="+cuda_type], \
                   cwd=build_folder)
    subprocess.run(["make", dllname], \
                   cwd=build_folder)
    print("Done. ", end='')

