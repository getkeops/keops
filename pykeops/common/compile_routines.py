import subprocess
from pykeops import build_folder, script_folder, verbose

def compile_generic_routine(aliases, formula, dllname, cuda_type):
    stdout = subprocess.DEVNULL if not verbose else None

    def process_alias(alias):
        return "auto " + str(alias) + "; "

    def process_disp_alias(alias):
        return str(alias) + "; "

    alias_string = "".join([process_alias(alias) for alias in aliases])
    alias_disp_string = "".join([process_disp_alias(alias) for alias in aliases])

    target = "keops"

    print("Compiling formula : " + formula + " with " + alias_disp_string + " ... ", end='', flush=True)
    subprocess.run(["cmake", script_folder, "-DPYTHON_LIB=TRUE", "-DUSENEWSYNTAX=TRUE" , "-DFORMULA_OBJ="+formula, "-DVAR_ALIASES="+alias_string, "-Dshared_obj_name="+dllname, "-D__TYPE__="+cuda_type], \
                   cwd=build_folder,stdout=stdout)
    subprocess.run(["make", target], \
                   cwd=build_folder,stdout=stdout)
    print("Done. ", end='', flush=True)

def compile_specific_routine(dllname, cuda_type):
    #print('Tried to load ' + dllname + ", ", end='')
    #print("but could not find the DLL. Compiling it... ", end='')
    stdout = subprocess.DEVNULL if not verbose else None

    print("Compiling "+dllname+" ... ", end='', flush=True)
    subprocess.run(["cmake", script_folder, "-DPYTHON_LIB=TRUE", "-Dshared_obj_name="+dllname, "-D__TYPE__="+cuda_type], cwd=build_folder, check=True,stdout=stdout)
    subprocess.run(["make", dllname], cwd=build_folder,check=True,stdout=stdout)
    print("Done. ", end='', flush=True)

