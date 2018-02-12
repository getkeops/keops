import os.path
import subprocess


def compile_generic_routine(aliases, formula, dllname, cuda_type, script_folder=None):
    print('Tried to load ' + dllname + ", ", end='')
    print("but could not find the DLL. Compiling it... ", end='')

    if script_folder is None:
        script_folder = os.path.dirname(os.path.abspath(__file__)) + os.path.sep + ('..' + os.path.sep) * 2 + "cuda" + os.path.sep + "autodiff" + os.path.sep

    def process_alias(alias):
        return "auto " + str(alias) + "; "

    alias_string = "\n".join([process_alias(alias) for alias in aliases])

    target = "shared_obj"

    print("\n")
    print(alias_string)
    print("Compiling formula = " + formula + " ... ", end='', flush=True)
    subprocess.run(["cmake", script_folder,"-DUSENEWSYNTAX=TRUE" , "-DFORMULA_OBJ="+formula, "-DVAR_ALIASES="+alias_string, "-Dshared_obj_name="+dllname, "-D__TYPE__="+cuda_type], \
                   cwd=script_folder+"build")
    subprocess.run(["make", target], \
                   cwd=script_folder+"build")
    print("Done. ", end='')

