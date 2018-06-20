import subprocess
from pykeops import build_folder, script_folder, verbose, build_type
from pykeops.common.parse_type import check_aliases_list

# def compile_generic_routine(aliases, formula, dllname, cuda_type):
    # stdout = subprocess.DEVNULL if ((not verbose) and (build_type=='Release')) else None

    # def process_alias(alias):
        # return "auto " + str(alias) + "; "

    # def process_disp_alias(alias):
        # return str(alias) + "; "

    # alias_string = "".join([process_alias(alias) for alias in aliases])
    # alias_disp_string = "".join([process_disp_alias(alias) for alias in aliases])

    # target = "keops"

    # print("Compiling formula : " + formula + " with " + alias_disp_string + " ... ", end='', flush=True)
    # subprocess.run(["cmake", script_folder, "-DCMAKE_BUILD_TYPE="+build_type, "-DUSENEWSYNTAX=TRUE" , "-DFORMULA_OBJ="+formula, "-DVAR_ALIASES="+alias_string, "-Dshared_obj_name="+dllname, "-D__TYPE__="+cuda_type], \
                   # cwd=build_folder,stdout=stdout)
    # subprocess.run(["make", target], \
                   # cwd=build_folder,stdout=stdout)
    # print("Done. ", end='', flush=True)


def compile_generic_routine2(formula, aliases, dllname, cuda_type):
    stdout = subprocess.DEVNULL if ((not verbose) and (build_type=='Release')) else None

    aliases = check_aliases_list(aliases)

    def process_alias(alias):
        return "auto " + str(alias) + "; "

    def process_disp_alias(alias):
        return str(alias) + "; "

    alias_string = "".join([process_alias(alias) for alias in aliases])
    alias_disp_string = "".join([process_disp_alias(alias) for alias in aliases])

    target = dllname
    
    print("Compiling formula : " + formula + " with " + alias_disp_string + " ... ", end='', flush=False)
    subprocess.run(["cmake", script_folder, "-DCMAKE_BUILD_TYPE="+build_type, "-DUSENEWSYNTAX=TRUE" , "-DFORMULA_OBJ="+formula, "-DVAR_ALIASES="+alias_string, "-Dshared_obj_name="+dllname, "-D__TYPE__="+cuda_type], cwd=build_folder,stdout=stdout)
    subprocess.run(["make", target], cwd=build_folder,stdout=stdout)
    print("Done. ", end='', flush=False)


def compile_specific_routine(dllname, cuda_type):
    #print('Tried to load ' + dllname + ", ", end='')
    #print("but could not find the DLL. Compiling it... ", end='')
    stdout = subprocess.DEVNULL if ((not verbose) and (build_type=='Release')) else None

    print("Compiling " + dllname + " ... ", end='', flush=False)
    subprocess.run(["cmake", script_folder, "-DCMAKE_BUILD_TYPE="+build_type, "-DPYTHON_LIB=TRUE", "-Dshared_obj_name="+dllname, "-D__TYPE__="+cuda_type], cwd=build_folder, check=True,stdout=stdout)
    subprocess.run(["make", dllname], cwd=build_folder,check=True,stdout=stdout)
    print("Done. ", end='', flush=False)

