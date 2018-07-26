import subprocess
from pykeops import build_folder, script_folder, verbose, build_type, torch_found
from pykeops.common.utils import c_type
from pykeops.common.parse_type import check_aliases_list
from pykeops.common.get_options import torch_include_path


stdout = subprocess.DEVNULL if ((not verbose) and (build_type=='Release')) else None


def compile_generic_routine(formula, aliases, dllname, cuda_type, lang):
    aliases = check_aliases_list(aliases)

    def process_alias(alias):
        return "auto " + str(alias) + "; "

    def process_disp_alias(alias):
        return str(alias) + "; "

    alias_string = "".join([process_alias(alias) for alias in aliases])
    alias_disp_string = "".join([process_disp_alias(alias) for alias in aliases])

    target = dllname
    
    print('Compiling formula : ' + formula + ' with ' + alias_disp_string + ' using ' + cuda_type + ' ... ', end='', flush=False)
    subprocess.run(['cmake', script_folder,
                    '-DCMAKE_BUILD_TYPE=' + build_type,
                    '-DUSENEWSYNTAX=1',
                    '-DFORMULA_OBJ=' + formula,
                    '-DVAR_ALIASES=' + alias_string,
                    '-Dshared_obj_name=' + dllname,
                    '-D__TYPE__=' + c_type[cuda_type],
                    '-DPYTHON_LANG=' + lang,
                    '-DPYTORCH_INCLUDE_DIR=' + torch_include_path,
                    ], cwd=build_folder,stdout=stdout)
    subprocess.run(['make', target, 'VERBOSE=' + str(int(verbose))], cwd=build_folder, stdout=stdout)
    print("Done. ", end='', flush=False)


def compile_specific_conv_routine(target, cuda_type):
    print('Compiling ' + target + ' using ' + cuda_type + ' ... ', end='', flush=False)
    subprocess.run(['cmake', script_folder,
                    '-DCMAKE_BUILD_TYPE=' + build_type,
                    '-Ushared_obj_name',
                    '-D__TYPE__=' + c_type[cuda_type],
                    ], cwd=build_folder, check=True, stdout=stdout)
    subprocess.run(['make', target], cwd=build_folder, check=True, stdout=stdout)
    print('Done. ', end='', flush=False)


def compile_specific_fshape_scp_routine(target, kernel_geom, kernel_sig, kernel_sphere, cuda_type):
    print('Compiling ' + target + ' using ' + cuda_type + ' ... ', end='', flush=False)
    subprocess.run(['cmake', script_folder,
                    '-DCMAKE_BUILD_TYPE=' + build_type,
                    '-Ushared_obj_name',
                    '-DKERNEL_GEOM=' + kernel_geom,
                    '-DKERNEL_SIG=' + kernel_sig,
                    '-DKERNEL_SPHERE=' + kernel_sphere,
                    '-D__TYPE__=' + c_type[cuda_type],
                    ], cwd=build_folder, check=True,stdout=stdout)
    subprocess.run(['make', target], cwd=build_folder, check=True, stdout=stdout)
    print('Done. ', end='', flush=False)