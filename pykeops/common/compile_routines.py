import os
import shutil
import sys
import subprocess

import pykeops.config
from pykeops.common.parse_type import check_aliases_list
from pykeops.common.utils import c_type
from hashlib import sha256

def run_and_display(args, build_folder, msg=''):
    """
    This function run the command stored in args and display the output if needed
    :param args: list
    :param msg: str
    :return: None
    """
    try:
        proc = subprocess.run(args, cwd=build_folder, stdout=subprocess.PIPE, check=True)
        if pykeops.config.verbose:
            print(proc.stdout.decode('utf-8'))

    except subprocess.CalledProcessError as e:
        print('\n--------------------- ' + msg + ' DEBUG -----------------')
        print(e)
        print(e.stdout.decode('utf-8'))
        print('--------------------- ----------- -----------------')
    

def replace_strings_in_file(filename, source_target_string_pairs):
        # replaces all occurences of source_string by target_string in file named filename
        with open(filename, 'r') as file :
            filedata = file.read()
        for source_string, target_string in source_target_string_pairs:
            filedata = filedata.replace(source_string, target_string)
        with open(filename, 'w') as file:
            file.write(filedata)

    
def get_or_build_pybind11_template(dtype, lang, include_dirs):
    
    command_line = ["cmake", pykeops.config.script_template_folder,
                     "-DCMAKE_BUILD_TYPE=" + "'{}'".format(pykeops.config.build_type),
                     "-DPYTHON_LANG=" + "'{}'".format(lang),
                     "-D__TYPE__=" + "'{}'".format(c_type[dtype]),
                     "-DPYTHON_EXECUTABLE=" + "'{}'".format(sys.executable),
                     "-DPYBIND11_PYTHON_VERSION=" + "'{}'".format(str(sys.version_info.major) + "." + str(sys.version_info.minor)),
                     "-DC_CONTIGUOUS=1",
                    ] + include_dirs
    template_name = 'libKeOps_template_' + sha256(''.join(command_line).encode("utf-8")).hexdigest()[:10]
    
    if not os.path.exists(pykeops.config.bin_folder + template_name + '.o'):
        
        print('Compiling pybind11 template ' + template_name + ' in ' + os.path.realpath(pykeops.config.bin_folder) + '...', flush=True)
        #print('(with dtype=',dtype,', lang=',lang,', include_dirs=',include_dirs,')', flush=True)
        
        template_build_folder = pykeops.config.bin_folder + '/build-pybind11_template-' + template_name
        
        os.mkdir(template_build_folder)
        
        command_line += ["-Dshared_obj_name=" + "'{}'".format(template_name)]
        
        run_and_display(command_line + ["-DcommandLine=" + " ".join(command_line)],
                        template_build_folder,
                        msg="CMAKE")
                        
        run_and_display(["cmake", "--build", ".", "--target", template_name, "--", "VERBOSE=1"], template_build_folder, msg="MAKE")
        
        shutil.rmtree(template_build_folder)
        
        print('Done.')
        
    return template_name
    
def create_keops_include_file(build_folder, dllname, dtype, formula, alias_string, optional_flags):
    # creating KeOps include file for formula
    template_include_file = pykeops.config.script_formula_folder + os.path.sep + 'formula.h.in'
    target_include_file = build_folder + os.path.sep + dllname + '.h'
    shutil.copyfile(template_include_file,target_include_file)
    
    optional_flags_string = ''
    for flag in optional_flags:
        if flag[:2] != '-D':
            raise ValueError('Optional flag must be in the form "-D..."')
        macroid, macroval = str.split(flag[2:],'=')
        optional_flags_string += '#define ' + macroid + ' ' + macroval + '\n'
        
    replace_pairs = [
            ('@__TYPE__@', str(c_type[dtype])),
            ('@FORMULA_OBJ@', formula),
            ('@VAR_ALIASES@', alias_string),
            ('@OPTIONAL_FLAGS@', optional_flags_string)
            ]
    replace_strings_in_file(target_include_file, replace_pairs)
    
    
def get_build_folder_name_and_command(dtype, dllname):
    command_line = ["cmake", pykeops.config.script_formula_folder,
                 "-DCMAKE_BUILD_TYPE=" + "'{}'".format(pykeops.config.build_type),
                 "-Dshared_obj_name=" + "'{}'".format(dllname),
                 "-D__TYPE__=" + "'{}'".format(c_type[dtype]),
                 "-DC_CONTIGUOUS=1",
                ]
    build_folder = pykeops.config.bin_folder + '/build-' + sha256(''.join(command_line).encode("utf-8")).hexdigest()[:10]
    return build_folder, command_line

def get_build_folder_name(dtype, dllname):
    build_folder, _ = get_build_folder_name_and_command(dtype, dllname)
    return build_folder

def check_or_prebuild(dtype, dllname):
    build_folder, command_line = get_build_folder_name_and_command(dtype, dllname)
    if not os.path.exists(build_folder+'/CMakeCache.txt'):
        run_and_display(command_line + ["-DcommandLine=" + " ".join(command_line)],
                build_folder,
                msg="CMAKE")


def compile_generic_routine(formula, aliases, dllname, dtype, lang, optional_flags, include_dirs, build_folder):
    
    aliases = check_aliases_list(aliases)

    def process_alias(alias):
        if alias.find("=") == -1:
            return ''  # because in this case it is not really an alias, the variable is just named
        else:
            return 'auto ' + str(alias) + '; '

    def process_disp_alias(alias):
        return str(alias) + '; '

    alias_string = ''.join([process_alias(alias) for alias in aliases])
    alias_disp_string = ''.join([process_disp_alias(alias) for alias in aliases])

    print(
        'Compiling ' + dllname + ' in ' + os.path.realpath(build_folder + os.path.sep + '..' ) + ':\n' + '       formula: ' + formula + '\n       aliases: ' + alias_disp_string + '\n       dtype  : ' + dtype + '\n... ',
        end='', flush=True)

    template_name = get_or_build_pybind11_template(dtype, lang, include_dirs)
    
    create_keops_include_file(build_folder, dllname, dtype, formula, alias_string, optional_flags)
                        
    run_and_display(["cmake", "--build", ".","--target", dllname, "--", "VERBOSE=1"], build_folder, msg="MAKE")
    
    pykeops_folder = os.path.abspath(pykeops.config.script_template_folder+"/../..")
    
    # copy object file link_autodiff.(cpp,cu).o to build_folder
    #subprocess.run(["cp "+build_folder+"/CMakeFiles/"+pykeops.config.template_formula_name+".dir/"+pykeops_folder+"/keops/core/link_autodiff.*.o "+build_folder], cwd=pykeops.config.script_template_folder, shell=True)
    
    # special script "mylink" to do the linking
    subprocess.run([pykeops.config.script_template_folder+"/mylink "+template_name+" "+build_folder+"/link_autodiff.cpp.o "+dllname], cwd=build_folder+"/..", shell=True)
                    
    print('Done.')


def compile_specific_conv_routine(dllname, dtype, build_folder):
    print('Compiling ' + dllname + ' using ' + dtype + ' in ' + os.path.realpath(build_folder + os.path.sep + '..' ) + '... ', end='', flush=True)
    run_and_display(['cmake', pykeops.config.script_specific_folder,
                     '-DCMAKE_BUILD_TYPE=' + pykeops.config.build_type,
                     '-Ushared_obj_name',
                     '-D__TYPE__=' + c_type[dtype],
                     ],
                    build_folder,
                    msg='CMAKE')

    run_and_display(['cmake', '--build', '.', '--target', dllname, '--', 'VERBOSE=1'], build_folder, msg='MAKE')
    print('Done.')


def compile_specific_fshape_scp_routine(dllname, kernel_geom, kernel_sig, kernel_sphere, dtype, build_folder):
    print('Compiling ' + dllname + ' using ' + dtype + ' in ' + os.path.realpath(build_folder + os.path.sep + '..' ) + '... ', end='', flush=True)
    run_and_display(['cmake', pykeops.config.script_specific_folder,
                     '-DCMAKE_BUILD_TYPE=' + pykeops.config.build_type,
                     '-Ushared_obj_name',
                     '-DKERNEL_GEOM=' + kernel_geom,
                     '-DKERNEL_SIG=' + kernel_sig,
                     '-DKERNEL_SPHERE=' + kernel_sphere,
                     '-D__TYPE__=' + c_type[dtype],
                     ],
                    build_folder,
                    msg='CMAKE')

    run_and_display(['cmake', '--build', '.', '--target', dllname, '--', 'VERBOSE=1'], build_folder, msg='MAKE')
    print('Done.')
