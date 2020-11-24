import os
import sys
import subprocess

import pykeops.config
from pykeops.common.parse_type import check_aliases_list
from pykeops.common.utils import c_type


def run_and_display(args, build_folder, msg=''):
    """
    This function run the command stored in args and display the output if needed
    :param args: list
    :param msg: str
    :return: None
    """
    os.makedirs(build_folder, exist_ok=True)
    try:
        proc = subprocess.run(args, cwd=build_folder, stdout=subprocess.PIPE, check=True)
        if pykeops.config.verbose:
            print(proc.stdout.decode('utf-8'))

    except subprocess.CalledProcessError as e:
        print('\n--------------------- ' + msg + ' DEBUG -----------------')
        print(e)
        print(e.stdout.decode('utf-8'))
        print('--------------------- ----------- -----------------')
    
def compile_generic_routine(formula, aliases, dllname, dtype, lang, optional_flags, build_folder):
    
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
    
    template_name = 'libKeOps' + lang
    target_tag = str.replace(dllname,template_name,'')            
    template_tag = 'XXXtemplateXXX'
    template_loc_tag = 'LOCtemplateLOC'
    templates_folder = pykeops.config.bin_folder
    template_build_folder = templates_folder + '/build-' + template_name
    pykeops_root_folder = pykeops.config.bin_folder + '/../'
    template_include_file = pykeops_root_folder + '/' + template_name + '.h'
    template_dllname = template_name + template_tag
    target_include_file = build_folder+'/'+template_dllname+'.h'

    def replace_string_in_file(filename, source_string, target_string):
        # replaces all occurences of source_string by target_string in file named filename
        subprocess.run(['LC_ALL=C sed -i.bak "s/'+source_string+'/'+target_string+'/g" '+filename+' && rm '+filename+'.bak'],shell=True)
    
    def copy_directory(source_folder,target_folder):
        # copy source_folder to target_folder
        subprocess.run(['cp -rf '+source_folder+' '+target_folder],shell=True)

    def copy_directory_content(source_folder,target_folder):
        # copy all content inside source_folder to (already existing) target_folder
        subprocess.run(['cp -rf '+source_folder+'/* '+target_folder],shell=True)

    def replace_string_in_all_files(source_string, target_string, folder):
        # replaces all occurences of source_string by target_string in all files 
        # belonging to folder or recursively any of its sub-folders.
        subprocess.run(["export LC_ALL=C && export LANG=C && export LC_CTYPE=C && grep -rli '"+source_string+"' "+folder+"/* | xargs -I@ sed -i.bak 's/"+source_string+"/"+target_string+"/g' @"],shell=True)

    def copy_file(source_file, target_file):
        # simple file copy
        subprocess.run(['cp '+source_file+' '+target_file], shell=True)
    
    def move_files(source_file, target_file):
        # simple file move
        subprocess.run(['mv '+source_file+' '+target_file], shell=True)
    
    def make_directory(dirname):
        # make directory
        subprocess.run(['mkdir '+dirname], shell=True)
    
    def create_keops_formula_include_file(dtype, formula, alias_string, target_include_file):
        replace_pairs = [
                ('@__TYPE__@', c_type[dtype]),
                ('@FORMULA_OBJ@', formula),
                ('@VAR_ALIASES@', alias_string),
                ('@NARGS@', 3),
                ('@POS_FIRST_ARGI@', 0),
                ('@POS_FIRST_ARGJ@', 1),
                ]
        for p in replace_pairs:
            replace_string_in_file(target_include_file, p[0], '{}'.format(p[1]))
    
    if not os.path.isdir(template_build_folder):
        
        command_line = ["cmake", pykeops.config.script_folder,
                     "-DCMAKE_BUILD_TYPE=" + "'{}'".format(pykeops.config.build_type),
                     "-Dshared_obj_name=" + "'{}'".format(template_dllname),
                     "-D__TYPE__=" + "'{}'".format(c_type[dtype]),
                     "-DPYTHON_LANG=" + "'{}'".format(lang),
                     "-DPYTHON_EXECUTABLE=" + "'{}'".format(sys.executable),
                     "-DPYBIND11_PYTHON_VERSION=" + "'{}'".format(str(sys.version_info.major) + "." + str(sys.version_info.minor)),
                     "-DC_CONTIGUOUS=1",
                    ] + optional_flags
                    
        run_and_display(command_line + ["-DcommandLine=" + " ".join(command_line)],
                    build_folder,
                    msg="CMAKE")
        
        # creating keops formula include file :
        subprocess.run(['cp '+template_include_file+' '+target_include_file], shell=True)
        create_keops_formula_include_file(dtype, formula, alias_string, target_include_file)
        
        run_and_display(["cmake", "--build", ".", "--target", template_dllname, "--", "VERBOSE=1"], build_folder, msg="MAKE")
        
        copy_directory(build_folder, template_build_folder)
        
        def remove_files(expr):
            # remove file(s)
            subprocess.run(['rm -f ',expr], shell=True)
            
        remove_files(template_build_folder+'/*.lock')
        remove_files(template_build_folder+'/'+template_dllname+'.h')
        remove_files(template_build_folder+'/'+template_dllname+'.so')
        replace_string_in_all_files(target_tag, template_loc_tag, template_build_folder)    
        
    copy_directory_content(template_build_folder,build_folder)
            
    replace_string_in_all_files(template_loc_tag, target_tag, build_folder)
          
    # creating keops formula include file :        
    copy_file(template_include_file, target_include_file)
    create_keops_formula_include_file(dtype, formula, alias_string, target_include_file)
    
    dllfolder = build_folder+'/../'+dllname+'.dir'
    make_directory(dllfolder)
    move_files(build_folder+'/'+template_dllname+'*.so',dllfolder)
    
    #pykeops.config.verbose = True
    subprocess.run(['cd ' + build_folder + ' && make clean'], shell=True)
    run_and_display(["cmake", "--build", ".", "--target", 'keops'+template_dllname, "--", "VERBOSE=1"], build_folder, msg="MAKE")
    
    move_files(build_folder+'/../'+template_dllname+'*.so', dllfolder)
    
    print('Done.')


def compile_specific_conv_routine(dllname, dtype, build_folder):
    print('Compiling ' + dllname + ' using ' + dtype + ' in ' + os.path.realpath(build_folder + os.path.sep + '..' ) + '... ', end='', flush=True)
    run_and_display(['cmake', pykeops.config.script_folder,
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
    run_and_display(['cmake', pykeops.config.script_folder,
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
