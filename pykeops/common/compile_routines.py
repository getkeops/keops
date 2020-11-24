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

   
   
    target_tag = str.replace(dllname,'libKeOpstorch','')
    
    def replace_string_in_file(filename, source_string, target_string):
        subprocess.run(['LC_ALL=C sed -i.bak "s/'+source_string+'/'+target_string+'/g" '+filename+' && rm '+filename+'.bak'],shell=True)

    def change_tag_build_folder(foldername, source_tag, target_tag):
        commands = [
            'mv '+foldername+'/CMakeFiles/keopslibKeOpstorch'+source_tag+'.dir '+foldername+'/CMakeFiles/keopslibKeOpstorch'+target_tag+'.dir',
            "rename 's/"+source_tag+"/"+target_tag+"/' "+foldername+"/CMakeFiles/keopslibKeOpstorch"+target_tag+".dir/keops/core/*",
            'mv '+foldername+'/CMakeFiles/libKeOpstorch'+source_tag+'.dir '+foldername+'/CMakeFiles/libKeOpstorch'+target_tag+'.dir',
            "LANG=C grep -rli '"+source_tag+"' "+foldername+"/* | xargs -i@ sed -i 's/"+source_tag+"/"+target_tag+"/g' @",
            "rename 's/"+source_tag+"/"+target_tag+"/' "+foldername+"/*"
            ]
        for c in commands:
            subprocess.run([c], shell=True)
            
    template_tag = 'XXXtemplateXXX'
    template_loc_tag = 'LOCtemplateLOC'
    template_name = 'libKeOpstorch'
    templates_folder = pykeops.config.bin_folder
    template_build_folder = templates_folder + '/build-' + template_name
    pykeops_root_folder = pykeops.config.bin_folder + '/../'
    template_include_file = pykeops_root_folder + '/' + template_name + '.h'
    template_dllname = template_name + template_tag
    
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
        target_include_file = build_folder+'/'+template_dllname+'.h'
        subprocess.run(['cp '+template_include_file+' '+target_include_file], shell=True)
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
        
        run_and_display(["cmake", "--build", ".", "--target", template_dllname, "--", "VERBOSE=1"], build_folder, msg="MAKE")
        
        subprocess.run(['cp','-rf',build_folder,template_build_folder])
        subprocess.run(['rm -f ',template_build_folder+'/*.lock'], shell=True)
        subprocess.run(['rm -f '+template_build_folder+'/'+template_dllname+'.h'], shell=True)
        subprocess.run(['rm -f '+template_build_folder+'/'+template_dllname+'.so'], shell=True)
        #change_tag_build_folder(template_build_folder, target_tag, template_tag)
        subprocess.run(["export LC_ALL=C && export LANG=C && export LC_CTYPE=C && grep -rli '"+target_tag+"' "+template_build_folder+"/* | xargs -I@ sed -i '' 's/"+target_tag+"/"+template_loc_tag+"/g' @"],shell=True)
    
    
    
    subprocess.run(['cp -rf '+template_build_folder+'/* '+build_folder],shell=True)
    subprocess.run(["export LC_ALL=C && export LANG=C && export LC_CTYPE=C && grep -rli '"+template_loc_tag+"' "+build_folder+"/* | xargs -I@ sed -i.bak 's/"+template_loc_tag+"/"+target_tag+"/g' @"],shell=True)
    #change_tag_build_folder(build_folder, template_tag, target_tag)
        
    
    
    # creating keops formula include file :
    target_include_file = build_folder+'/'+template_dllname+'.h'
    subprocess.run(['cp '+template_include_file+' '+target_include_file], shell=True)
    
    dllfolder = build_folder+'/../'+dllname+'.dir'
    subprocess.run(['mkdir '+dllfolder],shell=True)
    subprocess.run(['mv '+build_folder+'/'+template_dllname+'*.so '+dllfolder],shell=True)
    
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
            
    #pykeops.config.verbose = True
    subprocess.run(['cd ' + build_folder + ' && make clean'], shell=True)
    run_and_display(["cmake", "--build", ".", "--target", 'keops'+template_dllname, "--", "VERBOSE=1"], build_folder, msg="MAKE")
    
    subprocess.run(['mv '+build_folder+'/../'+template_dllname+'*.so '+dllfolder],shell=True)
    #subprocess.run(['mv '+dllfolder+'/'+template_dllname+'.cpython-38-x86_64-linux-gnu.so '+dllfolder+'/'+dllname+'.cpython-38-x86_64-linux-gnu.so '],shell=True)
    
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
