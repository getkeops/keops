function testbuild = compile_formula(code1, code2, filename)

    disp(['Compiling formula ',code2,' with ',code1,' ...'])
    
    [src_dir,build_dir,precision,verbosity] = default_options();
    
    % it seems to be a workaround to flush Matlab's default LD_LIBRARY_PATH
    setenv('LD_LIBRARY_PATH','') 
    % I do not have a better option to set working dir...
    cur_dir= pwd; cd(build_dir) ;
    % find cmake :
    cmake = getcmake();
    % cmake command:
    cmdline = [cmake,' ', src_dir , ' -DVAR_ALIASES="',code1,'" -DFORMULA_OBJ="',code2,'" -DUSENEWSYNTAX=TRUE -D__TYPE__=',precision,' -Dmex_name="',filename,'" -Dshared_obj_name="',filename,'" -DMatlab_ROOT_DIR="',matlabroot,'"' ];
    %fprintf([cmdline,'\n'])
    try
        [~,prebuild_output] = system(cmdline);
        [~,build_output]  = system(['make mex_cpp']);
        if verbosity ==1
            disp(' ')
            disp('------------------------------------  DEBUG ------------------------------------------')
            disp(' ')
            disp(prebuild_output)
            disp(build_output)
            disp(' ')
            disp('---------------------------------- END  DEBUG ----------------------------------------')
            disp(' ')
        end
    catch
        cd(cur_dir)
        error('Compilation  Failed')
    end
    % ...comming back to curent directory
    cd(cur_dir)

    testbuild = (exist([filename,'.',mexext],'file')==3);
    if  testbuild
        disp('Compilation succeeded')
    else
        error(['File "',filename,'.',mexext, '" not found!'])
    end
end


