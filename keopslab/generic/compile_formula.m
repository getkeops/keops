function testbuild = compile_formula(code1, code2, filename)

    [src_dir,build_dir,precision,verbosity,use_cuda_if_possible] = default_options();
    
    fprintf(['Compiling ', filename, ' in ', build_dir, ':\n        Formula: ',code2,'\n        aliases: ',code1, '\n        dtype: ', precision, '\n ... '])

    % it seems to be a workaround to flush Matlab's default LD_LIBRARY_PATH
    setenv('LD_LIBRARY_PATH','') 
    % I do not have a better option to set working dir...
    cur_dir= pwd; cd(build_dir) ;
    % find cmake :
    cmake = getcmake();
    % cmake command:
    cmdline = [cmake,' ', src_dir , ' -DUSE_CUDA=',num2str(use_cuda_if_possible),' -DCMAKE_BUILD_TYPE=Release -DVAR_ALIASES="',code1,'" -DFORMULA_OBJ="',code2,'" -DUSENEWSYNTAX=TRUE -D__TYPE__=',precision,' -Dmex_name="',filename,'" -Dshared_obj_name="',filename,'" -DMatlab_ROOT_DIR="',matlabroot,'"' ];
    %fprintf([cmdline,'\n'])
    try
        [prebuild_status,prebuild_output] = system(cmdline);
        [build_status,build_output]  = system(['make mex_cpp']);
        if (verbosity ==1) || (prebuild_status ~=0) || (build_status ~= 0)
            disp(' ')
            disp('-------------------------------------  DEBUG  ------------------------------------------')
            disp(' ')
            disp(prebuild_output)
            disp(build_output)
            disp(' ')
            disp('-----------------------------------  END DEBUG  ----------------------------------------')
            disp(' ')
        end
    catch
        cd(cur_dir)
        error('Compilation  Failed')
    end
    % ...coming back to current directory
    cd(cur_dir)

    testbuild = (exist([filename,'.',mexext],'file')==3);
    if  testbuild
        fprintf('Done.\n')
    else
        error(['File "',filename,'.',mexext, '" not found!'])
    end
end


