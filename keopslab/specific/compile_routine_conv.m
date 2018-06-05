function compile_routine_conv(conv_type)
% This function call cmake to compile the specific cuda code
% that compute the shapes scalar product.

    disp('Formula is not compiled yet ; compiling...')
    
    [src_dir,build_dir,precision,verbosity] = default_options();
    
    % it seems to be a workaround to flush Matlab's default LD_LIBRARY_PATH
    setenv('LD_LIBRARY_PATH','') 
    % I do not have a better option to set working dir...
    cur_dir= pwd; cd(build_dir) ;
    % find cmake :
    cmake = getcmake();
    % cmake command:
    cmdline = ['cmake ', src_dir , ' -D__TYPE__="', precision, '" -DMatlab_ROOT_DIR="',matlabroot,'"' ];
    %fprintf([cmdline,'\n'])
    try
        [prebuild_status,prebuild_output] = system(cmdline);
        [build_status,build_output]  = system(['make mex_conv',conv_type,' VERBOSE=1']);
        if (verbosity ==1) || (prebuild_status ~=0) || (build_status ~= 0)
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

    fname = ['conv.',mexext];
    testbuild = (exist(fname,'file')==3);
    if  testbuild
        disp('Compilation succeeded')
    else
        error(['File "', fname, '" not found!'])
    end
end

