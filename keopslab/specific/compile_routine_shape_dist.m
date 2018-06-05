function compile_routine_shape_dist(kernel_geom,kernel_sig,kernel_sphere,ext_name)
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
    cmdline = ['cmake ', src_dir , ' -D__TYPE__="', precision, '" -DKERNEL_GEOM="',lower(kernel_geom), '" -DKERNEL_SIG="', lower(kernel_sig),'" -DKERNEL_SPHERE="', lower(kernel_sphere) ,'" -DMatlab_ROOT_DIR="',matlabroot,'"' ];
    %fprintf([cmdline,'\n'])
    try
        [prebuild_status,prebuild_output] = system(cmdline);
        [build_status,build_output]  = system(['make mex_fshape_scp',ext_name,' VERBOSE=1']);
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

    testbuild = (exist(CreateMexName(kernel_geom,kernel_sig,kernel_sphere,ext_name,mexext),'file')==3);
    if  testbuild
        disp('Compilation succeeded')
    else
        error(['File "',CreateMexName(kernel_geom,kernel_sig,kernel_sphere,ext_name,mexext), '" not found!'])
    end
end

