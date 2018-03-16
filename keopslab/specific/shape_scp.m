function [] = shape_scp()


mex_name = [Fname,'.',mexext];

if ~(exist(mex_name,'file')==3)
    buildRoutine(CodeVars,formula,Fname,precision,src_dir,build_dir,cur_dir);
end


end


function buildRoutine()

    disp('Formula is not compiled yet ; compiling...')
    
    [src_dir,build_dir,precision] = default_options();
    
    % it seems to be a workaround to flush Matlab's default LD_LIBRARY_PATH
    setenv('LD_LIBRARY_PATH','') 
    % I do not have a better option to set working dir...
    cur_dir= pwd; cd(build_dir) ;
    % cmake command:
    cmdline = ['cmake ', src_dir , '"-D__TYPE__=', precision,' -Dmex_name="',filename,'" -Dshared_obj_name="',filename,'" -DMatlab_ROOT_DIR="',matlabroot,'"' ];
    fprintf([cmdline,'\n'])
    try
        [~,prebuild_output] = system(cmdline)
        [~,build_output]  = system(['make mex_cpp'])
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