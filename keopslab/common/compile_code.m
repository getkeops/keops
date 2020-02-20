function output = compile_code(cmd_cmake, cmd_make, filename, msg, tag_no_compile)

if nargin < 5
    tag_no_compile = '';
end

[src_dir, bin_folder, precision, verbosity, use_cuda_if_possible] = default_options();

% find cmake :
cmake = getcmake();
% cmake command:
cmdline = [cmake, ' ', src_dir , ' -DUSE_CUDA=', num2str(use_cuda_if_possible), ...
    ' -DCMAKE_BUILD_TYPE=Release', ' -D__TYPE__=', precision, ...
    ' -DMatlab_ROOT_DIR="', matlabroot, '" ', cmd_cmake];
cmdline = [cmdline, ' -DcommandLine=''',cmdline,''''];

if strcmp(tag_no_compile,'no_compile')
    output = cmdline;
else
    
    fprintf(['Compiling ', filename, ' in ', bin_folder, msg, '\n        dtype  : ', precision, '\n ... '])
    
    % crerate a separate subfolder to perform the compilation. Shared object files will be automatically copied in currentdir.
    build_folder = fullfile(bin_folder, filename, filesep);
    mkdir(build_folder);
    
    % it seems to be a workaround to flush Matlab's default LD_LIBRARY_PATH
    setenv('LD_LIBRARY_PATH','')
    % I do not have a better option to set working dir...
    cur_dir= pwd; cd(build_folder) ;
    
    try
                
        try
            
            [prebuild_status,prebuild_output] = system(cmdline);
            if (verbosity ==1) || (prebuild_status ~=0)
                fprintf('\n\n-------------------------------------  DEBUG  ------------------------------------------\n\n');
                disp(prebuild_output)
                fprintf('\n\n-----------------------------------  END DEBUG  ----------------------------------------\n\n');
            end
            
            [build_status,build_output] = system(['make ', cmd_make]);
            if (verbosity ==1) || (build_status ~= 0)
                fprintf('\n\n-------------------------------------  DEBUG  ------------------------------------------\n\n');
                disp(build_output)
                fprintf('\n\n-----------------------------------  END DEBUG  ----------------------------------------\n\n');
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
        output = testbuild;
    end
    
    % ...coming back to current directory
    cd(cur_dir);
    % clean build folder
    rmdir(build_folder, 's');
    
    testbuild = (exist([filename,'.',mexext],'file')==3);
    if  testbuild
        fprintf('Done.\n')
    else
        error(['File "',filename,'.',mexext, '" not found!'])
    end
end



