function cmake = getcmake()
    % check wether cmake is available from Matlab. Since matlab overrides
    % default search path, we may need to ask the user to enter its 
    % location manually the first time and then save the path in a local file
    S = dbstack('-completenames');
    pathtocmakefile = [fileparts(S(1).file),'/pathtocmake'];
    if exist(pathtocmakefile,'file')
        fid = fopen(pathtocmakefile);
        pathtocmake = fgetl(fid);
        fclose(fid);
    else
        pathtocmake = '';
    end
    [testcmake,~]=system([pathtocmake,'cmake']);
    if testcmake==127 % cmake is not found
        system(['rm -f ',pathtocmakefile]);
        [testcmake,~]=system('cmake');
        if testcmake==127
            pathtocmake = '';
        else
            pathtocmake = input('cmake command is required but was not found. Enter path to cmake command here : ','s');
            if length(pathtocmake)>4 && strcmp(pathtocmake(end-4:end),'cmake')
                pathtocmake = pathtocmake(1:end-5);
            end
            if ~isempty(pathtocmake) && pathtocmake(end)~='/'
                pathtocmake = [pathtocmake,'/'];
            end
            [testcmake,~]=system([pathtocmake,'cmake']);
            if testcmake==127
                error('cmake command not found.')
            end
            if ~isempty(pathtocmake)
                fid = fopen(pathtocmakefile,'w');
                fprintf(fid,pathtocmake);
                fclose(fid);
            end
        end
    end
    % now cmake is found, but it may not work
    % usually the problem comes from wrong paths to shared libraries
    % we try several workarounds
    [testcmake,~]=system([pathtocmake,'cmake']);
    if testcmake==1        
        tmp = getenv('LD_PRELOAD');
        % first workaround
        setenv('LD_PRELOAD','')
        [testcmake,~]=system([pathtocmake,'cmake']);
        if testcmake==1
            % second workaround
            setenv('LD_PRELOAD','/usr/lib/x86_64-linux-gnu/libstdc++.so.6')
            [testcmake,~]=system([pathtocmake,'cmake']);
            if testcmake==1
                % problem is elsewhere, we come back to original config and
                % issue an error
                setenv('LD_PRELOAD',tmp)
                system([pathtocmake,'cmake']) % to display the error message
                error('There was a problem with the cmake command.')
            end
        end
    end
    cmake = [pathtocmake,'cmake'];
end

