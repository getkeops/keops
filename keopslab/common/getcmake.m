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
    if testcmake~=0
        system(['rm -f ',pathtocmakefile]);
        [testcmake,~]=system('cmake');
        if testcmake==0
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
            if testcmake~=0
                error('cmake command not found.')
            end
            if ~isempty(pathtocmake)
                fid = fopen(pathtocmakefile,'w');
                fprintf(fid,pathtocmake);
                fclose(fid);
            end
        end
    end
    cmake = [pathtocmake,'cmake'];
end

