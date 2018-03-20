function [res] = shape_scp(center_faceX,center_faceY,signalX,signalY,normalsX,normalsY,kernel_size_geom,kernel_size_signal,kernel_size_sphere,opt)
% This function computes the (oriented) varifold scalar product 
% between two fshapes as docummented in [Kaltenmark, Charlier, Charon - CVPR2017].
% Basically it computes 
%  sum(sum(...
%          kernel_geom(  |center_faceX - center_faceY|^2 / kernel_size_geom ) ...
%        * kernel_sig(   |signalX - signalY|^2 )  ...
%        * kernel_sphere(<unit_normalX, unit_normalsY>) ...
%        ))
%
% Inputs
%   center_faceX: a (nx x d) matrix
%   center_faceY: a (ny x d) matrix
%   signalX: a (nx x 1) matrix
%   signalY: a (ny x 1) matrix
%   normalsX: a (nx x d) matrix
%   normalsY: a (ny x d) matrix
%   kernel_size_geom: a positive real number
%   kernel_size_signal: a positive real number
%   kernel_size_sphere: a positive real number
%   opt: a struct with fields 
%             'kernel.geom' (should be 'gaussian' or 'cauchy')
%             'kernel.sig' (should be 'gaussian' or 'cauchy')
%             'kernel.sphere' (should be 'gaussian_unoriented', 'binet', 'gaussian_oriented' or 'linear')
%        Note: for binet and linear kernel, the value of kernel_sphere is not used (should be anything)
%
% Output
%   res : a real number
%

if ~(exist(CreateMexName(opt.kernel_geom,opt.kernel_signal,opt.kernel_sphere,'',mexext),'file')==3)
    buildRoutine(opt.kernel_geom,opt.kernel_signal,opt.kernel_sphere,'')
end

eval(['res = sum(', CreateMexName(opt.kernel_geom,opt.kernel_signal,opt.kernel_sphere,''),'(center_faceX'',center_faceY'',signalX'',signalY'',normalsX'',normalsY'',kernel_size_geom,kernel_size_signal,kernel_size_sphere));']);

end


function buildRoutine(kernel_geom,kernel_sig,kernel_sphere,ext_name)
% This function call cmake to compile the specific cuda code
% that compute the shapes scalar product.

    disp('Formula is not compiled yet ; compiling...')
    
    [src_dir,build_dir,precision] = default_options();
    
    % it seems to be a workaround to flush Matlab's default LD_LIBRARY_PATH
    setenv('LD_LIBRARY_PATH','') 
    % I do not have a better option to set working dir...
    cur_dir= pwd; cd(build_dir) ;
    % cmake command:
    cmdline = ['cmake ', src_dir , ' -D__TYPE__="', precision, '" -DKERNEL_GEOM="',lower(kernel_geom), '" -DKERNEL_SIG="', lower(kernel_sig),'" -DKERNEL_SPHERE="', lower(kernel_sphere) ,'" -DMatlab_ROOT_DIR="',matlabroot,'"' ];
    fprintf([cmdline,'\n'])
    try
        [~,prebuild_output] = system(cmdline)
        [~,build_output]  = system(['make mex_fshape_scp',ext_name])
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
