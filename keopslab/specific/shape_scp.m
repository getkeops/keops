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

if ~(exist(create_mex_name(opt.kernel_geom,opt.kernel_signal,opt.kernel_sphere,'',mexext),'file')==3)
    compile_routine_shape_dist(opt.kernel_geom,opt.kernel_signal,opt.kernel_sphere,'')
end

eval(['res = sum(', create_mex_name(opt.kernel_geom,opt.kernel_signal,opt.kernel_sphere,''),'(center_faceX'',center_faceY'',signalX'',signalY'',normalsX'',normalsY'',kernel_size_geom,kernel_size_signal,kernel_size_sphere));']);

end
