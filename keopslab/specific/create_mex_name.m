function mex_name = create_mex_name(kernel_geom, kernel_sig, kernel_sphere, ext_name, mex_name)
if nargin == 4
    mex_name = ['keops_fshape_scp',ext_name,'_',lower(kernel_geom),lower(kernel_sig),lower(kernel_sphere)];
elseif nargin == 5
    mex_name = ['keops_fshape_scp',ext_name,'_',lower(kernel_geom),lower(kernel_sig),lower(kernel_sphere),'.',mexext];
end
end

