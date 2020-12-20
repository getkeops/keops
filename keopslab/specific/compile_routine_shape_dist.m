function testbuild = compile_routine_shape_dist(kernel_geom, kernel_sig, kernel_sphere, ext_name)
% This function call cmake to compile the specific cuda code
% that compute the shapes scalar product.

cmd_cmake = [' -DKERNEL_GEOM=',lower(kernel_geom), ...
             ' -DKERNEL_SIG=', lower(kernel_sig),' -DKERNEL_SPHERE=', ...
             lower(kernel_sphere)];
           
cmd_make = ['mex_fshape_scp', ext_name];

filename =  create_mex_name(kernel_geom, kernel_sig, kernel_sphere, ext_name);

msg = '';

testbuild = compile_code(cmd_cmake, cmd_make, filename, msg);

end
