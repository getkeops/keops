function testbuild = compile_routine_conv(conv_type)
% This function call cmake to compile the specific cuda code
% that compute the shapes scalar product.
cmd_cmake = '';

cmd_make = 'mex_conv';

filename = 'conv';

msg = '';

testbuild = compile_code(cmd_cmake, cmd_make, filename, msg);

end
