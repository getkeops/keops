function testbuild = compile_formula(code1, code2, filename, options, tag_no_compile)

if nargin < 5
    tag_no_compile = [];
end

% This function call cmake to compile the generic cuda code

cmd_cmake =  [ '-DVAR_ALIASES="', code1, '" -DFORMULA_OBJ="', code2,...
               '" -DUSENEWSYNTAX=1', ' -DC_CONTIGUOUS=0',...
               ' -Dshared_obj_name="', filename, '"'];
           
if options.use_double_acc
    cmd_cmake = [cmd_cmake, ' -D__TYPEACC__=double'];
end
if options.sum_scheme
    cmd_cmake = [cmd_cmake, ' -DSUM_SCHEME=',num2str(options.sum_scheme)];
end
        
cmd_make = 'mex_cpp'; 

mexname = ['keops', filename];

msg = [':\n        Formula: ',code2, '\n        aliases: ',code1];

testbuild = compile_code(cmd_cmake, cmd_make, mexname, msg, tag_no_compile);
                      
end
