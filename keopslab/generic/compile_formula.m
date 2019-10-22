function testbuild = compile_formula(code1, code2, filename, options, tag_no_compile)

if nargin < 5
    tag_no_compile = [];
end

% This function call cmake to compile the generic cuda code

cmd_cmake =  [ '-DVAR_ALIASES="', code1, '" -DFORMULA_OBJ="',...
               code2,'" -DUSENEWSYNTAX=TRUE', ' -Dmex_name="',...
               filename, '" -Dshared_obj_name="', filename, '"'];
           
if options.use_double_acc
    cmd_cmake = [cmd_cmake, ' -D__TYPEACC__=double'];
end
if options.use_blockred
    cmd_cmake = [cmd_cmake, ' -DUSE_BLOCKRED=1'];
end
if options.use_kahan
    cmd_cmake = [cmd_cmake, ' -DUSE_KAHAN=1'];
end
        
cmd_make = 'mex_cpp'; 

mexname = ['keops', filename];

msg = [':\n        Formula: ',code2, '\n        aliases: ',code1];

testbuild = compile_code(cmd_cmake, cmd_make, mexname, msg, tag_no_compile);
                      
end
