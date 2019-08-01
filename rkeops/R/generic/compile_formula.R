# function testbuild = compile_formula(code1, code2, filename)
# % This function call cmake to compile the generic cuda code
# 
# cmd_cmake =  [ '-DVAR_ALIASES="', code1, '" -DFORMULA_OBJ="',...
#                code2,'" -DUSENEWSYNTAX=TRUE', ' -Dmex_name="',...
#                filename, '" -Dshared_obj_name="', filename, '"'];
#            
# cmd_make = 'mex_cpp'; 
# 
# mexname = ['keops', filename];
# 
# msg = [':\n        Formula: ',code2, '\n        aliases: ',code1];
# 
# testbuild = compile_code(cmd_cmake, cmd_make, mexname, msg);
#                       
# end
