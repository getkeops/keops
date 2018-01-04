function cstr = encodestring(str)
% u = unicode2native(str);
% u = dec2hex(u);
% u = u';
% cstr = u(:)';

cstr = strrep(str,'<','Z');
cstr = strrep(cstr,'>','Y');
cstr = strrep(cstr,',','X');
cstr = strrep(cstr,'(','W');
cstr = strrep(cstr,')','V');

