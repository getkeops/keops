function out = GradFormula(F,var,newvar,varargin)
G = ['Grad<',F,',',var,',',newvar,'>'];
fname = ['F',encodestring(G)];
if ~(exist(fname,'file')==3)
    buildFormula(G)
end
out = feval(fname,varargin{:});