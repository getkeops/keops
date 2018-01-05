classdef KernelClass < handle
    properties
        Fname
        indxy
    end
    methods
        function obj = KernelClass(varargin)
            obj.indxy = [-1,-1,nargin-1];
            formula = varargin{nargin};
            for k=1:nargin-1
                [varname,vartype] = sepeqstr(varargin{k});
                formula = strrep(formula,varname,vartype);
                if vartype(1)=='x' && obj.indxy(1)==-1
                    obj.indxy(1) = k-1;
                elseif vartype(1)=='y' && obj.indxy(2)==-1
                    obj.indxy(2) = k-1;
                end
            end
            filename = [formula,'.',mexext];
            if ~(exist(filename,'file')==3)
                buildFormula(formula)
            end
            obj.Fname = ['F',clockstr,num2str(floor(rand(1)*1e16))];
            eval(['!cp "build/',filename,'" build/',obj.Fname,'.',mexext])
        end
        function out = Eval(obj,varargin)
            out = feval(obj.Fname,obj.indxy,varargin{:});
        end
        function delete(obj)
            eval(['!rm build/',obj.Fname,'.',mexext])
        end
    end
end

function buildFormula(F)
mysetenv('PATH','/Developer/NVIDIA/CUDA-9.1/bin/')
mysetenv('PATH','/Applications/MATLAB_R2014a.app/bin/')
cd ..
eval(['!./compile_mex_cpu "',F,'"'])
cd matlab_bindings
eval(['!mv build/tmp.',mexext,' "build/',F,'.',mexext,'"'])
end

function mysetenv(var,string)
if isempty(strfind(getenv(var),string))
    setenv(var, [getenv(var) ':' string])
end
end

function c = clockstr
c = num2str(clock');
c(:,end+1) = '_';
c = c';
c = strrep(c(:)',' ','');
c = strrep(c,'.','_');
end

function [left,right] = sepeqstr(str)
% get string before and after equal sign
pos = find(str=='=');
left = str(1:pos-1);
right = str(pos+1:end);
end



