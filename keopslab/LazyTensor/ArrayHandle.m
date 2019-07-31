classdef ArrayHandle < handle
    properties
        array
    end
    methods
        function obj = ArrayHandle(x)
            obj.array = x;
        end
    end
end

