classdef lineartrial2 < handle
    properties(Access = public)
        zdes(12,1) double;
        k(4,12) double;
        u0(1,1) double;
    end


    methods(Access = public)
        function obj = lineartrial2(gains, quadrotor)
            obj.zdes = zeros(12,1)
            obj.u0 = quadrotor.m*quadrotor.g/4;
            obj.k = gains;
        end

        function u = output(obj, ~, z, y)
            obj.zdes(1:3) = y(1:3);
            u = repmat(obj.u0, [4,1]) + obj.k*(obj.zdes - z);
        end
    end


end