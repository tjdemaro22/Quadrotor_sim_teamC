classdef sample_hover_control < handle
    properties(Access = public)
        altitude(1,1) double;
        k(1,2) double;
        u0(1,1) double;
    end


    methods(Access = public)
        function obj = sample_hover_control(altitude, gains, quadrotor)
            obj.u0 = quadrotor.m*quadrotor.g/4;
            obj.altitude = altitude;
            obj.k = gains;
        end

        function u = output(obj, ~, z, ~)
            u = obj.u0 + repmat(obj.k*[(obj.altitude - z(3)); -z(9)],[4,1]);
        end
    end


end