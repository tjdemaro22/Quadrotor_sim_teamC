classdef simulator < handle

    properties(Access=private)
        quadrotor;
        control;
        uav;
    end

    properties(Access=public)
        epsilon(1,1) double;
        timestep(1,1) double = 0.01;
        simtime(1,2) double = [0, 10];
    end

    properties(Access=public)
        animation_axis;
        airspace_box_length = 10;
        axis_font_size = 14;
    end

    methods(Access=public) % Constructors
        function obj = simulator(quadrotor, control, uav)
            obj.quadrotor = quadrotor;
            obj.control = control;
            obj.uav = uav;
        end
    end


    methods(Access=public)
        function [t,z,u,d,y] = simulate(obj, z0)
            ts = obj.simtime(1):obj.timestep:obj.simtime(end);

            t = []; z = []; u = []; d = []; y = [];

            iscaptured = false;

            for k=1:length(ts)-1
                tspan = [ts(k), ts(k+1)];
                [t_, z_, u_, d_, y_] = obj.step(iscaptured,tspan, z0);
                % t = [t; t_(1:end-1,:)];
                % z = [z; z_(1:end-1,:)];
                % u = [u; u_(1:end-1,:)];
                % d = [d; d_(1:end-1,:)];
                % y = [y; y_(1:end-1,:)];

                if k == 1
                    t = [t; t_([1,end])];
                    z = [z; z_([1,end],:)];
                    u = [u; u_([1,end],:)];
                    d = [d; d_([1,end],:)];
                    y = [y; y_([1,end],:)];
                else
                    t = [t; t_(end)];
                    z = [z; z_(end,:)];
                    u = [u; u_(end,:)];
                    d = [d; d_(end,:)];
                    y = [y; y_(end,:)];
                end

                z0 = z(end,:)';

                if min(vecnorm(z_(:,1:3) - y_,2,2)) < obj.epsilon
                    iscaptured = true;
                end
            end

        end

        function [t,z,u,d,y] = step(obj, iscaptured, tspan, z0)

            y0 = obj.uav.location(iscaptured, tspan(1), z0);
            u = obj.control.output(iscaptured, z0, y0);
            if isnumeric(u) && isvector(u)
                if length(u) ~= 4
                    error("u=control(z0, quadrotor, uav) is not a 4x1 vector.");
                end
            else
                error("u=control(z0, quadrotor, uav) must be numeric vector of size 4.");
            end

            dist = obj.uav.disturbance(iscaptured);
            [t, z] = obj.quadrotor.solve(tspan, z0, u, dist);
            d = [obj.uav.eval(dist.r, t,z), obj.uav.eval(dist.n,t,z)];
            y = obj.uav.location(iscaptured, t, z);
        end


        function buildvisuals(obj, ax)
            obj.quadrotor.draw([], [], ax);
            obj.uav.draw([], [], ax);
        end

        function show(obj, z, y)
            obj.quadrotor.show(z);
            obj.uav.show(y);
        end

        function animate(obj, t, z, y, ax)
            if nargin < 4
                error("One or more inputs are missing: t, z, y are required.");
            end
            if nargin < 5
                ax = obj.buildaxes();
            end

            obj.buildvisuals(ax);
            tic;
            for k=1:length(t)
                if ~isvalid(ax)
                    return
                end
                obj.show(z(k,:), y(k,:));
                drawnow();
                %pause(t(k)-toc);
            end
        end

        function ax = buildaxes(obj)
            if isempty(obj.animation_axis) || ~isvalid(obj.animation_axis)
                obj.animation_axis = axes(Parent=figure(),...
                    NextPlot="add", DataAspectRatio=[1 1 1],...
                    Xlim=obj.airspace_box_length*[-0.5 0.5],...
                    Ylim=obj.airspace_box_length*[-0.5 0.5],...
                    Zlim=obj.airspace_box_length*[0 1],...
                    Box="on", XGrid="on", YGrid="on", Zgrid="on",...
                    TickLabelInterpreter="latex",...
                    FontSize=obj.axis_font_size);

                view(obj.animation_axis, 3);
            end
            ax = obj.animation_axis;
        end

    end

end