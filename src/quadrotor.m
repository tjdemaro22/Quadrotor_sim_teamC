% State vector definition:
%     z(1:3) = [x1; x2; x3] : quadrotor center in world frame
%     z(4:6) = [phi; theta; psi] : roll, pitch and yaw
%     z(7:9) = [v1; v2; v3] : quadrotor velocity in world frame
%   z(10:12) = [w1; w2; w3] : Angular velocity in quadrotor frame
%
%         O u1
%         |
%   u2 O--|--O u4
%         |
%         O u3
%
classdef quadrotor < handle

    %-- Properties --------------------------------------------------------
    properties(Access=public) % System parameters
        g(1,1) double = 9.81; % The gravitational acceleration [m/s^2]
        l(1,1) double = 0.2; % Distance from the center to each rotor [m]
        m(1,1) double = 0.5; % Total mass of the quadrotor [kg]
        I(3,3) double = diag([1, 1, 2]); % Mass moment of inertia [kg m^2]
        mu(1,1) double = Inf; % Maximum thrust of each rotor [N]
        sigma(1,1) double = 0.01; % The proportionality constant relating thrust to torque [m]
        sigma_div_l(1,1) double % = sigma/l
    end

    properties(Access=private) % Function handles
        % Rotation mapping from {C} to {W}
        R = @(phi, theta, psi) rotz(psi)*roty(theta)*rotx(phi);
        
        % Angular velocities in {C} to the rate of roll, pitch and yaw
        T_inv = @(phi, theta) [1, sin(phi)*tan(theta), cos(phi)*tan(theta);
            0, cos(phi), -sin(phi);
            0, sin(phi)/cos(theta), cos(phi)/cos(theta)];
        
        % Place holder for saturation function defined in the initializer
        sat
    end

    properties(Access=private) % Visualization and plotting parameters and handles
        axis_font_size = 14;
        plot_line_width = 1.5;
        animation_axis = [];
        airspace_box_length = 4;
        default_body_color = lines(1);
        default_silhouette_color = 0.5*ones(1,3);
        default_line_width = 2;
        rotor_scale = 0.3;
        rotor(10,3) double
        chassis
        shape
    end

    %-- Methods -----------------------------------------------------------
    methods(Access=public) % Constructors
        function obj = quadrotor(varargin)
            if nargin > 0
                obj.parse(varargin{:})
            end
            obj.sigma_div_l = obj.sigma/obj.l;
            obj.sat = @(u) max(min(u, obj.mu), 0);
            N = 10;
            Q = linspace(0,2*pi,N)';
            obj.rotor = obj.rotor_scale * obj.l*[cos(Q) sin(Q) zeros(size(Q))];
            obj.chassis = obj.l*[1 0 0; 0 1 0; -1 0 0; 0 -1 0];
        end
    end

    methods(Access=public) % Public dynamics and IVP solvers
        function dz = state_space(obj, z, u)
            dz = obj.dynamics(z, u, [0; 0; 0], [0; 0; 0]);
        end

        function [t, z] = solve(obj, tspan, z0, control, dist)
            if nargin < 4
                control = @(t,z) zeros(4,1);
            end

            if isnumeric(control)
                control = @(t,z) control;
            end
            


            if nargin < 5
                dist = struct("r", @(t,z)zeros(3,1), ...
                    "n", @(t,z)zeros(3,1));
            end

            odefun = @(t,z) obj.dynamics(z,...
                obj.sat(control(t,z)),...
                dist.r(t,z), dist.n(t,z) );

            [t, z] = ode45(odefun, tspan, z0);
        end

    end

    methods(Access=private) % Public dynamics
        function dz = dynamics(obj, z, u, r, n)
            % rt = torque vector induced by rotor thrusts
            rt = obj.l*[(u(2)-u(4)); (u(3)-u(1)); (u(1)-u(2)+u(3)-u(4))*obj.sigma_div_l];

            % Computing time derivative of the state vector
            dz(1:3,1) = z(7:9,1);
            dz(4:6,1) = obj.T_inv(z(4), z(5))*z(10:12,1);
            dz(7:9,1) = obj.R(z(4),z(5),z(6))*[r(1);r(2);r(3)+sum(u)]/obj.m - [0;0;obj.g];
            dz(10:12,1) = obj.I\(rt+n-cross(z(10:12,1), obj.I*z(10:12,1)));
        end
    end
    
    methods(Access=public) % Visualization and plotting functions
        function animate(obj, t, z)
            tic;
            k = 1;
            while k <= length(t)
                obj.show(z(k,:));
                pause(t(k)-toc);
                pause(0.01);
                if ~isvalid(obj.animation_axis)
                    return
                end
                k = k + 1;
            end
        end

        function plot(obj, t, z)
            fig = figure();
            ax = gobjects(4,1);
            for i=1:4
                ax(i) = subplot(2,2,i, Parent=fig, NextPlot="add", ...
                    Box="on", XGrid="on", YGrid="on",...
                    Xlim=[t(1), t(end)],...
                    TickLabelInterpreter="latex",...
                    FontSize=obj.axis_font_size);
                xlabel("$t$", Interpreter="latex",...
                    FontSize=obj.axis_font_size);
            end

            plot(ax(1), t,z(:,1:3), LineWidth=obj.plot_line_width);
            legend(ax(1), {'$x_1$', '$x_2$', '$x_3$'},...
                Interpreter="latex", FontSize=obj.axis_font_size);
            title(ax(1), '${\bf x}$', Interpreter="latex",...
                FontSize=obj.axis_font_size);

            plot(ax(3), t, z(:,4:6), LineWidth=obj.plot_line_width);
            legend(ax(3), {'$\phi$', '$\theta$', '$\psi$'},...
                Interpreter="latex", FontSize=obj.axis_font_size);
            title(ax(3), '\boldmath$\alpha$', Interpreter="latex",...
                FontSize=obj.axis_font_size);

            plot(ax(2), t, z(:,7:9), LineWidth=obj.plot_line_width);
            legend(ax(2), {'$\dot{x}_1$', '$\dot{x}_2$', '$\dot{x}_3$'},...
                Interpreter="latex", FontSize=obj.axis_font_size);
            title(ax(2), '$\dot{\bf x}$', Interpreter="latex",...
                FontSize=obj.axis_font_size);

            plot(ax(4), t, z(:,10:12), 'LineWidth', obj.plot_line_width);
            legend(ax(4), {'$\omega_1$', '$\omega_2$', '$\omega_3$'},...
                'Interpreter', 'LaTeX', 'FontSize', obj.axis_font_size);
            title(ax(4), '\boldmath$\omega$', Interpreter="latex",...
                FontSize=obj.axis_font_size);
        end

        function draw(obj, color, linewidth, ax)
            if nargin < 2, color = obj.default_body_color; end
            if nargin < 3, linewidth = obj.default_line_width; end
            if nargin < 4, ax = obj.set_animation_axis(); end
            obj.animation_axis = ax;

            if isempty(color)
                color = obj.default_body_color;
            end
            if isempty(linewidth)
                linewidth = obj.default_line_width;
            end

            obj.shape.silhouette = plot3(0,0,0, '--',...
                Color=obj.default_silhouette_color,...
                LineWidth=linewidth/2,...
                Parent=ax);
            obj.shape.chassis = plot3(0,0,0,...
                Color=color,...
                LineWidth=linewidth,...
                Parent=ax);
            for i=1:4
                obj.shape.rotor(i) = plot3(0,0,0,...
                    Color=color,...
                    LineWidth=linewidth,...
                    Parent=ax);
            end

        end

        function show(obj, z)
            if isempty(obj.animation_axis) || ~isvalid(obj.animation_axis)
                obj.draw();
            end

            z = reshape(z, 1, length(z)); % Convert z to a row vector

            rotation = obj.R(z(4),z(5),z(6))';

            centers = repmat(z(1:3),[4, 1]) + obj.chassis*rotation;
            rotor_points = obj.rotor*rotation;
            
            for i=1:4
                set(obj.shape.rotor(i),...
                    XData=rotor_points(:,1) + centers(i,1),...
                    YData=rotor_points(:,2) + centers(i,2),...
                    ZData=rotor_points(:,3) + centers(i,3));
            end
            
            set(obj.shape.silhouette, XData=[0, z(1), z(1), z(1)],...
                YData=[0, 0, z(2), z(2)],...
                ZData=[0, 0, 0, z(3)]);
            set(obj.shape.chassis,...
                XData=[centers([1 3],1); NaN; centers([2 4],1)], ...
                YData=[centers([1 3],2); NaN; centers([2 4],2)],...
                ZData=[centers([1 3],3); NaN; centers([2 4],3)] );
        end
    end

    methods(Access=private) % Axes initializations
        function ax = set_animation_axis(obj)
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

    methods(Access=private) % Input parsers
        function parse(obj, varargin)
            if length(varargin) == 1
                if isstruct(varargin{1})
                    for field = fieldnames(varargin{1})
                        obj.(field) = varargin{1}.(field);
                    end
                else
                    obj.g = varargin{1}(1);
                    obj.l = varargin{1}(2);
                    obj.m = varargin{1}(3);
                    obj.I = diag(varargin{1}(4:6));
                    obj.mu = varargin{1}(7);
                    obj.sigma = varargin{1}(8);
                end
            else
                obj.g = varargin{1};
                obj.l = varargin{2};
                obj.m = varargin{3};
                obj.I = varargin{4};
                obj.mu = varargin{5};
                obj.sigma = varargin{6};
            end
        end
    end
end

%% External Functions

function R = rotx(q)
    R = [1, 0, 0; 0, cos(q), -sin(q); 0, sin(q), cos(q)];
end

function R = roty(q)
    R = [cos(q), 0, sin(q); 0, 1, 0; -sin(q), 0, cos(q)];
end

function R = rotz(q)
    R = [cos(q), -sin(q), 0; sin(q), cos(q), 0; 0, 0, 1];
end