classdef uav < handle

    %-- Properties --------------------------------------------------------
    properties(Access=private) % System parameters
        movement_fcn
        disturbance_fcn
    end

    properties(Access=private) % Visualization and plotting parameters and handles
        axis_font_size = 14;
        plot_line_width = 1.5;
        airspace_box_length = 4;
        default_body_color = [0, 1]*lines(2);
        default_body_size = 10;
        default_silhouette_color = 0.5*ones(1,3);
        shape
    end

    properties(Access=private)
        zero_dist = struct("r", @(t,z) [0; 0; 0], "n", @(t,z) [0; 0; 0]);
    end

    properties(Access=public) % Visualization and plotting parameters and handles
        animation_axis = [];
    end

    %-- Methods -----------------------------------------------------------
    methods(Access=public) % Constructors
        function obj = uav(movement_fcn, disturbance_fcn, varargin)
            obj.movement_fcn = movement_fcn;
            obj.disturbance_fcn = disturbance_fcn;
            if nargin > 2
                for field = fieldnames(varargin)
                    obj.(field) = varargin.(field);
                end
            end
        end
    end

    methods(Access=public) % Public dynamics and IVP solvers
        function y = location(obj, iscaptured, t, z)
            if iscaptured == false
                if length(t) < 2
                    y = obj.movement_fcn(t);
                else
                    y = zeros(length(t),3);
                    for k=1:length(t)
                        y(k,:) = obj.movement_fcn(t(k))';
                    end
                end
            else
                if length(t) < 2
                    y = z(1:3);
                else
                    y = z(:, 1:3);
                end
            end
        end

        function dist = disturbance(obj, iscaptured)
            if iscaptured == false
                dist = obj.zero_dist;
            else
                dist = obj.disturbance_fcn;
            end
        end

        function y = eval(~,f,t,z)
            for k=1:length(t)
                y(k,:) = f(t(k),z(k,:))';
            end
        end
    end

    methods(Access=private)
        function z = zero(~,t)
            if(length(t) <= 1)
                z = zeros(3,1);
            else
                z = zeros(length(t),3);
            end
        end

        
    end

    methods(Access=public) % Visualization and plotting functions
        function animate(obj, t, y)
            tic;
            k = 1;
            while k <= length(t)
                obj.show(y(k,:));
                pause(t(k)-toc);
                pause(0.01);
                if ~isvalid(obj.animation_axis)
                    return
                end
                k = k + 1;
            end
        end


        function draw(obj, color, bodysize, ax)
            if nargin < 2, color = obj.default_body_color; end
            if nargin < 3, bodysize = obj.default_body_size; end
            if nargin < 4, ax = obj.set_animation_axis(); end
            obj.animation_axis = ax;

            if isempty(color)
                color = obj.default_body_color;
            end
            if isempty(bodysize)
                bodysize = obj.default_body_size;
            end

            obj.shape.silhouette = plot3(0,0,0, '--',...
                Color=obj.default_silhouette_color,...
                Parent=ax);
            obj.shape.body = plot3(0,0,0,...
                Color=color,...
                Marker="o", MarkerEdgeColor="none",...
                MarkerFaceColor=color, MarkerSize= bodysize,...
                Parent=ax);
        end

        function show(obj, y)
            if isempty(obj.animation_axis) || ~isvalid(obj.animation_axis)
                obj.draw();
            end


            set(obj.shape.silhouette, XData=[0, y(1), y(1), y(1)],...
                YData=[0, 0, y(2), y(2)],...
                ZData=[0, 0, 0, y(3)]);
            set(obj.shape.body,...
                XData=y(1), ...
                YData=y(2),...
                ZData=y(3));
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

end