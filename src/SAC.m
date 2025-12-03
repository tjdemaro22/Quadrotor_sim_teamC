classdef SAC < handle
    properties(Access = public)
        altitude(1,1) double;
        k(4, 12) double;
        u0(1,1) double;
        A(12,12) double;
        B(12,4) double;
        prev_coeffs; 
        timeStep(1,1) double;
        output_count(1,1) double;
        target_time(1,1) double;
        jump_ahead_level(1,1) double;
        fast_k(4, 12) double;
        home_k(4, 12) double;
        uvec;
    end

    methods(Access = public)
        function obj = SAC(quadrotor)
            obj.u0 = quadrotor.m*quadrotor.g/4;
            obj.timeStep = 0.01;
            position = [0,0,1];
            Q = diag([5, 5,8,15, 15, 8,1.25,1.25,1.0,3,3,1.5]);
            R = 1.55*eye(4);

            fast_Q = diag([14, 14, 20, 12, 12, 4,1.0,1.0,1.0,1.5,1.5,1.5]);
            fast_R = 0.2*eye(4);

            %home_Q = diag([4, 4, 6, 18, 18, 10,8.0,8.0,8.0,4.5,4.5,4.5]);
            home_Q = eye(12);
            home_R = 3*eye(4);
            
            % ULTRA FAST LINE FOLLOWER
%             Q = diag([10, 10, 20, 15, 15, 4, 0.5, 0.5, 0.5, 3,3,1.5]);
%             R = 2.25*eye(4);
% 
%             fast_Q = diag([14, 14, 16, 12, 12, 4, 0.1, 0.1, 0.5, 1.5,1.5,1.5]);
%             fast_R = 0.2eye(4);
% 
%             home_Q = eye(12);
%             home_R = 6*eye(4);

            [A, B] = linearize_quad(quadrotor, position);
            [K,~, ~] = lqr(A,B,Q,R);
            [fast_K,~, ~] = lqr(A,B,fast_Q,fast_R);
            [home_K,~, ~] = lqr(A,B,home_Q,home_R);

            obj.k = K;
            obj.fast_k = fast_K;
            obj.home_k = home_K;
            obj.output_count = 0;
            obj.prev_coeffs = zeros(3, 8);
            obj.target_time = -1;
            obj.jump_ahead_level = 3.6;

        end

        function u = output(self, isCaptured, z, y)
            kill_dist = 0;
            if isCaptured == false
                %find desired point
                r = zeros(12, 1);
                % determine trajectory of uav
                [~, coeff_size] = size(self.prev_coeffs);
                coeffs = self.solveCoeffs(y, coeff_size - 1);
                % collect data first 15 timesteps
                
                error_vec = z(1:3) - y;
                error_mag = norm(error_vec);
                if (self.output_count > 200)
                    % no target time, attempt to aquire one
                    if (self.target_time == -1)
                        if (error_mag < kill_dist)
                            r(1:3) = solvePoly(coeffs, 0.08);
                            disp("in kill mode");
                            disp(error_mag);
                        else
                            self.target_time = self.jump_ahead_level;
                            self.jump_ahead_level = self.jump_ahead_level + 0.1;
                            r(1:3) = solvePoly(coeffs, self.target_time);
                        end
                    else
                        % update for time passing, and go to the expected
                        % location
                        self.target_time = max(self.target_time - self.timeStep, 0);
                        if ((self.target_time == 0) || (error_mag < (kill_dist*0.1)))
                            self.target_time = -1;
                        end
                        r(1:3) = solvePoly(coeffs, self.target_time);

                    end
                    
                else
                    % update data collection timer, go straight to UAV
                    self.output_count = self.output_count + 1;
                    r(1:2) = [0;0];
                    r(3) = y(3);
                end
                if error_mag < kill_dist
                    temp_k = self.fast_k; % When within range of UAV
                else
                    temp_k = self.k;
                end

                %bound target location
                r(1:2) = min(5, max(-5, r(1:2)));
                r(3) = min(10, max(0, r(3)));

                u = repmat(self.u0, [4,1]) + temp_k*(r - z);
            else
                home = zeros(12, 1);
                if not((abs(z(1)) < 0.5) && (abs(z(2)) < 0.5))
                    home(3) = z(3);
                else
                    home(3) = 0.2;
                end
                u = repmat(self.u0, [4,1]) + self.home_k*(home - z);
            end
            self.uvec = cat(2,self.uvec,u);
        end

        function coeffs = solveCoeffs(self, y, degree)
            threshold = 1e-11; % threshold for rounding errors
            yder = zeros(3, degree + 1);
            yder(:, 1) = y;
            for i = 2:degree + 1
                yder(:, i) = (yder(:, i - 1) - factorial(i - 2)*self.prev_coeffs(:, (degree + 3 - i))) / self.timeStep;
                for j = 1:3
                    if (abs(yder(j, i)) < threshold)
                        yder(j, i) = 0;
                    end
                end
            end
            coeffs = zeros(3, degree + 1);
            for i = 1:degree + 1
                coeffs(:, i) = yder(:, degree + 2 - i) / factorial(degree + 1 - i);
            end

            self.prev_coeffs = coeffs;
        end
    end
end

function futurePose = solvePoly(coeffs, t)
    [~, col] = size(coeffs);
    time = [];
    for i = (col-1):-1:0
        time = [time; (t^(i))];
    end
    futurePose = coeffs*time;
end

function [A, B] = linearize_quad(quad, position)
    syms x y z xdot ydot zdot phi theta psiVar omega1 omega2 omega3 u1 u2 u3 u4 ...
        n1 n2 n3 r1 r2 r3
    l = quad.l; m = quad.m; I1 = quad.I(1, 1); I2 = quad.I(2, 2); I3 = quad.I(3,3); g = quad.g; sigma = quad.sigma; % all in quad object
    
    Rz = [cos(psiVar) -sin(psiVar) 0; sin(psiVar) cos(psiVar) 0; 0 0 1]; % R3
    Ry = [cos(theta) 0 sin(theta); 0 1 0; -sin(theta) 0 cos(theta)]; %R2
    Rx = [1 0 0; 0 cos(phi) -sin(phi); 0 sin(phi) cos(phi)]; %R1
    
    R_CE = Rz * Ry * Rx; 
    R_EC = transpose(Rx) * transpose(Ry) * transpose(Rz);
    
    T = [1 0 -sin(theta); 0 cos(phi) sin(phi)*cos(theta); 0 -sin(phi) cos(phi)*cos(theta)]; % omega = T * alphadot
    
    pos = [x; y; z];
    alpha = [phi; theta; psiVar];
    vel = [xdot; ydot; zdot];
    omega = [omega1; omega2; omega3];
    
    
    q = [pos; alpha; vel; omega];
    
    I = diag([I1, I2, I3]);
    u = [u1; u2; u3; u4];
    n = [n1; n2; n3];
    r = [r1; r2; r3];
    
    pos_dot = [xdot; ydot; zdot];
    alpha_dot = T\omega;
    vel_dot = (-g * [0; 0; 1]) + (1/m * R_CE * (u(1) + u(2) + u(3) + u(4)) * [0; 0; 1]) + (1/m * R_CE * r);
    omega_dot = I\(((u(2) - u(4)) * l * [1;0;0]) + ((u(3) - u(1)) * l * [0;1;0]) + ((u(1) - u(2) + u(3) - u(4)) * sigma * [0;0;1]) + n - cross(omega,I*omega));
    
    qdot = [pos_dot; alpha_dot; vel_dot; omega_dot];
    
    z0 = [position, 0, 0, 0, 0, 0, 0, 0, 0, 0]';
    u0 = m*g/4 * ones(4,1);
    Ja = jacobian(qdot, q);
    Ja_eval = subs(Ja, [q; u; r], [z0; u0; zeros(size(r))]);
    A = eval(Ja_eval);
    Jb = jacobian(qdot, u);
    Jb_eval = subs(Jb, [q; u; r], [z0; u0; zeros(size(r))]);
    B = eval(Jb_eval);
    if rank(ctrb(A,B)) ~= 12
        error("Controllability maxtrix is not full rank.")
    end
end 