% Attempting to linearize parts of the system dynamics
clear all;
close all;
clc;

syms x y z xdot ydot zdot phi theta psi omega1 omega2 omega3 u1 u2 u3 u4 ...
    n1 n2 n3 r1 r2 r3

l = 0.2; m = 0.5; I1 = 1.24; I2 = 1.24; I3 = 2.48; g = 9.8; sigma = 0.01;

Rz = [cos(psi) -sin(psi) 0; sin(psi) cos(psi) 0; 0 0 1]; % R3
Ry = [cos(theta) 0 sin(theta); 0 1 0; -sin(theta) 0 cos(theta)]; %R2
Rx = [1 0 0; 0 cos(phi) -sin(phi); 0 sin(phi) cos(phi)]; %R1

R_CE = Rz * Ry * Rx;
R_EC = transpose(Rx) * transpose(Ry) * transpose(Rz);

T = [1 0 -sin(theta); 0 cos(phi) sin(phi)*cos(theta); 0 -sin(phi) cos(phi)*cos(theta)]; % omega = T * alphadot

pos = [x; y; z];
alpha = [phi; theta; psi];
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

z0 = [0 0 1 0 0 0 0 0 0 0 0 0]';
u0 = m*g/4 * ones(4,1);
Ja = jacobian(qdot, q);
Ja_eval = subs(Ja, [q; u; r], [z0; u0; zeros(size(r))]);
A = eval(Ja_eval)
Jb = jacobian(qdot, u);
Jb_eval = subs(Jb, [q; u; r], [z0; u0; zeros(size(r))]);
B = eval(Jb_eval)

rank(ctrb(A,B))

function zdot = f(z,u,r,n)
    Rz = [cos(q(6)) -sin(q(6)) 0; sin(q(6)) cos(q(6)) 0; 0 0 1]; % R3
    Ry = [cos(q(5)) 0 sin(q(5)); 0 1 0; -sin(q(5)) 0 cos(q(5))]; %R2
    Rx = [1 0 0; 0 cos(q(4)) -sin(q(4)); 0 sin(q(4)) cos(q(4))]; %R1
    R_CE = Rz * Ry * Rx;

    T = [1 0 -sin(q(5)); 0 cos(q(4)) sin(q(4))*cos(q(5)); 0 -sin(q(4)) cos(q(4))*cos(q(5))]; % omega = T * alphadot
    I = diag([I1, I2, I3]);
    u = [u1; u2; u3; u4];
    n = [n1; n2; n3];
    r = [r1; r2; r3];
    omega = q(10:12,1);

    pos_dot = q(7:9,1);
    alpha_dot = T\omega;
    vel_dot = (-g * [0; 0; 1]) + (1/m * R_CE * (u(1) + u(2) + u(3) + u(4)) * [0; 0; 1]) + (1/m * R_CE * r);
    omega_dot = I\(((u(2) - u(4)) * l * [1;0;0]) + ((u(3) - u(1)) * l * [0;1;0]) + ((u(1) - u(2) + u(3) - u(4)) * sigma * [0;0;1]) + n - cross(omega,I*omega));
end