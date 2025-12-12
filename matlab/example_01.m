clc; clear; close all

addpath('./src');

g = 9.81;  % The gravitational acceleration [m/s^2]
l = 0.2;  % Distance from the center of mass to each rotor [m]
m = 0.5;  % Total mass of the quadrotor [kg]
I = [1.24, 1.24, 2.48];  % Mass moment of inertia [kg m^2]
mu = 3.0;  % Maximum thrust of each rotor [N]
sigma = 0.01;  % The proportionality constant relating thrust to torque [m]

quad = quadrotor(g, l, m, diag(I), mu, sigma);

% Simple Hover Controller
desired_altitude = 2;
K = [3 1];

u = @(t, z) m*g/4 + repmat(K*[desired_altitude - z(3);  -z(9)], [4,1]);

% Simulation
z0 = zeros(12,1);
tspan = [0, 5];

[t,z] = quad.solve(tspan, z0, u);

% Results
quad.animate(t,z)
quad.plot(t,z)
