clc; clear; close all

addpath('./src');


path = @(t) [cos(t); sin(t); 2];
dist = struct("r", @(t)0.1*[sin(t); sin(2*t); sin(4*t)],...
    "n", @(t)[0.1; 0.01; 0.1]);

intruder = uav(path, dist);

tspan = linspace(0,10,100);

y = intruder.location(false,tspan,[]);

intruder.animate(tspan, y);

plot3(intruder.animation_axis, y(:,1), y(:,2), y(:,3));

