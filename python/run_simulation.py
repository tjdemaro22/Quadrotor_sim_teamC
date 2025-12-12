#!/usr/bin/env python3
"""
Quadrotor Tracking Simulation

This script runs a simulation of a quadrotor tracking a target UAV.
Equivalent to the MATLAB demaro_test_sim.m script.

Usage:
    python run_simulation.py [--no-animation] [--save-gif] [--html-report]
"""

import argparse
import numpy as np
import matplotlib.pyplot as plt
import sys
import os

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from src.quadrotor import Quadrotor
from src.uav import UAV
from src.controller import SACController
from src.simulator import Simulator


def circular_path(t: float) -> np.ndarray:
    """
    Circular trajectory for the target UAV.
    
    Args:
        t: Time
        
    Returns:
        Position [x, y, z]
    """
    x = 2 * np.cos(t)
    y = 2 * np.sin(t)
    z = 5 + np.sin(t)
    return np.array([x, y, z])


def linear_path(t: float) -> np.ndarray:
    """
    Linear trajectory for the target UAV.
    
    Args:
        t: Time
        
    Returns:
        Position [x, y, z]
    """
    x = -1
    y = -3 + t
    z = t
    return np.array([x, y, z])


def diagonal_path(t: float) -> np.ndarray:
    """
    Diagonal trajectory.
    
    Args:
        t: Time
        
    Returns:
        Position [x, y, z]
    """
    x = -3 + t
    y = 5 - t
    z = t
    return np.array([x, y, z])


def sinusoidal_path(t: float) -> np.ndarray:
    """
    Sinusoidal trajectory.
    
    Args:
        t: Time
        
    Returns:
        Position [x, y, z]
    """
    x = -5 + t
    y = np.cos(t)
    z = 5
    return np.array([x, y, z])


def main():
    parser = argparse.ArgumentParser(description='Quadrotor Tracking Simulation')
    parser.add_argument('--no-animation', action='store_true', help='Skip animation, show static plot')
    parser.add_argument('--save-gif', action='store_true', help='Save animation as GIF')
    parser.add_argument('--html-report', action='store_true', help='Generate HTML report')
    parser.add_argument('--trajectory', type=str, default='circular',
                       choices=['circular', 'linear', 'diagonal', 'sinusoidal'],
                       help='Target trajectory type')
    args = parser.parse_args()
    
    print("=" * 60)
    print("Quadrotor Tracking Simulation")
    print("=" * 60)
    
    # =========================================================================
    # QUADROTOR PARAMETERS
    # =========================================================================
    print("\nInitializing quadrotor...")
    
    g = 9.81   # Gravitational acceleration [m/s^2]
    l = 0.2    # Distance from center of mass to each rotor [m]
    m = 0.5    # Total mass [kg]
    I = np.diag([1.24, 1.24, 2.48])  # Mass moment of inertia [kg*m^2]
    mu = 3.0   # Maximum thrust of each rotor [N]
    sigma = 0.01  # Thrust to torque proportionality constant [m]
    
    quad = Quadrotor(g=g, l=l, m=m, I=I, mu=mu, sigma=sigma)
    
    # =========================================================================
    # TARGET UAV (INTRUDER)
    # =========================================================================
    print("Initializing target UAV...")
    
    # Select trajectory
    trajectory_map = {
        'circular': circular_path,
        'linear': linear_path,
        'diagonal': diagonal_path,
        'sinusoidal': sinusoidal_path
    }
    path = trajectory_map[args.trajectory]
    print(f"  Using {args.trajectory} trajectory")
    
    # Disturbance functions (applied when UAV is captured)
    disturbance = {
        'r': lambda t, z: 0.1 * np.array([np.sin(t), np.sin(2*t), np.sin(4*t)]),
        'n': lambda t, z: 0.1 * np.array([0.1, 0.01, 0.1])
    }
    
    intruder = UAV(movement_fn=path, disturbance_fn=disturbance)
    
    # =========================================================================
    # CONTROLLER
    # =========================================================================
    print("Initializing SAC controller...")
    print("  (Computing LQR gains via linearization - this may take a moment)")
    
    ctrl = SACController(quad, timestep=0.01)
    print("  Controller ready!")
    
    # =========================================================================
    # SIMULATION
    # =========================================================================
    print("\nConfiguring simulation...")
    
    sim = Simulator(quad, ctrl, intruder)
    sim.simtime = (0, 10)
    sim.timestep = 0.01
    sim.epsilon = 0.1  # Capture distance
    
    print(f"  Time span: {sim.simtime[0]} to {sim.simtime[1]} seconds")
    print(f"  Time step: {sim.timestep} seconds")
    print(f"  Capture distance: {sim.epsilon} meters")
    
    # Initial conditions
    z0 = np.zeros(12)
    
    print("\nRunning simulation...")
    t, z, u, d, y = sim.simulate(z0)
    print(f"  Simulation complete! {len(t)} time steps")
    
    # Check if capture occurred
    distances = np.linalg.norm(z[:, :3] - y, axis=1)
    min_dist = np.min(distances)
    capture_idx = np.argmin(distances)
    print(f"  Minimum distance to target: {min_dist:.4f} m at t={t[capture_idx]:.2f} s")
    
    if min_dist < sim.epsilon:
        print(f"  ✓ Target captured at t={t[capture_idx]:.2f} s!")
    else:
        print(f"  ✗ Target not captured (threshold: {sim.epsilon} m)")
    
    # =========================================================================
    # VISUALIZATION
    # =========================================================================
    
    if args.html_report:
        print("\nGenerating HTML report...")
        from generate_html_report import generate_html_report
        report_path = generate_html_report(t, z, u, d, y, quad, args.trajectory)
        print(f"  HTML report saved to: {report_path}")
    
    if args.no_animation:
        print("\nShowing static 3D plot...")
        sim.plot_static_3d(z, y, show=True)
    else:
        print("\nStarting animation...")
        print("  (Close the window to continue)")
        
        save_path = 'quadrotor_simulation.gif' if args.save_gif else None
        sim.animate(t, z, y, interval=20, save_path=save_path, show=True)
    
    # Plot states
    print("\nPlotting state history...")
    fig_states = quad.plot_states(t, z)
    
    # Plot control inputs
    print("Plotting control inputs...")
    fig_ctrl = plt.figure(figsize=(10, 6))
    uvec = np.array(ctrl.uvec)
    plt.plot(uvec[:, 0], label='u1')
    plt.plot(uvec[:, 1], label='u2')
    plt.plot(uvec[:, 2], label='u3')
    plt.plot(uvec[:, 3], label='u4')
    plt.xlabel('Time Step')
    plt.ylabel('Rotor Thrust (N)')
    plt.title('Control Inputs')
    plt.legend()
    plt.grid(True)
    
    plt.show()
    
    print("\nSimulation complete!")
    return t, z, u, d, y


if __name__ == '__main__':
    main()

