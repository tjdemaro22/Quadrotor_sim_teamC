"""
Controller module for quadrotor control.

Includes:
- SACController: Sophisticated Adaptive Controller using LQR
- HoverController: Simple altitude hold controller
"""

import numpy as np
from scipy.linalg import solve_continuous_are
import sympy as sp
from typing import Tuple, Optional
from math import factorial


def compute_lqr_gain(A: np.ndarray, B: np.ndarray, Q: np.ndarray, R: np.ndarray) -> np.ndarray:
    """
    Compute LQR gain matrix K.
    
    Solves the continuous-time algebraic Riccati equation and returns
    the optimal gain matrix K such that u = -K @ x minimizes J.
    
    Args:
        A: State matrix (n x n)
        B: Input matrix (n x m)
        Q: State cost matrix (n x n)
        R: Input cost matrix (m x m)
        
    Returns:
        K: Optimal gain matrix (m x n)
    """
    # Solve ARE: A'P + PA - PBR^{-1}B'P + Q = 0
    P = solve_continuous_are(A, B, Q, R)
    K = np.linalg.solve(R, B.T @ P)
    return K


def linearize_quadrotor(quad, position: np.ndarray = None) -> Tuple[np.ndarray, np.ndarray]:
    """
    Linearize quadrotor dynamics about hover equilibrium.
    
    Uses symbolic computation to derive the Jacobians.
    
    Args:
        quad: Quadrotor object with parameters g, l, m, I, sigma
        position: Equilibrium position [x, y, z] (default: [0, 0, 1])
        
    Returns:
        A: State matrix (12 x 12)
        B: Input matrix (12 x 4)
    """
    if position is None:
        position = np.array([0.0, 0.0, 1.0])
    
    # Define symbolic variables
    x, y, z = sp.symbols('x y z')
    xdot, ydot, zdot = sp.symbols('xdot ydot zdot')
    phi, theta, psi = sp.symbols('phi theta psi')
    omega1, omega2, omega3 = sp.symbols('omega1 omega2 omega3')
    u1, u2, u3, u4 = sp.symbols('u1 u2 u3 u4')
    r1, r2, r3 = sp.symbols('r1 r2 r3')
    n1, n2, n3 = sp.symbols('n1 n2 n3')
    
    # Get parameters from quadrotor
    l = quad.l
    m = quad.m
    I1 = quad.I[0, 0]
    I2 = quad.I[1, 1]
    I3 = quad.I[2, 2]
    g = quad.g
    sigma = quad.sigma
    
    # Rotation matrices
    Rz = sp.Matrix([
        [sp.cos(psi), -sp.sin(psi), 0],
        [sp.sin(psi), sp.cos(psi), 0],
        [0, 0, 1]
    ])
    
    Ry = sp.Matrix([
        [sp.cos(theta), 0, sp.sin(theta)],
        [0, 1, 0],
        [-sp.sin(theta), 0, sp.cos(theta)]
    ])
    
    Rx = sp.Matrix([
        [1, 0, 0],
        [0, sp.cos(phi), -sp.sin(phi)],
        [0, sp.sin(phi), sp.cos(phi)]
    ])
    
    R_CE = Rz * Ry * Rx
    
    # Transformation matrix for angular velocities
    T = sp.Matrix([
        [1, 0, -sp.sin(theta)],
        [0, sp.cos(phi), sp.sin(phi) * sp.cos(theta)],
        [0, -sp.sin(phi), sp.cos(phi) * sp.cos(theta)]
    ])
    
    # State and input vectors
    pos = sp.Matrix([x, y, z])
    alpha = sp.Matrix([phi, theta, psi])
    vel = sp.Matrix([xdot, ydot, zdot])
    omega = sp.Matrix([omega1, omega2, omega3])
    
    q = sp.Matrix([pos, alpha, vel, omega])
    
    I_mat = sp.Matrix([
        [I1, 0, 0],
        [0, I2, 0],
        [0, 0, I3]
    ])
    u = sp.Matrix([u1, u2, u3, u4])
    r = sp.Matrix([r1, r2, r3])
    n = sp.Matrix([n1, n2, n3])
    
    # Dynamics equations
    pos_dot = sp.Matrix([xdot, ydot, zdot])
    alpha_dot = T.inv() * omega
    vel_dot = -g * sp.Matrix([0, 0, 1]) + (1/m) * R_CE * (u1 + u2 + u3 + u4) * sp.Matrix([0, 0, 1]) + (1/m) * R_CE * r
    
    # Torque from rotors
    torque = sp.Matrix([
        (u2 - u4) * l,
        (u3 - u1) * l,
        (u1 - u2 + u3 - u4) * sigma
    ]) + n - omega.cross(I_mat * omega)
    
    omega_dot = I_mat.inv() * torque
    
    qdot = sp.Matrix([pos_dot, alpha_dot, vel_dot, omega_dot])
    
    # Equilibrium point
    z0_vals = {
        x: position[0], y: position[1], z: position[2],
        phi: 0, theta: 0, psi: 0,
        xdot: 0, ydot: 0, zdot: 0,
        omega1: 0, omega2: 0, omega3: 0
    }
    
    u0_val = m * g / 4
    u0_vals = {u1: u0_val, u2: u0_val, u3: u0_val, u4: u0_val}
    r0_vals = {r1: 0, r2: 0, r3: 0}
    
    all_subs = {**z0_vals, **u0_vals, **r0_vals}
    
    # Compute Jacobians
    Ja = qdot.jacobian(q)
    Jb = qdot.jacobian(u)
    
    # Evaluate at equilibrium
    A_sym = Ja.subs(all_subs)
    B_sym = Jb.subs(all_subs)
    
    # Convert to numpy arrays
    A = np.array(A_sym.tolist(), dtype=float)
    B = np.array(B_sym.tolist(), dtype=float)
    
    # Verify controllability
    from numpy.linalg import matrix_rank
    C = B.copy()
    for i in range(1, 12):
        C = np.hstack([C, np.linalg.matrix_power(A, i) @ B])
    
    if matrix_rank(C) != 12:
        print(f"Warning: Controllability matrix rank is {matrix_rank(C)}, not 12")
    
    return A, B


class SACController:
    """
    Sophisticated Adaptive Controller using LQR gains.
    
    Implements trajectory prediction and LQR-based tracking control.
    """
    
    def __init__(self, quadrotor, timestep: float = 0.01):
        """
        Initialize SAC controller.
        
        Args:
            quadrotor: Quadrotor object
            timestep: Simulation timestep
        """
        self.u0 = quadrotor.m * quadrotor.g / 4
        self.timestep = timestep
        
        # Linearize around hover at z=1
        position = np.array([0.0, 0.0, 1.0])
        A, B = linearize_quadrotor(quadrotor, position)
        
        # LQR gains for different modes
        # Normal tracking
        Q = np.diag([5, 5, 8, 15, 15, 8, 1.25, 1.25, 1.0, 3, 3, 1.5])
        R = 1.55 * np.eye(4)
        self.k = compute_lqr_gain(A, B, Q, R)
        
        # Fast/aggressive tracking (when close to target)
        fast_Q = np.diag([14, 14, 20, 12, 12, 4, 1.0, 1.0, 1.0, 1.5, 1.5, 1.5])
        fast_R = 0.2 * np.eye(4)
        self.fast_k = compute_lqr_gain(A, B, fast_Q, fast_R)
        
        # Return home
        home_Q = np.eye(12)
        home_R = 3 * np.eye(4)
        self.home_k = compute_lqr_gain(A, B, home_Q, home_R)
        
        # Trajectory prediction
        self.output_count = 0
        self.prev_coeffs = np.zeros((3, 8))
        self.target_time = -1
        self.jump_ahead_level = 3.6
        
        # Store control history
        self.uvec = []
    
    def output(self, is_captured: bool, z: np.ndarray, y: np.ndarray) -> np.ndarray:
        """
        Compute control output.
        
        Args:
            is_captured: Whether target has been captured
            z: Quadrotor state (12,)
            y: Target position (3,)
            
        Returns:
            u: Control inputs (4,)
        """
        kill_dist = 0  # Distance threshold for "kill mode"
        
        if not is_captured:
            # Initialize reference
            r = np.zeros(12)
            
            # Compute trajectory prediction coefficients
            coeff_size = self.prev_coeffs.shape[1]
            coeffs = self._solve_coeffs(y, coeff_size - 1)
            
            # Compute error
            error_vec = z[:3] - y
            error_mag = np.linalg.norm(error_vec)
            
            if self.output_count > 200:
                if self.target_time == -1:
                    if error_mag < kill_dist:
                        r[:3] = self._solve_poly(coeffs, 0.08)
                    else:
                        self.target_time = self.jump_ahead_level
                        self.jump_ahead_level += 0.1
                        r[:3] = self._solve_poly(coeffs, self.target_time)
                else:
                    self.target_time = max(self.target_time - self.timestep, 0)
                    if self.target_time == 0 or error_mag < (kill_dist * 0.1):
                        self.target_time = -1
                    r[:3] = self._solve_poly(coeffs, self.target_time)
            else:
                self.output_count += 1
                r[:2] = [0, 0]
                r[2] = y[2]
            
            # Select gain based on distance
            if error_mag < kill_dist:
                temp_k = self.fast_k
            else:
                temp_k = self.k
            
            # Bound target location
            r[:2] = np.clip(r[:2], -5, 5)
            r[2] = np.clip(r[2], 0, 10)
            
            u = np.full(4, self.u0) + temp_k @ (r - z)
        else:
            # Return home mode
            home = np.zeros(12)
            if not (abs(z[0]) < 0.5 and abs(z[1]) < 0.5):
                home[2] = z[2]
            else:
                home[2] = 0.2
            
            u = np.full(4, self.u0) + self.home_k @ (home - z)
        
        self.uvec.append(u.copy())
        return u
    
    def _solve_coeffs(self, y: np.ndarray, degree: int) -> np.ndarray:
        """
        Compute polynomial coefficients for trajectory prediction.
        
        Args:
            y: Current target position (3,)
            degree: Polynomial degree
            
        Returns:
            coeffs: Polynomial coefficients (3, degree+1)
        """
        threshold = 1e-11
        yder = np.zeros((3, degree + 1))
        yder[:, 0] = y
        
        for i in range(1, degree + 1):
            # MATLAB: yder(:, i) = (yder(:, i - 1) - factorial(i - 2)*prev_coeffs(:, degree + 3 - i)) / dt
            # Python indexing adjustment: i starts at 1 (not 2), so factorial(i-1) and index degree + 1 - i
            coeff_idx = min(degree + 1 - i, self.prev_coeffs.shape[1] - 1)
            yder[:, i] = (yder[:, i-1] - factorial(i-1) * self.prev_coeffs[:, coeff_idx]) / self.timestep
            # Apply threshold for numerical stability
            yder[:, i] = np.where(np.abs(yder[:, i]) < threshold, 0, yder[:, i])
        
        coeffs = np.zeros((3, degree + 1))
        for i in range(degree + 1):
            coeffs[:, i] = yder[:, degree - i] / factorial(degree - i)
        
        self.prev_coeffs = coeffs
        return coeffs
    
    def _solve_poly(self, coeffs: np.ndarray, t: float) -> np.ndarray:
        """
        Evaluate polynomial at time t.
        
        Args:
            coeffs: Polynomial coefficients (3, n)
            t: Time value
            
        Returns:
            position: Predicted position (3,)
        """
        n = coeffs.shape[1]
        time_powers = np.array([t ** (n - 1 - i) for i in range(n)])
        return coeffs @ time_powers
    
    def reset(self):
        """Reset controller state."""
        self.output_count = 0
        self.prev_coeffs = np.zeros((3, 8))
        self.target_time = -1
        self.jump_ahead_level = 3.6
        self.uvec = []


class HoverController:
    """
    Simple altitude hold controller.
    """
    
    def __init__(self, quadrotor, altitude: float = 1.0, gains: Tuple[float, float] = (1.0, 0.5)):
        """
        Initialize hover controller.
        
        Args:
            quadrotor: Quadrotor object
            altitude: Target altitude [m]
            gains: (position gain, velocity gain)
        """
        self.u0 = quadrotor.m * quadrotor.g / 4
        self.altitude = altitude
        self.k = np.array(gains)
        self.uvec = []
    
    def output(self, is_captured: bool, z: np.ndarray, y: np.ndarray = None) -> np.ndarray:
        """
        Compute control output for hover.
        
        Args:
            is_captured: Not used
            z: Quadrotor state (12,)
            y: Not used
            
        Returns:
            u: Control inputs (4,)
        """
        # Simple altitude PD control
        altitude_error = self.altitude - z[2]
        velocity_error = -z[8]
        
        delta_u = self.k[0] * altitude_error + self.k[1] * velocity_error
        u = np.full(4, self.u0 + delta_u)
        
        self.uvec.append(u.copy())
        return u
    
    def reset(self):
        """Reset controller state."""
        self.uvec = []

