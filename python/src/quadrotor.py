"""
Quadrotor dynamics and visualization module.

State vector definition:
    z[0:3] = [x, y, z] : quadrotor center in world frame
    z[3:6] = [phi, theta, psi] : roll, pitch and yaw  
    z[6:9] = [vx, vy, vz] : quadrotor velocity in world frame
    z[9:12] = [w1, w2, w3] : angular velocity in quadrotor frame

Rotor layout (top view):
         O u1
         |
   u2 O--|--O u4
         |
         O u3
"""

import numpy as np
from scipy.integrate import solve_ivp
from typing import Callable, Optional, Tuple, Dict
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D


def rotx(phi: float) -> np.ndarray:
    """Rotation matrix about x-axis."""
    return np.array([
        [1, 0, 0],
        [0, np.cos(phi), -np.sin(phi)],
        [0, np.sin(phi), np.cos(phi)]
    ])


def roty(theta: float) -> np.ndarray:
    """Rotation matrix about y-axis."""
    return np.array([
        [np.cos(theta), 0, np.sin(theta)],
        [0, 1, 0],
        [-np.sin(theta), 0, np.cos(theta)]
    ])


def rotz(psi: float) -> np.ndarray:
    """Rotation matrix about z-axis."""
    return np.array([
        [np.cos(psi), -np.sin(psi), 0],
        [np.sin(psi), np.cos(psi), 0],
        [0, 0, 1]
    ])


class Quadrotor:
    """
    Quadrotor dynamics and visualization class.
    """
    
    def __init__(
        self,
        g: float = 9.81,
        l: float = 0.2,
        m: float = 0.5,
        I: np.ndarray = None,
        mu: float = np.inf,
        sigma: float = 0.01
    ):
        """
        Initialize quadrotor parameters.
        
        Args:
            g: Gravitational acceleration [m/s^2]
            l: Distance from center of mass to each rotor [m]
            m: Total mass [kg]
            I: Mass moment of inertia matrix [kg*m^2]
            mu: Maximum thrust of each rotor [N]
            sigma: Proportionality constant relating thrust to torque [m]
        """
        self.g = g
        self.l = l
        self.m = m
        self.I = I if I is not None else np.diag([1.0, 1.0, 2.0])
        self.mu = mu
        self.sigma = sigma
        self.sigma_div_l = sigma / l
        
        # Precompute rotor visualization geometry
        N = 10
        Q = np.linspace(0, 2 * np.pi, N)
        rotor_scale = 0.3
        self.rotor_points = rotor_scale * l * np.column_stack([np.cos(Q), np.sin(Q), np.zeros(N)])
        self.chassis_points = l * np.array([
            [1, 0, 0],
            [0, 1, 0],
            [-1, 0, 0],
            [0, -1, 0]
        ])
        
        # Visualization handles
        self._ax = None
        self._rotor_lines = []
        self._chassis_line = None
        self._silhouette_line = None
        self._trail_line = None
        
        # Default colors
        self.body_color = '#1f77b4'  # Blue
        self.silhouette_color = '#808080'
        self.trail_color = '#1f77b4'  # Blue trail
    
    def _rotation_matrix(self, phi: float, theta: float, psi: float) -> np.ndarray:
        """Compute rotation matrix from body frame to world frame (ZYX Euler)."""
        return rotz(psi) @ roty(theta) @ rotx(phi)
    
    def _T_inv(self, phi: float, theta: float) -> np.ndarray:
        """Transform angular velocities in body frame to Euler angle rates."""
        return np.array([
            [1, np.sin(phi) * np.tan(theta), np.cos(phi) * np.tan(theta)],
            [0, np.cos(phi), -np.sin(phi)],
            [0, np.sin(phi) / np.cos(theta), np.cos(phi) / np.cos(theta)]
        ])
    
    def _saturate(self, u: np.ndarray) -> np.ndarray:
        """Apply saturation limits to control inputs."""
        return np.clip(u, 0, self.mu)
    
    def dynamics(
        self,
        z: np.ndarray,
        u: np.ndarray,
        r: np.ndarray = None,
        n: np.ndarray = None
    ) -> np.ndarray:
        """
        Compute state derivatives.
        
        Args:
            z: State vector (12,)
            u: Control inputs (4,) - thrust from each rotor
            r: External force disturbance in body frame (3,)
            n: External torque disturbance in body frame (3,)
            
        Returns:
            dz: State derivatives (12,)
        """
        if r is None:
            r = np.zeros(3)
        if n is None:
            n = np.zeros(3)
        
        # Extract states
        phi, theta, psi = z[3:6]
        v = z[6:9]
        omega = z[9:12]
        
        # Rotation matrix from body to world
        R = self._rotation_matrix(phi, theta, psi)
        
        # Torque vector induced by rotor thrusts
        rt = self.l * np.array([
            u[1] - u[3],  # Roll torque
            u[2] - u[0],  # Pitch torque
            (u[0] - u[1] + u[2] - u[3]) * self.sigma_div_l  # Yaw torque
        ])
        
        # State derivatives
        dz = np.zeros(12)
        
        # Position derivative = velocity
        dz[0:3] = v
        
        # Euler angle derivatives
        dz[3:6] = self._T_inv(phi, theta) @ omega
        
        # Velocity derivative (translational dynamics)
        total_thrust = np.sum(u)
        dz[6:9] = R @ np.array([r[0], r[1], r[2] + total_thrust]) / self.m - np.array([0, 0, self.g])
        
        # Angular velocity derivative (rotational dynamics)
        dz[9:12] = np.linalg.solve(self.I, rt + n - np.cross(omega, self.I @ omega))
        
        return dz
    
    def state_space(self, z: np.ndarray, u: np.ndarray) -> np.ndarray:
        """Compute state derivatives without disturbances."""
        return self.dynamics(z, u, np.zeros(3), np.zeros(3))
    
    def solve(
        self,
        tspan: Tuple[float, float],
        z0: np.ndarray,
        control: Callable[[float, np.ndarray], np.ndarray] = None,
        dist: Dict = None
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Solve the initial value problem.
        
        Args:
            tspan: Time span [t0, tf]
            z0: Initial state (12,)
            control: Control function u = control(t, z)
            dist: Disturbance dictionary with 'r' and 'n' functions
            
        Returns:
            t: Time array
            z: State array (n_times, 12)
        """
        if control is None:
            control = lambda t, z: np.zeros(4)
        
        if dist is None:
            dist = {
                'r': lambda t, z: np.zeros(3),
                'n': lambda t, z: np.zeros(3)
            }
        
        def odefun(t, z):
            u = self._saturate(control(t, z))
            r = dist['r'](t, z)
            n = dist['n'](t, z)
            return self.dynamics(z, u, r, n)
        
        sol = solve_ivp(odefun, tspan, z0, method='RK45', dense_output=True)
        return sol.t, sol.y.T
    
    def draw(self, ax: plt.Axes = None, color: str = None, trail_color: str = None):
        """
        Initialize visualization handles on the given axes.
        
        Args:
            ax: Matplotlib 3D axes
            color: Body color
            trail_color: Trail color
        """
        if ax is None:
            fig = plt.figure(figsize=(10, 8))
            ax = fig.add_subplot(111, projection='3d')
            ax.set_xlim([-2, 2])
            ax.set_ylim([-2, 2])
            ax.set_zlim([0, 4])
            ax.set_xlabel('X')
            ax.set_ylabel('Y')
            ax.set_zlabel('Z')
        
        self._ax = ax
        
        if color is not None:
            self.body_color = color
        if trail_color is not None:
            self.trail_color = trail_color
        
        # Silhouette (projection lines)
        self._silhouette_line, = ax.plot([], [], [], '--', color=self.silhouette_color, linewidth=1)
        
        # Chassis
        self._chassis_line, = ax.plot([], [], [], color=self.body_color, linewidth=2)
        
        # Rotors
        self._rotor_lines = []
        for _ in range(4):
            line, = ax.plot([], [], [], color=self.body_color, linewidth=2)
            self._rotor_lines.append(line)
        
        # Trail
        self._trail_line, = ax.plot([], [], [], '-', color=self.trail_color, linewidth=1.5, alpha=0.7)
        self._trail_data = {'x': [], 'y': [], 'z': []}
    
    def show(self, z: np.ndarray, update_trail: bool = True):
        """
        Update visualization for current state.
        
        Args:
            z: Current state vector (12,) or (1, 12)
        """
        z = np.ravel(z)
        
        if self._ax is None:
            self.draw()
        
        # Extract position and orientation
        pos = z[0:3]
        phi, theta, psi = z[3:6]
        
        # Rotation matrix
        R = self._rotation_matrix(phi, theta, psi)
        
        # Transform chassis points to world frame
        centers = pos + self.chassis_points @ R.T
        
        # Transform rotor points
        rotor_world = self.rotor_points @ R.T
        
        # Update rotors
        for i, line in enumerate(self._rotor_lines):
            rotor_pos = rotor_world + centers[i]
            line.set_data(rotor_pos[:, 0], rotor_pos[:, 1])
            line.set_3d_properties(rotor_pos[:, 2])
        
        # Update chassis (X shape)
        chassis_x = [centers[0, 0], centers[2, 0], np.nan, centers[1, 0], centers[3, 0]]
        chassis_y = [centers[0, 1], centers[2, 1], np.nan, centers[1, 1], centers[3, 1]]
        chassis_z = [centers[0, 2], centers[2, 2], np.nan, centers[1, 2], centers[3, 2]]
        self._chassis_line.set_data(chassis_x, chassis_y)
        self._chassis_line.set_3d_properties(chassis_z)
        
        # Update silhouette
        sil_x = [0, pos[0], pos[0], pos[0]]
        sil_y = [0, 0, pos[1], pos[1]]
        sil_z = [0, 0, 0, pos[2]]
        self._silhouette_line.set_data(sil_x, sil_y)
        self._silhouette_line.set_3d_properties(sil_z)
        
        # Update trail
        if update_trail:
            self._trail_data['x'].append(pos[0])
            self._trail_data['y'].append(pos[1])
            self._trail_data['z'].append(pos[2])
            self._trail_line.set_data(self._trail_data['x'], self._trail_data['y'])
            self._trail_line.set_3d_properties(self._trail_data['z'])
    
    def clear_trail(self):
        """Clear the trail data."""
        self._trail_data = {'x': [], 'y': [], 'z': []}
        if self._trail_line is not None:
            self._trail_line.set_data([], [])
            self._trail_line.set_3d_properties([])
    
    def plot_states(self, t: np.ndarray, z: np.ndarray, fig: plt.Figure = None):
        """
        Plot state history.
        
        Args:
            t: Time array
            z: State array (n_times, 12)
            fig: Optional figure to plot on
        """
        if fig is None:
            fig = plt.figure(figsize=(12, 10))
        
        # Position
        ax1 = fig.add_subplot(2, 2, 1)
        ax1.plot(t, z[:, 0], label='x')
        ax1.plot(t, z[:, 1], label='y')
        ax1.plot(t, z[:, 2], label='z')
        ax1.set_xlabel('Time (s)')
        ax1.set_ylabel('Position (m)')
        ax1.set_title('Position')
        ax1.legend()
        ax1.grid(True)
        
        # Velocity
        ax2 = fig.add_subplot(2, 2, 2)
        ax2.plot(t, z[:, 6], label='vx')
        ax2.plot(t, z[:, 7], label='vy')
        ax2.plot(t, z[:, 8], label='vz')
        ax2.set_xlabel('Time (s)')
        ax2.set_ylabel('Velocity (m/s)')
        ax2.set_title('Velocity')
        ax2.legend()
        ax2.grid(True)
        
        # Orientation
        ax3 = fig.add_subplot(2, 2, 3)
        ax3.plot(t, np.rad2deg(z[:, 3]), label='φ (roll)')
        ax3.plot(t, np.rad2deg(z[:, 4]), label='θ (pitch)')
        ax3.plot(t, np.rad2deg(z[:, 5]), label='ψ (yaw)')
        ax3.set_xlabel('Time (s)')
        ax3.set_ylabel('Angle (deg)')
        ax3.set_title('Orientation')
        ax3.legend()
        ax3.grid(True)
        
        # Angular velocity
        ax4 = fig.add_subplot(2, 2, 4)
        ax4.plot(t, z[:, 9], label='ω1')
        ax4.plot(t, z[:, 10], label='ω2')
        ax4.plot(t, z[:, 11], label='ω3')
        ax4.set_xlabel('Time (s)')
        ax4.set_ylabel('Angular Velocity (rad/s)')
        ax4.set_title('Angular Velocity')
        ax4.legend()
        ax4.grid(True)
        
        plt.tight_layout()
        return fig

