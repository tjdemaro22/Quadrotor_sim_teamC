"""
Simulator module for quadrotor tracking simulation.
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from mpl_toolkits.mplot3d import Axes3D
from typing import Tuple, Optional, Dict, List
import time


class Simulator:
    """
    Main simulation class that coordinates quadrotor, controller, and target UAV.
    """
    
    def __init__(self, quadrotor, controller, uav):
        """
        Initialize simulator.
        
        Args:
            quadrotor: Quadrotor object
            controller: Controller object with output() method
            uav: UAV/target object
        """
        self.quadrotor = quadrotor
        self.controller = controller
        self.uav = uav
        
        # Simulation parameters
        self.simtime = (0, 10)  # (start, end) time
        self.timestep = 0.01
        self.epsilon = 0.1  # Capture distance
        
        # Visualization parameters
        self.airspace_box_length = 4
        self._ax = None
        self._fig = None
    
    def simulate(self, z0: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """
        Run the simulation.
        
        Args:
            z0: Initial quadrotor state (12,)
            
        Returns:
            t: Time array
            z: Quadrotor states (n_times, 12)
            u: Control inputs (n_times, 4)
            d: Disturbances (n_times, 6)
            y: UAV positions (n_times, 3)
        """
        ts = np.arange(self.simtime[0], self.simtime[1] + self.timestep, self.timestep)
        
        # Storage
        t_list = []
        z_list = []
        u_list = []
        d_list = []
        y_list = []
        
        is_captured = False
        z_current = z0.copy()
        
        for k in range(len(ts) - 1):
            tspan = (ts[k], ts[k + 1])
            t_, z_, u_, d_, y_ = self._step(is_captured, tspan, z_current)
            
            if k == 0:
                t_list.extend([t_[0], t_[-1]])
                z_list.extend([z_[0], z_[-1]])
                u_list.extend([u_[0], u_[-1]])
                d_list.extend([d_[0], d_[-1]])
                y_list.extend([y_[0], y_[-1]])
            else:
                t_list.append(t_[-1])
                z_list.append(z_[-1])
                u_list.append(u_[-1])
                d_list.append(d_[-1])
                y_list.append(y_[-1])
            
            z_current = z_[-1]
            
            # Check capture condition
            distances = np.linalg.norm(z_[:, :3] - y_, axis=1)
            if np.min(distances) < self.epsilon:
                is_captured = True
        
        # Convert to arrays
        t = np.array(t_list)
        z = np.array(z_list)
        u = np.array(u_list)
        d = np.array(d_list)
        y = np.array(y_list)
        
        return t, z, u, d, y
    
    def _step(
        self,
        is_captured: bool,
        tspan: Tuple[float, float],
        z0: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """
        Perform one simulation step.
        
        Args:
            is_captured: Whether target is captured
            tspan: Time span for this step
            z0: Initial state
            
        Returns:
            t, z, u, d, y arrays for this step
        """
        # Get target location
        y0 = self.uav.location(is_captured, tspan[0], z0)
        
        # Compute control
        u = self.controller.output(is_captured, z0, y0)
        
        # Get disturbances
        dist = self.uav.disturbance(is_captured)
        
        # Solve dynamics
        t, z = self.quadrotor.solve(tspan, z0, lambda t, z: u, dist)
        
        # Compute disturbance values at solution times
        d = np.zeros((len(t), 6))
        for i, ti in enumerate(t):
            d[i, :3] = dist['r'](ti, z[i])
            d[i, 3:] = dist['n'](ti, z[i])
        
        # Get UAV locations at solution times
        y = self.uav.location(is_captured, t, z)
        
        # Tile control input
        u_arr = np.tile(u, (len(t), 1))
        
        return t, z, u_arr, d, y
    
    def build_axes(self, figsize: Tuple[int, int] = (12, 10)) -> plt.Axes:
        """
        Create and configure 3D axes for animation.
        
        Args:
            figsize: Figure size (width, height)
            
        Returns:
            3D axes object
        """
        self._fig = plt.figure(figsize=figsize)
        self._ax = self._fig.add_subplot(111, projection='3d')
        
        L = self.airspace_box_length
        self._ax.set_xlim([-L/2, L/2])
        self._ax.set_ylim([-L/2, L/2])
        self._ax.set_zlim([0, L])
        
        self._ax.set_xlabel('X (m)', fontsize=12)
        self._ax.set_ylabel('Y (m)', fontsize=12)
        self._ax.set_zlabel('Z (m)', fontsize=12)
        self._ax.set_title('Quadrotor Tracking Simulation', fontsize=14)
        
        # Add grid
        self._ax.grid(True, alpha=0.3)
        
        return self._ax
    
    def build_visuals(self, ax: plt.Axes = None):
        """
        Initialize visualization for quadrotor and UAV.
        
        Args:
            ax: 3D axes (creates new if None)
        """
        if ax is None:
            ax = self.build_axes()
        
        self.quadrotor.draw(ax, color='#1f77b4', trail_color='#1f77b4')  # Blue
        self.uav.draw(ax, color='#d62728', trail_color='#d62728')  # Red
    
    def animate(
        self,
        t: np.ndarray,
        z: np.ndarray,
        y: np.ndarray,
        ax: plt.Axes = None,
        interval: int = 20,
        save_path: str = None,
        show: bool = True
    ) -> Optional[FuncAnimation]:
        """
        Create animation of the simulation.
        
        Args:
            t: Time array
            z: Quadrotor states (n_times, 12)
            y: UAV positions (n_times, 3)
            ax: 3D axes (creates new if None)
            interval: Animation interval in ms
            save_path: Path to save animation (mp4, gif)
            show: Whether to display animation
            
        Returns:
            FuncAnimation object
        """
        if ax is None:
            ax = self.build_axes()
        
        self.build_visuals(ax)
        
        # Clear any existing trails
        self.quadrotor.clear_trail()
        self.uav.clear_trail()
        
        # Time text
        time_text = ax.text2D(0.02, 0.98, '', transform=ax.transAxes, fontsize=12,
                              verticalalignment='top', bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
        
        def init():
            self.quadrotor.show(z[0], update_trail=False)
            self.uav.show(y[0], update_trail=False)
            time_text.set_text(f'Time: {t[0]:.2f} s')
            return []
        
        def update(frame):
            self.quadrotor.show(z[frame])
            self.uav.show(y[frame])
            time_text.set_text(f'Time: {t[frame]:.2f} s')
            return []
        
        anim = FuncAnimation(
            self._fig, update,
            frames=len(t),
            init_func=init,
            interval=interval,
            blit=False,
            repeat=False
        )
        
        if save_path is not None:
            print(f"Saving animation to {save_path}...")
            if save_path.endswith('.gif'):
                anim.save(save_path, writer='pillow', fps=30)
            else:
                anim.save(save_path, writer='ffmpeg', fps=30)
            print("Animation saved!")
        
        if show:
            plt.show()
        
        return anim
    
    def plot_static_3d(
        self,
        z: np.ndarray,
        y: np.ndarray,
        ax: plt.Axes = None,
        show: bool = True
    ) -> plt.Figure:
        """
        Create static 3D plot showing trajectories.
        
        Args:
            z: Quadrotor states (n_times, 12)
            y: UAV positions (n_times, 3)
            ax: 3D axes (creates new if None)
            show: Whether to display plot
            
        Returns:
            Figure object
        """
        if ax is None:
            fig = plt.figure(figsize=(12, 10))
            ax = fig.add_subplot(111, projection='3d')
        else:
            fig = ax.figure
        
        L = self.airspace_box_length
        ax.set_xlim([-L/2, L/2])
        ax.set_ylim([-L/2, L/2])
        ax.set_zlim([0, L])
        
        # Plot quadrotor trajectory (blue)
        ax.plot(z[:, 0], z[:, 1], z[:, 2], 'b-', linewidth=2, label='Quadrotor', alpha=0.8)
        ax.scatter(z[0, 0], z[0, 1], z[0, 2], c='blue', s=100, marker='o', label='Quad Start')
        ax.scatter(z[-1, 0], z[-1, 1], z[-1, 2], c='blue', s=100, marker='s', label='Quad End')
        
        # Plot UAV trajectory (red)
        ax.plot(y[:, 0], y[:, 1], y[:, 2], 'r-', linewidth=2, label='Target UAV', alpha=0.8)
        ax.scatter(y[0, 0], y[0, 1], y[0, 2], c='red', s=100, marker='o', label='UAV Start')
        ax.scatter(y[-1, 0], y[-1, 1], y[-1, 2], c='red', s=100, marker='s', label='UAV End')
        
        ax.set_xlabel('X (m)', fontsize=12)
        ax.set_ylabel('Y (m)', fontsize=12)
        ax.set_zlabel('Z (m)', fontsize=12)
        ax.set_title('Quadrotor Tracking Simulation - 3D Trajectories', fontsize=14)
        ax.legend(loc='upper left')
        ax.grid(True, alpha=0.3)
        
        if show:
            plt.show()
        
        return fig

