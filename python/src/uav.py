"""
UAV (Intruder/Target) module for quadrotor tracking simulation.
"""

import numpy as np
from typing import Callable, Dict, Optional
import matplotlib.pyplot as plt


class UAV:
    """
    UAV/Intruder class representing the target to be tracked.
    """
    
    def __init__(
        self,
        movement_fn: Callable[[float], np.ndarray],
        disturbance_fn: Dict = None
    ):
        """
        Initialize UAV.
        
        Args:
            movement_fn: Function that returns position [x, y, z] given time t
            disturbance_fn: Dictionary with 'r' and 'n' functions for disturbances
                           when quadrotor captures UAV
        """
        self.movement_fn = movement_fn
        
        if disturbance_fn is None:
            self.disturbance_fn = {
                'r': lambda t, z: np.zeros(3),
                'n': lambda t, z: np.zeros(3)
            }
        else:
            self.disturbance_fn = disturbance_fn
        
        # Visualization handles
        self._ax = None
        self._body_marker = None
        self._silhouette_line = None
        self._trail_line = None
        
        # Default colors
        self.body_color = '#d62728'  # Red
        self.silhouette_color = '#808080'
        self.trail_color = '#d62728'  # Red trail
        self.body_size = 80  # marker size
    
    def location(
        self,
        is_captured: bool,
        t: float | np.ndarray,
        z: np.ndarray = None
    ) -> np.ndarray:
        """
        Get UAV location at time t.
        
        Args:
            is_captured: Whether the UAV has been captured
            t: Time (scalar or array)
            z: Quadrotor state (used when captured to track quadrotor)
            
        Returns:
            Position array (3,) or (n_times, 3)
        """
        if is_captured:
            if z is None:
                raise ValueError("z must be provided when UAV is captured")
            # When captured, UAV moves with quadrotor
            if z.ndim == 1:
                return z[:3]
            else:
                return z[:, :3]
        else:
            if np.isscalar(t):
                return np.array(self.movement_fn(t)).flatten()
            else:
                positions = np.zeros((len(t), 3))
                for i, ti in enumerate(t):
                    positions[i] = np.array(self.movement_fn(ti)).flatten()
                return positions
    
    def disturbance(self, is_captured: bool) -> Dict:
        """
        Get disturbance functions.
        
        Args:
            is_captured: Whether the UAV has been captured
            
        Returns:
            Dictionary with 'r' and 'n' disturbance functions
        """
        if is_captured:
            return self.disturbance_fn
        else:
            return {
                'r': lambda t, z: np.zeros(3),
                'n': lambda t, z: np.zeros(3)
            }
    
    def draw(self, ax: plt.Axes = None, color: str = None, trail_color: str = None):
        """
        Initialize visualization handles.
        
        Args:
            ax: Matplotlib 3D axes
            color: Body color
            trail_color: Trail color
        """
        if ax is None:
            fig = plt.figure(figsize=(10, 8))
            ax = fig.add_subplot(111, projection='3d')
        
        self._ax = ax
        
        if color is not None:
            self.body_color = color
        if trail_color is not None:
            self.trail_color = trail_color
        
        # Silhouette
        self._silhouette_line, = ax.plot([], [], [], '--', color=self.silhouette_color, linewidth=1)
        
        # Body marker
        self._body_marker = ax.scatter([], [], [], c=self.body_color, s=self.body_size, marker='o')
        
        # Trail
        self._trail_line, = ax.plot([], [], [], '-', color=self.trail_color, linewidth=1.5, alpha=0.7)
        self._trail_data = {'x': [], 'y': [], 'z': []}
    
    def show(self, y: np.ndarray, update_trail: bool = True):
        """
        Update visualization for current position.
        
        Args:
            y: Current position (3,) or (1, 3)
        """
        y = np.ravel(y)
        
        if self._ax is None:
            self.draw()
        
        # Update body marker
        self._body_marker._offsets3d = ([y[0]], [y[1]], [y[2]])
        
        # Update silhouette
        sil_x = [0, y[0], y[0], y[0]]
        sil_y = [0, 0, y[1], y[1]]
        sil_z = [0, 0, 0, y[2]]
        self._silhouette_line.set_data(sil_x, sil_y)
        self._silhouette_line.set_3d_properties(sil_z)
        
        # Update trail
        if update_trail:
            self._trail_data['x'].append(y[0])
            self._trail_data['y'].append(y[1])
            self._trail_data['z'].append(y[2])
            self._trail_line.set_data(self._trail_data['x'], self._trail_data['y'])
            self._trail_line.set_3d_properties(self._trail_data['z'])
    
    def clear_trail(self):
        """Clear the trail data."""
        self._trail_data = {'x': [], 'y': [], 'z': []}
        if self._trail_line is not None:
            self._trail_line.set_data([], [])
            self._trail_line.set_3d_properties([])

