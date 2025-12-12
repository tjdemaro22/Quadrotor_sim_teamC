#!/usr/bin/env python3
"""
HTML Report Generator for Quadrotor Simulation

Generates an interactive HTML report with Plotly 3D visualization
and state/control plots, including animated simulation playback.
"""

import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import os
from datetime import datetime
from typing import Optional, Tuple, List


# =============================================================================
# Quadrotor Geometry Helper Functions
# =============================================================================

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


def rotation_matrix(phi: float, theta: float, psi: float) -> np.ndarray:
    """Compute rotation matrix from body frame to world frame (ZYX Euler)."""
    return rotz(psi) @ roty(theta) @ rotx(phi)


def generate_rotor_points(l: float, rotor_scale: float = 0.3, n_points: int = 20) -> np.ndarray:
    """
    Generate points for a single rotor circle in the body frame.
    
    Args:
        l: Arm length from center to rotor
        rotor_scale: Scale factor for rotor radius relative to arm length
        n_points: Number of points to generate for the circle
        
    Returns:
        Array of shape (n_points, 3) with rotor circle points
    """
    Q = np.linspace(0, 2 * np.pi, n_points)
    radius = rotor_scale * l
    return np.column_stack([radius * np.cos(Q), radius * np.sin(Q), np.zeros(n_points)])


def generate_chassis_points(l: float) -> np.ndarray:
    """
    Generate the 4 rotor center positions in the body frame.
    
    Rotor layout (top view):
             O u1 (rotor 0, +x)
             |
       u2 O--|--O u4 (rotor 1: +y, rotor 3: -y)
             |
             O u3 (rotor 2, -x)
    
    Args:
        l: Arm length from center to each rotor
        
    Returns:
        Array of shape (4, 3) with rotor center positions
    """
    return l * np.array([
        [1, 0, 0],   # Rotor 0 (u1): front (+x)
        [0, 1, 0],   # Rotor 1 (u2): left (+y)
        [-1, 0, 0],  # Rotor 2 (u3): back (-x)
        [0, -1, 0]   # Rotor 3 (u4): right (-y)
    ])


def get_quadrotor_geometry(
    pos: np.ndarray,
    phi: float,
    theta: float,
    psi: float,
    l: float,
    rotor_scale: float = 0.3
) -> Tuple[List[np.ndarray], np.ndarray, np.ndarray]:
    """
    Compute world-frame quadrotor geometry for visualization.
    
    Args:
        pos: Position [x, y, z] in world frame
        phi, theta, psi: Euler angles (roll, pitch, yaw)
        l: Arm length
        rotor_scale: Scale factor for rotor radius
        
    Returns:
        rotors_world: List of 4 arrays, each (n_points, 3) for rotor circles
        chassis_world: Array of shape (4, 3) with rotor center positions
        thrust_dir: Array of shape (3,) with thrust direction (body +z in world frame)
    """
    R = rotation_matrix(phi, theta, psi)
    
    # Generate base geometry
    rotor_points = generate_rotor_points(l, rotor_scale)
    chassis_points = generate_chassis_points(l)
    
    # Transform chassis points to world frame
    chassis_world = pos + chassis_points @ R.T
    
    # Transform rotor circles to world frame (centered at each rotor position)
    rotor_world = rotor_points @ R.T
    rotors_world = [rotor_world + chassis_world[i] for i in range(4)]
    
    # Thrust direction is body +z transformed to world frame
    thrust_dir = R @ np.array([0, 0, 1])
    
    return rotors_world, chassis_world, thrust_dir


def create_quadrotor_traces(
    pos: np.ndarray,
    phi: float,
    theta: float,
    psi: float,
    l: float,
    color: str = '#1f77b4',
    rotor_scale: float = 0.3
) -> List[go.Scatter3d]:
    """
    Create Plotly traces for quadrotor visualization.
    
    Args:
        pos: Position [x, y, z]
        phi, theta, psi: Euler angles
        l: Arm length
        color: Color for the quadrotor
        rotor_scale: Scale factor for rotor radius
        
    Returns:
        List of Scatter3d traces (4 rotors + 1 chassis)
    """
    rotors, chassis, _ = get_quadrotor_geometry(pos, phi, theta, psi, l, rotor_scale)
    
    traces = []
    
    # Add rotor traces
    for i, rotor in enumerate(rotors):
        traces.append(go.Scatter3d(
            x=rotor[:, 0].tolist(),
            y=rotor[:, 1].tolist(),
            z=rotor[:, 2].tolist(),
            mode='lines',
            line=dict(color=color, width=4),
            showlegend=False,
            hoverinfo='skip'
        ))
    
    # Add chassis trace (X-shape connecting opposite rotors)
    # Connect rotor 0-2 (front-back) and rotor 1-3 (left-right) with NaN separator
    chassis_x = [chassis[0, 0], chassis[2, 0], None, chassis[1, 0], chassis[3, 0]]
    chassis_y = [chassis[0, 1], chassis[2, 1], None, chassis[1, 1], chassis[3, 1]]
    chassis_z = [chassis[0, 2], chassis[2, 2], None, chassis[1, 2], chassis[3, 2]]
    
    traces.append(go.Scatter3d(
        x=chassis_x,
        y=chassis_y,
        z=chassis_z,
        mode='lines',
        line=dict(color=color, width=5),
        showlegend=False,
        hoverinfo='skip'
    ))
    
    return traces


def create_thrust_arrow_traces(
    pos: np.ndarray,
    phi: float,
    theta: float,
    psi: float,
    thrusts: np.ndarray,
    l: float,
    thrust_scale: float = 0.1,
    rotor_scale: float = 0.3
) -> List[go.Cone]:
    """
    Create Plotly Cone traces for thrust arrow visualization.
    
    Args:
        pos: Position [x, y, z]
        phi, theta, psi: Euler angles
        thrusts: Array of 4 thrust values [u1, u2, u3, u4]
        l: Arm length
        thrust_scale: Scale factor to convert thrust to arrow length
        rotor_scale: Scale factor for rotor radius
        
    Returns:
        List of Cone traces (one per rotor)
    """
    _, chassis, thrust_dir = get_quadrotor_geometry(pos, phi, theta, psi, l, rotor_scale)
    
    # Colors for each rotor (matching the control input plot)
    rotor_colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728']
    rotor_names = ['u1 (Front)', 'u2 (Left)', 'u3 (Back)', 'u4 (Right)']
    
    traces = []
    for i in range(4):
        thrust_mag = thrusts[i] * thrust_scale
        
        # Arrow points from rotor center in thrust direction
        traces.append(go.Cone(
            x=[chassis[i, 0]],
            y=[chassis[i, 1]],
            z=[chassis[i, 2]],
            u=[thrust_dir[0] * thrust_mag],
            v=[thrust_dir[1] * thrust_mag],
            w=[thrust_dir[2] * thrust_mag],
            colorscale=[[0, rotor_colors[i]], [1, rotor_colors[i]]],
            showscale=False,
            sizemode='absolute',
            sizeref=0.15,
            anchor='tail',
            name=rotor_names[i],
            showlegend=False,
            hovertemplate=f'<b>{rotor_names[i]}</b><br>Thrust: {thrusts[i]:.2f} N<extra></extra>'
        ))
    
    return traces


def generate_html_report(
    t: np.ndarray,
    z: np.ndarray,
    u: np.ndarray,
    d: np.ndarray,
    y: np.ndarray,
    quadrotor,
    trajectory_name: str = "simulation",
    output_path: str = None,
    sim_bounds: tuple = ((-2, 2), (-2, 2), (0, 4)),
    axis_padding: float = 2.0
) -> str:
    """
    Generate an interactive HTML report of the simulation.
    
    Args:
        t: Time array
        z: Quadrotor states (n_times, 12)
        u: Control inputs (n_times, 4)
        d: Disturbances (n_times, 6)
        y: UAV positions (n_times, 3)
        quadrotor: Quadrotor object (for parameters)
        trajectory_name: Name of the trajectory used
        output_path: Output file path (default: simulation_report.html)
        sim_bounds: Simulation bounds ((x_min, x_max), (y_min, y_max), (z_min, z_max))
        axis_padding: Extra units to add to each axis beyond sim bounds
        
    Returns:
        Path to generated HTML file
    """
    if output_path is None:
        output_path = os.path.join(os.path.dirname(__file__), 'simulation_report.html')
    
    # Convert numpy arrays to lists for Plotly compatibility
    t_list = t.tolist() if isinstance(t, np.ndarray) else list(t)
    
    # Quadrotor position data
    quad_x = z[:, 0].tolist() if isinstance(z, np.ndarray) else [zi[0] for zi in z]
    quad_y = z[:, 1].tolist() if isinstance(z, np.ndarray) else [zi[1] for zi in z]
    quad_z = z[:, 2].tolist() if isinstance(z, np.ndarray) else [zi[2] for zi in z]
    
    # UAV position data
    uav_x = y[:, 0].tolist() if isinstance(y, np.ndarray) else [yi[0] for yi in y]
    uav_y = y[:, 1].tolist() if isinstance(y, np.ndarray) else [yi[1] for yi in y]
    uav_z = y[:, 2].tolist() if isinstance(y, np.ndarray) else [yi[2] for yi in y]
    
    # Control input data
    u1 = u[:, 0].tolist() if isinstance(u, np.ndarray) else [ui[0] for ui in u]
    u2 = u[:, 1].tolist() if isinstance(u, np.ndarray) else [ui[1] for ui in u]
    u3 = u[:, 2].tolist() if isinstance(u, np.ndarray) else [ui[2] for ui in u]
    u4 = u[:, 3].tolist() if isinstance(u, np.ndarray) else [ui[3] for ui in u]
    
    # Calculate axis limits with padding
    x_range = [sim_bounds[0][0] - axis_padding, sim_bounds[0][1] + axis_padding]
    y_range = [sim_bounds[1][0] - axis_padding, sim_bounds[1][1] + axis_padding]
    z_range = [sim_bounds[2][0], sim_bounds[2][1] + axis_padding]  # Only pad top of z
    
    # Calculate distance from quadrotor to UAV (2-norm)
    distances = np.linalg.norm(z[:, :3] - y, axis=1)
    distances_list = distances.tolist()
    min_dist = float(np.min(distances))
    min_dist_idx = int(np.argmin(distances))
    capture_time = float(t[min_dist_idx]) if min_dist < 0.1 else None
    
    # Calculate component-wise distances (x, y, z separately)
    dist_x = (z[:, 0] - y[:, 0]).tolist()  # quadrotor_x - uav_x
    dist_y = (z[:, 1] - y[:, 1]).tolist()  # quadrotor_y - uav_y
    dist_z = (z[:, 2] - y[:, 2]).tolist()  # quadrotor_z - uav_z
    
    # =========================================================================
    # Create Animated 3D Plot (user-controlled playback)
    # =========================================================================
    
    # Subsample for animation performance (aim for ~200 frames)
    n_frames = min(200, len(t))
    frame_indices = np.linspace(0, len(t) - 1, n_frames, dtype=int)
    
    # Get quadrotor parameters for visualization
    arm_length = quadrotor.l
    rotor_scale = 0.3  # Match MATLAB visualization
    quad_color = '#1f77b4'
    
    # Get initial state for quadrotor shape
    init_pos = np.array([quad_x[0], quad_y[0], quad_z[0]])
    init_phi, init_theta, init_psi = z[0, 3], z[0, 4], z[0, 5]
    
    # Create base figure with initial positions
    fig_anim = go.Figure()
    
    # Trace 0: Quadrotor trail (will be updated in frames)
    fig_anim.add_trace(go.Scatter3d(
        x=quad_x[:1], y=quad_y[:1], z=quad_z[:1],
        mode='lines',
        name='Quadrotor Trail',
        line=dict(color=quad_color, width=4),
        showlegend=False
    ))
    
    # Traces 1-5: Quadrotor shape (4 rotors + 1 chassis)
    quad_traces = create_quadrotor_traces(
        init_pos, init_phi, init_theta, init_psi,
        arm_length, quad_color, rotor_scale
    )
    for trace in quad_traces:
        fig_anim.add_trace(trace)
    
    # Trace 6: UAV trail (will be updated in frames)
    fig_anim.add_trace(go.Scatter3d(
        x=uav_x[:1], y=uav_y[:1], z=uav_z[:1],
        mode='lines',
        name='Target Trail',
        line=dict(color='#d62728', width=4),
        showlegend=False
    ))
    
    # Trace 7: UAV current position marker
    fig_anim.add_trace(go.Scatter3d(
        x=[uav_x[0]], y=[uav_y[0]], z=[uav_z[0]],
        mode='markers',
        name='Target UAV',
        marker=dict(color='#d62728', size=12, symbol='circle'),
        showlegend=False
    ))
    
    # Create animation frames
    frames = []
    for i, idx in enumerate(frame_indices):
        idx = int(idx)
        
        # Get quadrotor state at this frame
        frame_pos = np.array([quad_x[idx], quad_y[idx], quad_z[idx]])
        frame_phi, frame_theta, frame_psi = z[idx, 3], z[idx, 4], z[idx, 5]
        
        # Generate quadrotor traces for this frame
        frame_quad_traces = create_quadrotor_traces(
            frame_pos, frame_phi, frame_theta, frame_psi,
            arm_length, quad_color, rotor_scale
        )
        
        frame_data = [
            # Trace 0: Quadrotor trail
            go.Scatter3d(
                x=quad_x[:idx+1], y=quad_y[:idx+1], z=quad_z[:idx+1],
                mode='lines',
                line=dict(color=quad_color, width=4)
            ),
        ]
        
        # Traces 1-5: Quadrotor shape
        frame_data.extend(frame_quad_traces)
        
        # Trace 6: UAV trail
        frame_data.append(go.Scatter3d(
            x=uav_x[:idx+1], y=uav_y[:idx+1], z=uav_z[:idx+1],
            mode='lines',
            line=dict(color='#d62728', width=4)
        ))
        
        # Trace 7: UAV position
        frame_data.append(go.Scatter3d(
            x=[uav_x[idx]], y=[uav_y[idx]], z=[uav_z[idx]],
            mode='markers',
            marker=dict(color='#d62728', size=12, symbol='circle')
        ))
        
        frame = go.Frame(
            data=frame_data,
            name=f'frame{i}',
            traces=[0, 1, 2, 3, 4, 5, 6, 7]  # Update all 8 traces
        )
        frames.append(frame)
    
    fig_anim.frames = frames
    
    # Create slider steps
    slider_steps = []
    for i, idx in enumerate(frame_indices):
        step = dict(
            args=[[f'frame{i}'], dict(
                frame=dict(duration=0, redraw=True),
                mode='immediate',
                transition=dict(duration=0)
            )],
            label=f'{t[idx]:.1f}',
            method='animate'
        )
        slider_steps.append(step)
    
    # Add play/pause buttons and slider - NO AUTO-LOOP, NO LEGEND
    fig_anim.update_layout(
        title=dict(text='Simulation Playback', font=dict(size=16)),
        scene=dict(
            xaxis=dict(title='X (m)', range=x_range),
            yaxis=dict(title='Y (m)', range=y_range),
            zaxis=dict(title='Z (m)', range=z_range),
            aspectmode='cube',
            camera=dict(
                up=dict(x=0, y=0, z=1),
                center=dict(x=0, y=0, z=0),
                eye=dict(x=1.5, y=1.5, z=1.0)
            )
        ),
        showlegend=False,
        updatemenus=[
            dict(
                type='buttons',
                showactive=True,
                y=0,
                x=0.1,
                xanchor='right',
                yanchor='top',
                buttons=[
                    dict(
                        label='▶ Play',
                        method='animate',
                        args=[None, dict(
                            frame=dict(duration=50, redraw=True),
                            fromcurrent=True,
                            mode='immediate',
                            transition=dict(duration=0)
                        )]
                    ),
                    dict(
                        label='⏸ Pause',
                        method='animate',
                        args=[[None], dict(
                            frame=dict(duration=0, redraw=False),
                            mode='immediate',
                            transition=dict(duration=0)
                        )]
                    )
                ]
            )
        ],
        sliders=[dict(
            active=0,
            yanchor='top',
            xanchor='left',
            currentvalue=dict(
                font=dict(size=12),
                prefix='Time (s): ',
                visible=True,
                xanchor='right'
            ),
            transition=dict(duration=0),
            pad=dict(b=10, t=50),
            len=0.9,
            x=0.1,
            y=0,
            steps=slider_steps[::max(1, len(slider_steps)//25)]  # ~25 slider steps
        )],
        width=550,
        height=550,
        margin=dict(l=0, r=0, b=60, t=50)
    )
    
    # =========================================================================
    # Create Static 3D Trajectory Plot (NO LEGEND)
    # =========================================================================
    fig_3d = go.Figure()
    
    # Quadrotor trajectory (blue)
    fig_3d.add_trace(go.Scatter3d(
        x=quad_x, y=quad_y, z=quad_z,
        mode='lines',
        name='Quadrotor Trajectory',
        line=dict(color='#1f77b4', width=5),
        hovertemplate='<b>Quadrotor</b><br>x: %{x:.2f}<br>y: %{y:.2f}<br>z: %{z:.2f}<extra></extra>',
        showlegend=False
    ))
    
    # Quadrotor start/end markers
    fig_3d.add_trace(go.Scatter3d(
        x=[quad_x[0]], y=[quad_y[0]], z=[quad_z[0]],
        mode='markers',
        name='Quad Start',
        marker=dict(color='#1f77b4', size=12, symbol='circle'),
        hovertemplate='<b>Quad Start</b><br>x: %{x:.2f}<br>y: %{y:.2f}<br>z: %{z:.2f}<extra></extra>',
        showlegend=False
    ))
    
    fig_3d.add_trace(go.Scatter3d(
        x=[quad_x[-1]], y=[quad_y[-1]], z=[quad_z[-1]],
        mode='markers',
        name='Quad End',
        marker=dict(color='#1f77b4', size=12, symbol='square'),
        hovertemplate='<b>Quad End</b><br>x: %{x:.2f}<br>y: %{y:.2f}<br>z: %{z:.2f}<extra></extra>',
        showlegend=False
    ))
    
    # UAV trajectory (red)
    fig_3d.add_trace(go.Scatter3d(
        x=uav_x, y=uav_y, z=uav_z,
        mode='lines',
        name='Target UAV Trajectory',
        line=dict(color='#d62728', width=5),
        hovertemplate='<b>Target UAV</b><br>x: %{x:.2f}<br>y: %{y:.2f}<br>z: %{z:.2f}<extra></extra>',
        showlegend=False
    ))
    
    # UAV start/end markers
    fig_3d.add_trace(go.Scatter3d(
        x=[uav_x[0]], y=[uav_y[0]], z=[uav_z[0]],
        mode='markers',
        name='UAV Start',
        marker=dict(color='#d62728', size=12, symbol='circle'),
        hovertemplate='<b>UAV Start</b><br>x: %{x:.2f}<br>y: %{y:.2f}<br>z: %{z:.2f}<extra></extra>',
        showlegend=False
    ))
    
    fig_3d.add_trace(go.Scatter3d(
        x=[uav_x[-1]], y=[uav_y[-1]], z=[uav_z[-1]],
        mode='markers',
        name='UAV End',
        marker=dict(color='#d62728', size=12, symbol='square'),
        hovertemplate='<b>UAV End</b><br>x: %{x:.2f}<br>y: %{y:.2f}<br>z: %{z:.2f}<extra></extra>',
        showlegend=False
    ))
    
    # Mark capture point if applicable
    if capture_time is not None:
        fig_3d.add_trace(go.Scatter3d(
            x=[quad_x[min_dist_idx]], y=[quad_y[min_dist_idx]], z=[quad_z[min_dist_idx]],
            mode='markers',
            name='Capture Point',
            marker=dict(color='#2ca02c', size=15, symbol='diamond'),
            hovertemplate=f'<b>Capture!</b><br>t: {capture_time:.2f}s<br>x: %{{x:.2f}}<br>y: %{{y:.2f}}<br>z: %{{z:.2f}}<extra></extra>',
            showlegend=False
        ))
    
    fig_3d.update_layout(
        title=dict(text='Complete Trajectories', font=dict(size=16)),
        scene=dict(
            xaxis=dict(title='X (m)', range=x_range),
            yaxis=dict(title='Y (m)', range=y_range),
            zaxis=dict(title='Z (m)', range=z_range),
            aspectmode='cube',
            camera=dict(
                up=dict(x=0, y=0, z=1),
                center=dict(x=0, y=0, z=0),
                eye=dict(x=1.5, y=1.5, z=1.0)
            )
        ),
        showlegend=False,
        width=550,
        height=550,
        margin=dict(l=0, r=0, b=0, t=50)
    )
    
    # =========================================================================
    # Create Thrust Visualization Animation (Debug)
    # =========================================================================
    
    # Calculate thrust scale based on max thrust for good arrow visibility
    max_thrust = max(max(u1), max(u2), max(u3), max(u4))
    thrust_scale = 0.3 / max(max_thrust, 0.1)  # Normalize arrows to reasonable size
    
    # Create base figure for thrust visualization
    fig_thrust = go.Figure()
    
    # Add initial quadrotor shape traces (4 rotors + 1 chassis = 5 traces)
    init_quad_traces = create_quadrotor_traces(
        init_pos, init_phi, init_theta, init_psi,
        arm_length, quad_color, rotor_scale
    )
    for trace in init_quad_traces:
        fig_thrust.add_trace(trace)
    
    # Add initial thrust arrow traces (4 cones)
    init_thrusts = np.array([u1[0], u2[0], u3[0], u4[0]])
    init_thrust_traces = create_thrust_arrow_traces(
        init_pos, init_phi, init_theta, init_psi,
        init_thrusts, arm_length, thrust_scale, rotor_scale
    )
    for trace in init_thrust_traces:
        fig_thrust.add_trace(trace)
    
    # Create animation frames for thrust visualization
    thrust_frames = []
    for i, idx in enumerate(frame_indices):
        idx = int(idx)
        
        # Get quadrotor state at this frame
        frame_pos = np.array([quad_x[idx], quad_y[idx], quad_z[idx]])
        frame_phi, frame_theta, frame_psi = z[idx, 3], z[idx, 4], z[idx, 5]
        frame_thrusts = np.array([u1[idx], u2[idx], u3[idx], u4[idx]])
        
        # Generate quadrotor shape traces
        frame_quad_traces = create_quadrotor_traces(
            frame_pos, frame_phi, frame_theta, frame_psi,
            arm_length, quad_color, rotor_scale
        )
        
        # Generate thrust arrow traces
        frame_thrust_traces = create_thrust_arrow_traces(
            frame_pos, frame_phi, frame_theta, frame_psi,
            frame_thrusts, arm_length, thrust_scale, rotor_scale
        )
        
        frame_data = frame_quad_traces + frame_thrust_traces
        
        thrust_frame = go.Frame(
            data=frame_data,
            name=f'thrust_frame{i}',
            traces=[0, 1, 2, 3, 4, 5, 6, 7, 8]  # 5 quad traces + 4 thrust cones
        )
        thrust_frames.append(thrust_frame)
    
    fig_thrust.frames = thrust_frames
    
    # Create slider steps for thrust animation
    thrust_slider_steps = []
    for i, idx in enumerate(frame_indices):
        step = dict(
            args=[[f'thrust_frame{i}'], dict(
                frame=dict(duration=0, redraw=True),
                mode='immediate',
                transition=dict(duration=0)
            )],
            label=f'{t[idx]:.1f}',
            method='animate'
        )
        thrust_slider_steps.append(step)
    
    # Add layout with play/pause buttons and slider
    fig_thrust.update_layout(
        title=dict(text='Thrust Visualization (Debug)', font=dict(size=16)),
        scene=dict(
            xaxis=dict(title='X (m)', range=x_range),
            yaxis=dict(title='Y (m)', range=y_range),
            zaxis=dict(title='Z (m)', range=z_range),
            aspectmode='cube',
            camera=dict(
                up=dict(x=0, y=0, z=1),
                center=dict(x=0, y=0, z=0),
                eye=dict(x=1.5, y=1.5, z=1.0)
            )
        ),
        showlegend=False,
        updatemenus=[
            dict(
                type='buttons',
                showactive=True,
                y=0,
                x=0.1,
                xanchor='right',
                yanchor='top',
                buttons=[
                    dict(
                        label='▶ Play',
                        method='animate',
                        args=[None, dict(
                            frame=dict(duration=50, redraw=True),
                            fromcurrent=True,
                            mode='immediate',
                            transition=dict(duration=0)
                        )]
                    ),
                    dict(
                        label='⏸ Pause',
                        method='animate',
                        args=[[None], dict(
                            frame=dict(duration=0, redraw=False),
                            mode='immediate',
                            transition=dict(duration=0)
                        )]
                    )
                ]
            )
        ],
        sliders=[dict(
            active=0,
            yanchor='top',
            xanchor='left',
            currentvalue=dict(
                font=dict(size=12),
                prefix='Time (s): ',
                visible=True,
                xanchor='right'
            ),
            transition=dict(duration=0),
            pad=dict(b=10, t=50),
            len=0.9,
            x=0.1,
            y=0,
            steps=thrust_slider_steps[::max(1, len(thrust_slider_steps)//25)]
        )],
        width=1100,
        height=600,
        margin=dict(l=0, r=0, b=60, t=50)
    )
    
    # =========================================================================
    # Create State Plots
    # =========================================================================
    fig_states = make_subplots(
        rows=2, cols=2,
        subplot_titles=('Position', 'Velocity', 'Orientation (Euler Angles)', 'Angular Velocity'),
        vertical_spacing=0.12,
        horizontal_spacing=0.1
    )
    
    # Position
    fig_states.add_trace(go.Scatter(x=t_list, y=quad_x, name='x', line=dict(color='#1f77b4')), row=1, col=1)
    fig_states.add_trace(go.Scatter(x=t_list, y=quad_y, name='y', line=dict(color='#ff7f0e')), row=1, col=1)
    fig_states.add_trace(go.Scatter(x=t_list, y=quad_z, name='z', line=dict(color='#2ca02c')), row=1, col=1)
    
    # Velocity
    vx = z[:, 6].tolist()
    vy = z[:, 7].tolist()
    vz = z[:, 8].tolist()
    fig_states.add_trace(go.Scatter(x=t_list, y=vx, name='vx', line=dict(color='#1f77b4'), showlegend=False), row=1, col=2)
    fig_states.add_trace(go.Scatter(x=t_list, y=vy, name='vy', line=dict(color='#ff7f0e'), showlegend=False), row=1, col=2)
    fig_states.add_trace(go.Scatter(x=t_list, y=vz, name='vz', line=dict(color='#2ca02c'), showlegend=False), row=1, col=2)
    
    # Orientation (convert to degrees)
    phi = np.rad2deg(z[:, 3]).tolist()
    theta = np.rad2deg(z[:, 4]).tolist()
    psi = np.rad2deg(z[:, 5]).tolist()
    fig_states.add_trace(go.Scatter(x=t_list, y=phi, name='φ (roll)', line=dict(color='#d62728')), row=2, col=1)
    fig_states.add_trace(go.Scatter(x=t_list, y=theta, name='θ (pitch)', line=dict(color='#9467bd')), row=2, col=1)
    fig_states.add_trace(go.Scatter(x=t_list, y=psi, name='ψ (yaw)', line=dict(color='#8c564b')), row=2, col=1)
    
    # Angular velocity
    w1 = z[:, 9].tolist()
    w2 = z[:, 10].tolist()
    w3 = z[:, 11].tolist()
    fig_states.add_trace(go.Scatter(x=t_list, y=w1, name='ω1', line=dict(color='#d62728'), showlegend=False), row=2, col=2)
    fig_states.add_trace(go.Scatter(x=t_list, y=w2, name='ω2', line=dict(color='#9467bd'), showlegend=False), row=2, col=2)
    fig_states.add_trace(go.Scatter(x=t_list, y=w3, name='ω3', line=dict(color='#8c564b'), showlegend=False), row=2, col=2)
    
    fig_states.update_xaxes(title_text='Time (s)', row=2, col=1)
    fig_states.update_xaxes(title_text='Time (s)', row=2, col=2)
    fig_states.update_yaxes(title_text='Position (m)', row=1, col=1)
    fig_states.update_yaxes(title_text='Velocity (m/s)', row=1, col=2)
    fig_states.update_yaxes(title_text='Angle (deg)', row=2, col=1)
    fig_states.update_yaxes(title_text='Angular Velocity (rad/s)', row=2, col=2)
    
    fig_states.update_layout(
        title=dict(text='Quadrotor State History', font=dict(size=20)),
        height=600,
        width=1100,
        showlegend=True,
        legend=dict(orientation='h', y=1.02, x=0.5, xanchor='center')
    )
    
    # =========================================================================
    # Create Control Input (Rotor Thrusts) Plot
    # =========================================================================
    fig_control = go.Figure()
    
    fig_control.add_trace(go.Scatter(x=t_list, y=u1, name='u1 (Front)', line=dict(color='#1f77b4', width=2)))
    fig_control.add_trace(go.Scatter(x=t_list, y=u2, name='u2 (Left)', line=dict(color='#ff7f0e', width=2)))
    fig_control.add_trace(go.Scatter(x=t_list, y=u3, name='u3 (Back)', line=dict(color='#2ca02c', width=2)))
    fig_control.add_trace(go.Scatter(x=t_list, y=u4, name='u4 (Right)', line=dict(color='#d62728', width=2)))
    
    fig_control.update_layout(
        title=dict(text='Control Inputs (Rotor Thrusts)', font=dict(size=20)),
        xaxis_title='Time (s)',
        yaxis_title='Thrust (N)',
        height=400,
        width=1100,
        legend=dict(orientation='h', y=1.02, x=0.5, xanchor='center')
    )
    
    # =========================================================================
    # Create Distance Plot (total distance to UAV)
    # =========================================================================
    fig_distance = go.Figure()
    
    fig_distance.add_trace(go.Scatter(
        x=t_list, y=distances_list,
        name='Distance to Target',
        line=dict(color='#17becf', width=2),
        fill='tozeroy',
        fillcolor='rgba(23, 190, 207, 0.2)'
    ))
    
    # Add capture threshold line
    fig_distance.add_hline(y=0.1, line_dash='dash', line_color='red',
                          annotation_text='Capture Threshold (0.1m)')
    
    if capture_time is not None:
        fig_distance.add_vline(x=capture_time, line_dash='dash', line_color='green',
                              annotation_text=f'Capture (t={capture_time:.2f}s)')
    
    fig_distance.update_layout(
        title=dict(text='Distance to Target UAV', font=dict(size=20)),
        xaxis_title='Time (s)',
        yaxis_title='Distance (m)',
        height=350,
        width=1100
    )
    
    # =========================================================================
    # Create Position Comparison Plot (Quadrotor vs Target in x, y, z)
    # =========================================================================
    fig_dist_components = go.Figure()
    
    # Quadrotor positions (solid lines)
    fig_dist_components.add_trace(go.Scatter(
        x=t_list, y=quad_x,
        name='Quadrotor X',
        line=dict(color='#1f77b4', width=2, dash='solid')
    ))
    fig_dist_components.add_trace(go.Scatter(
        x=t_list, y=quad_y,
        name='Quadrotor Y',
        line=dict(color='#ff7f0e', width=2, dash='solid')
    ))
    fig_dist_components.add_trace(go.Scatter(
        x=t_list, y=quad_z,
        name='Quadrotor Z',
        line=dict(color='#2ca02c', width=2, dash='solid')
    ))
    
    # Target UAV positions (dashed lines, same colors)
    fig_dist_components.add_trace(go.Scatter(
        x=t_list, y=uav_x,
        name='Target X',
        line=dict(color='#1f77b4', width=2, dash='dash')
    ))
    fig_dist_components.add_trace(go.Scatter(
        x=t_list, y=uav_y,
        name='Target Y',
        line=dict(color='#ff7f0e', width=2, dash='dash')
    ))
    fig_dist_components.add_trace(go.Scatter(
        x=t_list, y=uav_z,
        name='Target Z',
        line=dict(color='#2ca02c', width=2, dash='dash')
    ))
    
    fig_dist_components.update_layout(
        title=dict(text='Position Comparison: Quadrotor (solid) vs Target (dashed)', font=dict(size=20)),
        xaxis_title='Time (s)',
        yaxis_title='Position (m)',
        height=400,
        width=1100,
        legend=dict(orientation='h', y=1.02, x=0.5, xanchor='center')
    )
    
    # =========================================================================
    # Generate HTML
    # =========================================================================
    
    # Convert plots to HTML
    plot_anim_html = fig_anim.to_html(full_html=False, include_plotlyjs=False)
    plot_3d_html = fig_3d.to_html(full_html=False, include_plotlyjs=False)
    plot_thrust_html = fig_thrust.to_html(full_html=False, include_plotlyjs=False)
    plot_states_html = fig_states.to_html(full_html=False, include_plotlyjs=False)
    plot_control_html = fig_control.to_html(full_html=False, include_plotlyjs=False)
    plot_distance_html = fig_distance.to_html(full_html=False, include_plotlyjs=False)
    plot_dist_components_html = fig_dist_components.to_html(full_html=False, include_plotlyjs=False)
    
    # Generate timestamp
    timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    
    # Build HTML content
    html_content = f'''<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Quadrotor Simulation Report</title>
    <script src="https://cdn.plot.ly/plotly-2.27.0.min.js"></script>
    <style>
        :root {{
            --bg-primary: #0d1117;
            --bg-secondary: #161b22;
            --bg-card: #21262d;
            --text-primary: #f0f6fc;
            --text-secondary: #8b949e;
            --accent-blue: #58a6ff;
            --accent-green: #3fb950;
            --accent-red: #f85149;
            --accent-purple: #a371f7;
            --border-color: #30363d;
        }}
        
        * {{
            margin: 0;
            padding: 0;
            box-sizing: border-box;
        }}
        
        body {{
            font-family: 'Segoe UI', -apple-system, BlinkMacSystemFont, 'Roboto', sans-serif;
            background: var(--bg-primary);
            color: var(--text-primary);
            line-height: 1.6;
            min-height: 100vh;
        }}
        
        .container {{
            max-width: 1200px;
            margin: 0 auto;
            padding: 2rem;
        }}
        
        .container-wide {{
            max-width: 1400px;
            margin: 0 auto;
            padding: 2rem;
        }}
        
        header {{
            background: linear-gradient(135deg, var(--bg-secondary) 0%, #1a2332 100%);
            border-bottom: 1px solid var(--border-color);
            padding: 2rem 0;
            margin-bottom: 2rem;
        }}
        
        header .container {{
            display: flex;
            justify-content: space-between;
            align-items: center;
        }}
        
        .logo {{
            display: flex;
            align-items: center;
            gap: 1rem;
        }}
        
        .logo-icon {{
            width: 50px;
            height: 50px;
            background: linear-gradient(135deg, var(--accent-blue), var(--accent-purple));
            border-radius: 12px;
            display: flex;
            align-items: center;
            justify-content: center;
            font-size: 1.5rem;
        }}
        
        h1 {{
            font-size: 1.8rem;
            font-weight: 600;
            background: linear-gradient(90deg, var(--accent-blue), var(--accent-purple));
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
            background-clip: text;
        }}
        
        .timestamp {{
            color: var(--text-secondary);
            font-size: 0.9rem;
        }}
        
        .stats-grid {{
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
            gap: 1rem;
            margin-bottom: 2rem;
        }}
        
        .stat-card {{
            background: var(--bg-card);
            border: 1px solid var(--border-color);
            border-radius: 12px;
            padding: 1.5rem;
            transition: transform 0.2s, box-shadow 0.2s;
        }}
        
        .stat-card:hover {{
            transform: translateY(-2px);
            box-shadow: 0 8px 25px rgba(0,0,0,0.3);
        }}
        
        .stat-label {{
            color: var(--text-secondary);
            font-size: 0.85rem;
            margin-bottom: 0.5rem;
            text-transform: uppercase;
            letter-spacing: 0.5px;
        }}
        
        .stat-value {{
            font-size: 1.8rem;
            font-weight: 700;
        }}
        
        .stat-value.blue {{ color: var(--accent-blue); }}
        .stat-value.green {{ color: var(--accent-green); }}
        .stat-value.red {{ color: var(--accent-red); }}
        .stat-value.purple {{ color: var(--accent-purple); }}
        
        .section {{
            background: var(--bg-card);
            border: 1px solid var(--border-color);
            border-radius: 12px;
            margin-bottom: 2rem;
            overflow: hidden;
        }}
        
        .section-header {{
            background: var(--bg-secondary);
            padding: 1rem 1.5rem;
            border-bottom: 1px solid var(--border-color);
            display: flex;
            align-items: center;
            gap: 0.75rem;
        }}
        
        .section-header h2 {{
            font-size: 1.1rem;
            font-weight: 600;
        }}
        
        .section-icon {{
            width: 32px;
            height: 32px;
            border-radius: 8px;
            display: flex;
            align-items: center;
            justify-content: center;
            font-size: 1rem;
        }}
        
        .section-icon.blue {{ background: rgba(88, 166, 255, 0.2); }}
        .section-icon.green {{ background: rgba(63, 185, 80, 0.2); }}
        .section-icon.purple {{ background: rgba(163, 113, 247, 0.2); }}
        .section-icon.orange {{ background: rgba(255, 127, 14, 0.2); }}
        
        .section-content {{
            padding: 1.5rem;
        }}
        
        .plot-container {{
            background: white;
            border-radius: 8px;
            overflow: hidden;
        }}
        
        .dual-plot-container {{
            display: flex;
            gap: 1rem;
            flex-wrap: wrap;
            justify-content: center;
        }}
        
        .dual-plot-container .plot-container {{
            flex: 1;
            min-width: 500px;
            max-width: 600px;
        }}
        
        .params-grid {{
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(180px, 1fr));
            gap: 1rem;
        }}
        
        .param-item {{
            padding: 0.75rem;
            background: var(--bg-secondary);
            border-radius: 8px;
        }}
        
        .param-name {{
            color: var(--text-secondary);
            font-size: 0.8rem;
        }}
        
        .param-value {{
            font-weight: 600;
            color: var(--accent-blue);
        }}
        
        footer {{
            text-align: center;
            padding: 2rem;
            color: var(--text-secondary);
            font-size: 0.85rem;
            border-top: 1px solid var(--border-color);
            margin-top: 2rem;
        }}
        
        .axis-info {{
            color: var(--text-secondary);
            font-size: 0.85rem;
            margin-top: 0.5rem;
            text-align: center;
        }}
        
        @media (max-width: 768px) {{
            .container, .container-wide {{
                padding: 1rem;
            }}
            
            header .container {{
                flex-direction: column;
                gap: 1rem;
                text-align: center;
            }}
            
            .dual-plot-container {{
                flex-direction: column;
            }}
            
            .dual-plot-container .plot-container {{
                min-width: 100%;
            }}
        }}
    </style>
</head>
<body>
    <header>
        <div class="container">
            <div class="logo">
                <div class="logo-icon">🚁</div>
                <div>
                    <h1>Quadrotor Simulation Report</h1>
                    <div class="timestamp">Generated: {timestamp}</div>
                </div>
            </div>
        </div>
    </header>
    
    <main class="container-wide">
        <!-- Statistics Cards -->
        <div class="stats-grid">
            <div class="stat-card">
                <div class="stat-label">Simulation Duration</div>
                <div class="stat-value blue">{t_list[-1]:.1f} s</div>
            </div>
            <div class="stat-card">
                <div class="stat-label">Time Steps</div>
                <div class="stat-value purple">{len(t_list)}</div>
            </div>
            <div class="stat-card">
                <div class="stat-label">Minimum Distance</div>
                <div class="stat-value {'green' if min_dist < 0.1 else 'red'}">{min_dist:.4f} m</div>
            </div>
            <div class="stat-card">
                <div class="stat-label">Capture Status</div>
                <div class="stat-value {'green' if capture_time else 'red'}">{'✓ Captured' if capture_time else '✗ Not Captured'}</div>
            </div>
        </div>
        
        <!-- 3D Visualization - Animation + Static -->
        <div class="section">
            <div class="section-header">
                <div class="section-icon blue">📍</div>
                <h2>3D Trajectory Visualization</h2>
            </div>
            <div class="section-content">
                <div class="dual-plot-container">
                    <div class="plot-container">
                        {plot_anim_html}
                    </div>
                    <div class="plot-container">
                        {plot_3d_html}
                    </div>
                </div>
                <div class="axis-info">
                    Axis bounds: X [{x_range[0]}, {x_range[1]}] m | Y [{y_range[0]}, {y_range[1]}] m | Z [{z_range[0]}, {z_range[1]}] m
                    <br>
                    <span style="color: #1f77b4;">●</span> Quadrotor (Blue) | <span style="color: #d62728;">●</span> Target UAV (Red)
                    <br>
                    <em>Tip: Pause the animation to rotate/zoom the 3D view</em>
                </div>
            </div>
        </div>
        
        <!-- Thrust Visualization (Debug) -->
        <div class="section">
            <div class="section-header">
                <div class="section-icon orange">🔥</div>
                <h2>Thrust Visualization (Debug)</h2>
            </div>
            <div class="section-content">
                <div class="plot-container">
                    {plot_thrust_html}
                </div>
                <div class="axis-info">
                    Arrows show rotor thrust magnitude and direction (perpendicular to rotor disc, tilting with quadrotor)
                    <br>
                    <span style="color: #1f77b4;">▲</span> u1 (Front) | 
                    <span style="color: #ff7f0e;">▲</span> u2 (Left) | 
                    <span style="color: #2ca02c;">▲</span> u3 (Back) | 
                    <span style="color: #d62728;">▲</span> u4 (Right)
                </div>
            </div>
        </div>
        
        <!-- Distance Plot -->
        <div class="section">
            <div class="section-header">
                <div class="section-icon green">📏</div>
                <h2>Distance to Target UAV</h2>
            </div>
            <div class="section-content">
                <div class="plot-container">
                    {plot_distance_html}
                </div>
            </div>
        </div>
        
        <!-- Position Comparison Plot -->
        <div class="section">
            <div class="section-header">
                <div class="section-icon orange">📐</div>
                <h2>Position Comparison (Quadrotor vs Target)</h2>
            </div>
            <div class="section-content">
                <div class="plot-container">
                    {plot_dist_components_html}
                </div>
            </div>
        </div>
        
        <!-- State History -->
        <div class="section">
            <div class="section-header">
                <div class="section-icon purple">📊</div>
                <h2>State History</h2>
            </div>
            <div class="section-content">
                <div class="plot-container">
                    {plot_states_html}
                </div>
            </div>
        </div>
        
        <!-- Control Inputs -->
        <div class="section">
            <div class="section-header">
                <div class="section-icon blue">🎮</div>
                <h2>Control Inputs (Rotor Thrusts)</h2>
            </div>
            <div class="section-content">
                <div class="plot-container">
                    {plot_control_html}
                </div>
            </div>
        </div>
        
        <!-- Parameters -->
        <div class="section">
            <div class="section-header">
                <div class="section-icon green">⚙️</div>
                <h2>Simulation Parameters</h2>
            </div>
            <div class="section-content">
                <div class="params-grid">
                    <div class="param-item">
                        <div class="param-name">Mass (m)</div>
                        <div class="param-value">{quadrotor.m} kg</div>
                    </div>
                    <div class="param-item">
                        <div class="param-name">Arm Length (l)</div>
                        <div class="param-value">{quadrotor.l} m</div>
                    </div>
                    <div class="param-item">
                        <div class="param-name">Gravity (g)</div>
                        <div class="param-value">{quadrotor.g} m/s²</div>
                    </div>
                    <div class="param-item">
                        <div class="param-name">Max Thrust (μ)</div>
                        <div class="param-value">{quadrotor.mu} N</div>
                    </div>
                    <div class="param-item">
                        <div class="param-name">Thrust-Torque (σ)</div>
                        <div class="param-value">{quadrotor.sigma}</div>
                    </div>
                    <div class="param-item">
                        <div class="param-name">Trajectory</div>
                        <div class="param-value">{trajectory_name.capitalize()}</div>
                    </div>
                    <div class="param-item">
                        <div class="param-name">Time Step</div>
                        <div class="param-value">{t_list[1]-t_list[0]:.3f} s</div>
                    </div>
                    <div class="param-item">
                        <div class="param-name">Inertia (I₁₁)</div>
                        <div class="param-value">{quadrotor.I[0,0]} kg·m²</div>
                    </div>
                </div>
            </div>
        </div>
    </main>
    
    <footer>
        <p>Quadrotor Tracking Simulation • Python Implementation</p>
        <p>Original MATLAB code by Team C</p>
    </footer>
</body>
</html>'''
    
    # Write to file
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write(html_content)
    
    return output_path


if __name__ == '__main__':
    print("This module is meant to be imported. Run run_simulation.py with --html-report flag.")
