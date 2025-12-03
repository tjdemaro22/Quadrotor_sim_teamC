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
from typing import Optional


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
    
    # Calculate axis limits with padding
    x_range = [sim_bounds[0][0] - axis_padding, sim_bounds[0][1] + axis_padding]
    y_range = [sim_bounds[1][0] - axis_padding, sim_bounds[1][1] + axis_padding]
    z_range = [sim_bounds[2][0], sim_bounds[2][1] + axis_padding]  # Only pad top of z
    
    # Calculate some statistics
    distances = np.linalg.norm(z[:, :3] - y, axis=1)
    min_dist = np.min(distances)
    min_dist_idx = np.argmin(distances)
    capture_time = t[min_dist_idx] if min_dist < 0.1 else None
    
    # =========================================================================
    # Create Animated 3D Plot (looping simulation playback)
    # =========================================================================
    
    # Subsample for animation performance (aim for ~200 frames)
    n_frames = min(200, len(t))
    frame_indices = np.linspace(0, len(t) - 1, n_frames, dtype=int)
    
    # Create base figure with initial positions
    fig_anim = go.Figure()
    
    # Quadrotor trail (will be updated in frames)
    fig_anim.add_trace(go.Scatter3d(
        x=[z[0, 0]], y=[z[0, 1]], z=[z[0, 2]],
        mode='lines',
        name='Quadrotor Trail',
        line=dict(color='#1f77b4', width=3),
        showlegend=True
    ))
    
    # Quadrotor current position marker
    fig_anim.add_trace(go.Scatter3d(
        x=[z[0, 0]], y=[z[0, 1]], z=[z[0, 2]],
        mode='markers',
        name='Quadrotor',
        marker=dict(color='#1f77b4', size=8, symbol='diamond'),
        showlegend=True
    ))
    
    # UAV trail (will be updated in frames)
    fig_anim.add_trace(go.Scatter3d(
        x=[y[0, 0]], y=[y[0, 1]], z=[y[0, 2]],
        mode='lines',
        name='Target Trail',
        line=dict(color='#d62728', width=3),
        showlegend=True
    ))
    
    # UAV current position marker
    fig_anim.add_trace(go.Scatter3d(
        x=[y[0, 0]], y=[y[0, 1]], z=[y[0, 2]],
        mode='markers',
        name='Target UAV',
        marker=dict(color='#d62728', size=10, symbol='circle'),
        showlegend=True
    ))
    
    # Create animation frames
    frames = []
    for i, idx in enumerate(frame_indices):
        frame = go.Frame(
            data=[
                # Quadrotor trail
                go.Scatter3d(
                    x=z[:idx+1, 0], y=z[:idx+1, 1], z=z[:idx+1, 2],
                    mode='lines',
                    line=dict(color='#1f77b4', width=3)
                ),
                # Quadrotor position
                go.Scatter3d(
                    x=[z[idx, 0]], y=[z[idx, 1]], z=[z[idx, 2]],
                    mode='markers',
                    marker=dict(color='#1f77b4', size=8, symbol='diamond')
                ),
                # UAV trail
                go.Scatter3d(
                    x=y[:idx+1, 0], y=y[:idx+1, 1], z=y[:idx+1, 2],
                    mode='lines',
                    line=dict(color='#d62728', width=3)
                ),
                # UAV position
                go.Scatter3d(
                    x=[y[idx, 0]], y=[y[idx, 1]], z=[y[idx, 2]],
                    mode='markers',
                    marker=dict(color='#d62728', size=10, symbol='circle')
                )
            ],
            name=f't={t[idx]:.2f}s',
            traces=[0, 1, 2, 3]
        )
        frames.append(frame)
    
    fig_anim.frames = frames
    
    # Add play/pause buttons and slider
    fig_anim.update_layout(
        title=dict(text='Simulation Playback (Auto-loops)', font=dict(size=16)),
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
        updatemenus=[
            dict(
                type='buttons',
                showactive=False,
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
                prefix='Time: ',
                visible=True,
                xanchor='right'
            ),
            transition=dict(duration=0),
            pad=dict(b=10, t=50),
            len=0.9,
            x=0.1,
            y=0,
            steps=[
                dict(
                    args=[[f't={t[idx]:.2f}s'], dict(
                        frame=dict(duration=0, redraw=True),
                        mode='immediate',
                        transition=dict(duration=0)
                    )],
                    label=f'{t[idx]:.1f}s',
                    method='animate'
                )
                for idx in frame_indices[::max(1, len(frame_indices)//20)]  # ~20 slider steps
            ]
        )],
        width=550,
        height=550,
        margin=dict(l=0, r=0, b=60, t=50),
        legend=dict(x=0.02, y=0.98, bgcolor='rgba(255,255,255,0.8)', font=dict(size=10))
    )
    
    # =========================================================================
    # Create Static 3D Trajectory Plot
    # =========================================================================
    fig_3d = go.Figure()
    
    # Quadrotor trajectory (blue)
    fig_3d.add_trace(go.Scatter3d(
        x=z[:, 0], y=z[:, 1], z=z[:, 2],
        mode='lines',
        name='Quadrotor Trajectory',
        line=dict(color='#1f77b4', width=4),
        hovertemplate='<b>Quadrotor</b><br>x: %{x:.2f}<br>y: %{y:.2f}<br>z: %{z:.2f}<extra></extra>'
    ))
    
    # Quadrotor start/end markers
    fig_3d.add_trace(go.Scatter3d(
        x=[z[0, 0]], y=[z[0, 1]], z=[z[0, 2]],
        mode='markers',
        name='Quad Start',
        marker=dict(color='#1f77b4', size=10, symbol='circle'),
        hovertemplate='<b>Quad Start</b><br>x: %{x:.2f}<br>y: %{y:.2f}<br>z: %{z:.2f}<extra></extra>'
    ))
    
    fig_3d.add_trace(go.Scatter3d(
        x=[z[-1, 0]], y=[z[-1, 1]], z=[z[-1, 2]],
        mode='markers',
        name='Quad End',
        marker=dict(color='#1f77b4', size=10, symbol='square'),
        hovertemplate='<b>Quad End</b><br>x: %{x:.2f}<br>y: %{y:.2f}<br>z: %{z:.2f}<extra></extra>'
    ))
    
    # UAV trajectory (red)
    fig_3d.add_trace(go.Scatter3d(
        x=y[:, 0], y=y[:, 1], z=y[:, 2],
        mode='lines',
        name='Target UAV Trajectory',
        line=dict(color='#d62728', width=4),
        hovertemplate='<b>Target UAV</b><br>x: %{x:.2f}<br>y: %{y:.2f}<br>z: %{z:.2f}<extra></extra>'
    ))
    
    # UAV start/end markers
    fig_3d.add_trace(go.Scatter3d(
        x=[y[0, 0]], y=[y[0, 1]], z=[y[0, 2]],
        mode='markers',
        name='UAV Start',
        marker=dict(color='#d62728', size=10, symbol='circle'),
        hovertemplate='<b>UAV Start</b><br>x: %{x:.2f}<br>y: %{y:.2f}<br>z: %{z:.2f}<extra></extra>'
    ))
    
    fig_3d.add_trace(go.Scatter3d(
        x=[y[-1, 0]], y=[y[-1, 1]], z=[y[-1, 2]],
        mode='markers',
        name='UAV End',
        marker=dict(color='#d62728', size=10, symbol='square'),
        hovertemplate='<b>UAV End</b><br>x: %{x:.2f}<br>y: %{y:.2f}<br>z: %{z:.2f}<extra></extra>'
    ))
    
    # Mark capture point if applicable
    if capture_time is not None:
        fig_3d.add_trace(go.Scatter3d(
            x=[z[min_dist_idx, 0]], y=[z[min_dist_idx, 1]], z=[z[min_dist_idx, 2]],
            mode='markers',
            name='Capture Point',
            marker=dict(color='#2ca02c', size=15, symbol='diamond'),
            hovertemplate=f'<b>Capture!</b><br>t: {capture_time:.2f}s<br>x: %{{x:.2f}}<br>y: %{{y:.2f}}<br>z: %{{z:.2f}}<extra></extra>'
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
        width=550,
        height=550,
        margin=dict(l=0, r=0, b=0, t=50),
        legend=dict(x=0.02, y=0.98, bgcolor='rgba(255,255,255,0.8)', font=dict(size=10))
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
    fig_states.add_trace(go.Scatter(x=t, y=z[:, 0], name='x', line=dict(color='#1f77b4')), row=1, col=1)
    fig_states.add_trace(go.Scatter(x=t, y=z[:, 1], name='y', line=dict(color='#ff7f0e')), row=1, col=1)
    fig_states.add_trace(go.Scatter(x=t, y=z[:, 2], name='z', line=dict(color='#2ca02c')), row=1, col=1)
    
    # Velocity
    fig_states.add_trace(go.Scatter(x=t, y=z[:, 6], name='vx', line=dict(color='#1f77b4'), showlegend=False), row=1, col=2)
    fig_states.add_trace(go.Scatter(x=t, y=z[:, 7], name='vy', line=dict(color='#ff7f0e'), showlegend=False), row=1, col=2)
    fig_states.add_trace(go.Scatter(x=t, y=z[:, 8], name='vz', line=dict(color='#2ca02c'), showlegend=False), row=1, col=2)
    
    # Orientation (convert to degrees)
    fig_states.add_trace(go.Scatter(x=t, y=np.rad2deg(z[:, 3]), name='φ (roll)', line=dict(color='#d62728')), row=2, col=1)
    fig_states.add_trace(go.Scatter(x=t, y=np.rad2deg(z[:, 4]), name='θ (pitch)', line=dict(color='#9467bd')), row=2, col=1)
    fig_states.add_trace(go.Scatter(x=t, y=np.rad2deg(z[:, 5]), name='ψ (yaw)', line=dict(color='#8c564b')), row=2, col=1)
    
    # Angular velocity
    fig_states.add_trace(go.Scatter(x=t, y=z[:, 9], name='ω1', line=dict(color='#d62728'), showlegend=False), row=2, col=2)
    fig_states.add_trace(go.Scatter(x=t, y=z[:, 10], name='ω2', line=dict(color='#9467bd'), showlegend=False), row=2, col=2)
    fig_states.add_trace(go.Scatter(x=t, y=z[:, 11], name='ω3', line=dict(color='#8c564b'), showlegend=False), row=2, col=2)
    
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
    # Create Control Input Plot
    # =========================================================================
    fig_control = go.Figure()
    
    fig_control.add_trace(go.Scatter(x=t, y=u[:, 0], name='u1 (Front)', line=dict(color='#1f77b4')))
    fig_control.add_trace(go.Scatter(x=t, y=u[:, 1], name='u2 (Left)', line=dict(color='#ff7f0e')))
    fig_control.add_trace(go.Scatter(x=t, y=u[:, 2], name='u3 (Back)', line=dict(color='#2ca02c')))
    fig_control.add_trace(go.Scatter(x=t, y=u[:, 3], name='u4 (Right)', line=dict(color='#d62728')))
    
    fig_control.update_layout(
        title=dict(text='Control Inputs (Rotor Thrusts)', font=dict(size=20)),
        xaxis_title='Time (s)',
        yaxis_title='Thrust (N)',
        height=400,
        width=1100,
        legend=dict(orientation='h', y=1.02, x=0.5, xanchor='center')
    )
    
    # =========================================================================
    # Create Motor Torques Plot
    # =========================================================================
    # Each motor produces a reaction torque proportional to its thrust: τ = σ × u
    # This is the torque applied to the airframe due to motor spinning
    sigma = quadrotor.sigma
    motor_torques = u * sigma  # (n_times, 4) - torque from each motor
    
    fig_motor_torques = go.Figure()
    
    fig_motor_torques.add_trace(go.Scatter(
        x=t, y=motor_torques[:, 0], 
        name='τ₁ (Front)', 
        line=dict(color='#1f77b4', width=2)
    ))
    fig_motor_torques.add_trace(go.Scatter(
        x=t, y=motor_torques[:, 1], 
        name='τ₂ (Left)', 
        line=dict(color='#ff7f0e', width=2)
    ))
    fig_motor_torques.add_trace(go.Scatter(
        x=t, y=motor_torques[:, 2], 
        name='τ₃ (Back)', 
        line=dict(color='#2ca02c', width=2)
    ))
    fig_motor_torques.add_trace(go.Scatter(
        x=t, y=motor_torques[:, 3], 
        name='τ₄ (Right)', 
        line=dict(color='#d62728', width=2)
    ))
    
    fig_motor_torques.update_layout(
        title=dict(text='Motor Reaction Torques (τ = σ × u)', font=dict(size=20)),
        xaxis_title='Time (s)',
        yaxis_title='Torque (N·m)',
        height=400,
        width=1100,
        legend=dict(orientation='h', y=1.02, x=0.5, xanchor='center')
    )
    
    # =========================================================================
    # Create Distance Plot
    # =========================================================================
    fig_distance = go.Figure()
    
    fig_distance.add_trace(go.Scatter(
        x=t, y=distances,
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
    # Generate HTML
    # =========================================================================
    
    # Convert plots to HTML
    plot_anim_html = fig_anim.to_html(full_html=False, include_plotlyjs=False)
    plot_3d_html = fig_3d.to_html(full_html=False, include_plotlyjs=False)
    plot_states_html = fig_states.to_html(full_html=False, include_plotlyjs=False)
    plot_control_html = fig_control.to_html(full_html=False, include_plotlyjs=False)
    plot_motor_torques_html = fig_motor_torques.to_html(full_html=False, include_plotlyjs=False)
    plot_distance_html = fig_distance.to_html(full_html=False, include_plotlyjs=False)
    
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
                <div class="stat-value blue">{t[-1]:.1f} s</div>
            </div>
            <div class="stat-card">
                <div class="stat-label">Time Steps</div>
                <div class="stat-value purple">{len(t)}</div>
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
                </div>
            </div>
        </div>
        
        <!-- Distance Plot -->
        <div class="section">
            <div class="section-header">
                <div class="section-icon green">📏</div>
                <h2>Distance to Target</h2>
            </div>
            <div class="section-content">
                <div class="plot-container">
                    {plot_distance_html}
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
                <h2>Control Inputs</h2>
            </div>
            <div class="section-content">
                <div class="plot-container">
                    {plot_control_html}
                </div>
            </div>
        </div>
        
        <!-- Motor Torques -->
        <div class="section">
            <div class="section-header">
                <div class="section-icon purple">🔄</div>
                <h2>Motor Reaction Torques</h2>
            </div>
            <div class="section-content">
                <div class="plot-container">
                    {plot_motor_torques_html}
                </div>
                <div class="axis-info">
                    Each motor produces a reaction torque τ = σ × u (thrust-to-torque ratio σ = {quadrotor.sigma})
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
                        <div class="param-value">{t[1]-t[0]:.3f} s</div>
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
    
    <script>
        // Auto-play animation on page load and loop it
        document.addEventListener('DOMContentLoaded', function() {{
            // Find the animation plot div (it's the first plotly plot)
            setTimeout(function() {{
                const plots = document.querySelectorAll('.js-plotly-plot');
                if (plots.length > 0) {{
                    const animPlot = plots[0];
                    
                    // Function to play animation
                    function playAnimation() {{
                        Plotly.animate(animPlot, null, {{
                            frame: {{duration: 50, redraw: true}},
                            fromcurrent: false,
                            mode: 'immediate',
                            transition: {{duration: 0}}
                        }});
                    }}
                    
                    // Play on load
                    playAnimation();
                    
                    // Loop: restart when animation ends
                    animPlot.on('plotly_animated', function() {{
                        setTimeout(playAnimation, 1000);  // 1 second pause before restart
                    }});
                }}
            }}, 500);
        }});
    </script>
</body>
</html>'''
    
    # Write to file
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write(html_content)
    
    return output_path


if __name__ == '__main__':
    print("This module is meant to be imported. Run run_simulation.py with --html-report flag.")
