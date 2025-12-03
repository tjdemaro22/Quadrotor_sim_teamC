# Quadrotor Tracking Simulation - Python Implementation

This is a Python port of the MATLAB quadrotor tracking simulation. It simulates a quadrotor UAV tracking and intercepting a target UAV using LQR-based control.

## Features

- **Quadrotor Dynamics**: Full 12-state nonlinear dynamics with thrust and torque inputs
- **LQR Control**: Sophisticated Adaptive Controller (SAC) with trajectory prediction
- **Multiple Trajectories**: Circular, linear, diagonal, and sinusoidal target paths
- **3D Visualization**: Real-time matplotlib animation with trajectory trails
- **Interactive HTML Report**: Plotly-based interactive 3D visualization and plots

## Installation

1. Create a virtual environment (recommended):
```bash
python -m venv venv
venv\Scripts\activate  # Windows
# or
source venv/bin/activate  # Linux/Mac
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

## Usage

### Basic Simulation with Animation

```bash
python run_simulation.py
```

### Command Line Options

```bash
# Run with static plot instead of animation
python run_simulation.py --no-animation

# Save animation as GIF
python run_simulation.py --save-gif

# Generate HTML report
python run_simulation.py --html-report

# Choose different trajectory
python run_simulation.py --trajectory circular   # default
python run_simulation.py --trajectory linear
python run_simulation.py --trajectory diagonal
python run_simulation.py --trajectory sinusoidal

# Combine options
python run_simulation.py --html-report --trajectory circular
```

### Using as a Library

```python
import numpy as np
from src import Quadrotor, UAV, SACController, Simulator

# Create quadrotor
quad = Quadrotor(g=9.81, l=0.2, m=0.5, 
                 I=np.diag([1.24, 1.24, 2.48]),
                 mu=3.0, sigma=0.01)

# Create target UAV with circular trajectory
path = lambda t: np.array([2*np.cos(t), 2*np.sin(t), 5 + np.sin(t)])
uav = UAV(movement_fn=path)

# Create controller
ctrl = SACController(quad, timestep=0.01)

# Create simulator
sim = Simulator(quad, ctrl, uav)
sim.simtime = (0, 20)
sim.timestep = 0.01
sim.epsilon = 0.1  # Capture distance

# Run simulation
z0 = np.zeros(12)
t, z, u, d, y = sim.simulate(z0)

# Animate
sim.animate(t, z, y)
```

## State Vector

The quadrotor state vector `z` has 12 elements:

| Index | Symbol | Description |
|-------|--------|-------------|
| 0-2   | x, y, z | Position in world frame (m) |
| 3-5   | φ, θ, ψ | Roll, pitch, yaw angles (rad) |
| 6-8   | vx, vy, vz | Linear velocity in world frame (m/s) |
| 9-11  | ω1, ω2, ω3 | Angular velocity in body frame (rad/s) |

## Control Inputs

The control input vector `u` has 4 elements representing the thrust from each rotor (N):

```
     O u1 (front)
     |
u2 O-|-O u4 (left/right)
     |
     O u3 (back)
```

## Files Structure

```
python/
├── README.md
├── requirements.txt
├── run_simulation.py          # Main simulation script
├── generate_html_report.py    # HTML report generator
└── src/
    ├── __init__.py
    ├── quadrotor.py          # Quadrotor dynamics class
    ├── uav.py                # Target UAV class
    ├── controller.py         # SAC and Hover controllers
    └── simulator.py          # Main simulator class
```

## Output

The simulation produces:
- **Animation**: Real-time 3D visualization with:
  - Blue quadrotor with trajectory trail
  - Red target UAV with trajectory trail
  - Silhouette projections for depth perception
- **State Plots**: Position, velocity, orientation, angular velocity vs time
- **Control Plots**: Rotor thrust inputs vs time
- **HTML Report** (optional): Interactive Plotly visualization

## Parameters

| Parameter | Symbol | Default | Description |
|-----------|--------|---------|-------------|
| Gravity | g | 9.81 m/s² | Gravitational acceleration |
| Arm length | l | 0.2 m | Distance from center to rotors |
| Mass | m | 0.5 kg | Total quadrotor mass |
| Max thrust | μ | 3.0 N | Maximum rotor thrust |
| Thrust-torque | σ | 0.01 | Thrust to torque coefficient |
| Inertia | I | diag([1.24, 1.24, 2.48]) kg·m² | Moment of inertia |

## Notes

- The LQR linearization uses SymPy for symbolic computation, which may take a few seconds on first run
- The control library is used for LQR gain computation via the Riccati equation
- Animation performance depends on your system; use `--no-animation` for slower machines

