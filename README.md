# Tendon Control & Learning Framework

**A robotics control research codebase for comparing control schemes across soft robotic systems using MuJoCo simulation.**

This repository implements and compares different controllers, including a novel Soft ID-CLF-QP, across 3 soft robot models (helix, spirob, tendon) for 2 experiment types (set-point tracking, trajectory following).

## Quick Start

### Prerequisites

- Python 3.8+
- MuJoCo 3.0+
- CVXPY optimization library

### Installation

1. **Clone the repository:**
   ```bash
   git clone <repository-url>
   cd tendon_clf
   ```

2. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

### Basic Usage

**Single experiment:**
```bash
# Run ID-CLF-QP controller on helix robot with set-point tracking
python run.py --robot helix --control id_clf_qp --experiment set --target_pos pos1 

# Run impedance controller on tendon robot with trajectory tracking
python run.py --robot tendon --control impedance --experiment tracking --headless
```

**Batch experiments:**
```bash
# Run all combinations (configurable)
python run_all.py
```

## Architecture

### Controller Implementations

- **CLF-QP**: Standard implementation
- **Soft ID-CLF-QP**: Inverse Dynamics with Control Lyapunov Functions using quadratic programming
- **Impedance Control**: Traditional Cartesian impedance control
- **Impedance-QP**: Impedance control with QP-based actuator limit handling

### Robot Models

- **helix**: A high DOF soft-rigid robot inspired by wave springs. Based on https://ieeexplore.ieee.org/abstract/document/11020854
- **tendon**: 4-DOF tendon-driven finger
- **spirob**: A high DOF soft-rigid robot who's shape follows a logarithmic spiral. Based on https://www.sciencedirect.com/science/article/pii/S2666998624006033

### Experiment Types

- **Set-point tracking**: Move end-effector to target positions (pos1-pos4)
- **Trajectory tracking**: Follow circular trajectories in task space

## Development

### Code Organization

- `run.py` - Single experiment entry point
- `utils.py` - Simulation loop and utilities
- `robot.py` - Robot class with physics interface
- `controllers/` - All controller implementations
- `mujoco_models/` - Robot XML definitions and meshes
- `plot.py` - Analysis and visualization tools

### Adding New Controllers

1. Inherit from `BaseController` in `controllers/base.py`
2. Implement `__call__(robot, experiment, target, target_vel, target_acc, twist, previous_solution)`
3. Use `robot.get_*()` methods for physics data (simulator-agnostic)
4. Return `ControllerResult` with task error and control input

### Adding New Robots

1. Create MuJoCo XML model in `mujoco_models/{robot_name}/`
2. Add robot configuration in `robot.py` initialization
3. Update choices in `run.py` argument parser

