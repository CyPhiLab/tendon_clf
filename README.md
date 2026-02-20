# Tendon Control & Learning Framework

**A robotics control research codebase for comparing control schemes across soft robotic systems using MuJoCo simulation.**

This repository implements and compares 4 different controllers (ID-CLF-QP, Impedance Control, Impedance-QP, MPC) across 3 robot models (helix, spirob, tendon) for 2 experiment types (set-point tracking, trajectory following).

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
python run.py --robot helix --control id_clf_qp --experiment set --target_pos pos1 --headless

# Run MPC controller on tendon robot with trajectory tracking
python run.py --robot tendon --control mpc --experiment tracking --headless
```

**Batch experiments:**
```bash
# Run all combinations (configurable)
python run_all.py
```

**Generate analysis plots:**
```bash
python plot.py  # Creates publication-ready figures and CSV reports
```

## Architecture

### Controller Implementations

- **ID-CLF-QP**: Inverse Dynamics with Control Lyapunov Functions using quadratic programming
- **Impedance**: Traditional Cartesian impedance control
- **Impedance-QP**: Impedance control with QP-based actuator limit handling
- **MPC**: Model Predictive Control with receding horizon optimization

### Robot Models

- **helix**: 6-DOF helical soft robot
- **tendon**: 6-DOF tendon-driven finger
- **spirob**: 3-DOF logarithmic spiral robot

### Experiment Types

- **Set-point tracking**: Move end-effector to target positions (pos1-pos4)
- **Trajectory tracking**: Follow circular trajectories in task space

## Citation

If you use this codebase in your research, please cite:

```bibtex
@software{tendon_clf,
  title={Tendon Control \& Learning Framework},
  author={[Author Names]},
  year={2026},
  url={[Repository URL]}
}
```

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


### Performance Tips

- Use `--headless` flag for speedup
- Reduce logging frequency for very long simulations

## License

[License information]

## Contributing

Contributions are welcome! Please:
1. Fork the repository
2. Create a feature branch
3. Add tests for new functionality
4. Submit a pull request

---

**For questions or support, please open an issue or contact [contact information].**