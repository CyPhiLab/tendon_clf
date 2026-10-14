# Tendon Control & Learning Framework

**A robotics control research codebase for comparing control schemes across soft robotic systems using MuJoCo simulation.**

This repository implements and compares different controllers, including a novel Soft ID-CLF-QP, across 3 soft robot models (helix, spirob_horz, tendon) for 2 experiment types (set-point tracking, trajectory following).

## Quick Start

### Prerequisites

- Python 3.8+
- MuJoCo 3.10+
- CVXPY optimization library

### Installation

1. **Clone the repository:**
   ```bash
   git clone --recurse-submodules <repository-url>
   cd tendon_clf
   # or, in an existing clone: git submodule update --init external/spirob_mujoco
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

**Real-time stack:**
```bash
python -m spirob_zmq.launch
```
See [`spirob_zmq/README.md`](spirob_zmq/README.md).

## Architecture

### Controller Implementations

- **CLF-QP**: Standard implementation
- **Soft ID-CLF-QP**: Inverse Dynamics with Control Lyapunov Functions using quadratic programming
- **Impedance Control**: Traditional Cartesian impedance control
- **Impedance-QP**: Impedance control with QP-based actuator limit handling

### Robot Models

- **helix**: A high DOF soft-rigid robot inspired by wave springs. Based on https://ieeexplore.ieee.org/abstract/document/11020854
- **tendon**: 4-DOF tendon-driven finger
- **spirob_horz**: A high DOF soft-rigid robot whose shape follows a logarithmic spiral (based on https://www.sciencedirect.com/science/article/pii/S2666998624006033), mounted horizontally, with dcmotor-driven
  tendons instead of ideal force actuators. Its base is raised programmatically so
  the arm hangs in free space; see `SPIROB_HORZ_NOTES.md` for the port details.

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

## Hardware Debugging

### PCAN CAN Bus Setup (AK Motor Control)

If using PCAN USB adapters for motor control, follow these steps to verify and troubleshoot the setup:

1. **Check PCANBasic library installation:**
   ```bash
   find /usr /usr/local /opt -iname 'libpcanbasic.so*' 2>/dev/null
   ```

2. **Verify PCANBasic Python binding:**
   ```bash
   python3 -c "from can.interfaces.pcan.basic import PCANBasic; print('pcanbasic ok')"
   python3 -c "from can.interfaces.pcan.basic import PCANBasic; PCANBasic(); print('init ok')"
   ```

3. **Confirm PCAN device is connected:**
   ```bash
   lsusb | grep -i -E 'peak|pcan'
   ```

4. **Check PCAN driver installation:**
   ```bash
   sudo find / -type f \( -name 'pcan.ko' -o -name 'pcan.ko.*' \) 2>/dev/null
   ```

5. **Test CAN bus opening (try different channels if needed: PCAN_USBBUS1, PCAN_USBBUS2, etc.):**
   ```bash
   python3 -c "import can; b=can.Bus(interface='pcan', channel='PCAN_USBBUS1', bitrate=1000000); print('open ok'); b.shutdown()"
   ```

6. **If driver fails to load:** Check that the PCAN driver kernel version matches the current running kernel:
   ```bash
   uname -r
   ```
   If kernel versions differ, rebuild the PCAN driver:
   ```bash
   # 1) Install build tools
   sudo apt update
   sudo apt install -y build-essential linux-headers-$(uname -r)
   
   # 2) Rebuild PEAK driver (adjust path to your PEAK driver source)
   cd ~/peak-linux-driver-9.2.0
   make clean && make
   sudo make install
   
   # 3) Refresh module map and load
   sudo depmod -a
   sudo modprobe pcan
   
   # 4) Verify driver is loaded
   modinfo pcan | grep -E 'filename|vermagic'
   lsmod | grep pcan
   cat /proc/pcan
   ```

7. **Test motor control:**
   ```bash
   python3 motor_control/ak_motor/test.py --motor_id 10 --command current --value 0.5 --duration 2
   ```
