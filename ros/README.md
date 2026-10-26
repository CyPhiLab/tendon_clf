## 1. Architecture: No EKF

**Dataflow:** `control_node` → `get_state_node`

* **`control_node`**: Subscribes to `robot_state` to compute control inputs and publishes them to `motor_command`.
* **`get_state_node`**: Subscribes to `motor_command` to receive control inputs, updates the robot state, and publishes the result to `robot_state`.

---

## 2. Architecture: EKF

**Dataflow:** `control_node` → `virtual_measurement_node` + `hardware_node` → `state_estimation_node` → `simulation_node`

* **`control_node`**: Subscribes to `robot_state` to compute control inputs and publishes them to `motor_command`.
* **`hardware_node`**: Subscribes to `motor_command` to convert inputs into motor commands and publishes feedback to `motor_state`. 
  > *Note:* If `dry_run` is enabled (no physical hardware connected), feedback control inputs match commanded inputs directly.
* **`virtual_measurement_node`**: Subscribes to `motor_command` to update predicted states $(q, \dot{q})$. Runs forward kinematics to obtain $(x, y, z)$ positions for five sites, then adds noise to simulate measurement uncertainty before publishing to `site_measurement`.
* **`state_estimation_node`**: Subscribes to `motor_command` (for next-step predictions) and `site_measurement` (for measured marker positions). Fuses predicted and measured states using an **Extended Kalman Filter (EKF)**, publishing the corrected result to `robot_state`.
* **`simulation_node`**: Subscribes to `robot_state` and visualizes the robot kinematics using the EKF-updated state.

---

## 3. Running the Program

### Build & Setup
```bash
cd ~/tendon_clf/ros
colcon build --symlink-install
source install/setup.bash
```

### Launch options
```bash
ros2 launch spirob_ros spirob.launch.py
```
```bash
ros2 launch spirob_ros spirob_no_ekf.launch.py
```

### Open-loop keyboard teleop
Drives the tendons directly from the keyboard (no controller, no state feedback) by publishing to `/spirob/motor_command`. Run it in its own terminal instead of `control_node`, alongside `hardware_node` (real robot) or `get_state_node` + `simulation_node` (sim):
```bash
ros2 run spirob_ros teleop_node
# optional: ros2 run spirob_ros teleop_node --ros-args -p step:=0.02 -p u_min:=-0.5
```
`w/s` bend toward ±x, `a/d` toward ±y, `r/f` add/remove pretension on all tendons (stiffens without bending), `space` releases everything, `q` releases and quits. Each key press nudges the command by `step`; tendon tensions are kept within `[u_min, u_max]`. If a direction looks wrong on the hardware, change `tendon_angles_deg` (the direction the tip bends when each tendon is pulled).
