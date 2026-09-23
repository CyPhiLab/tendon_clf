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
