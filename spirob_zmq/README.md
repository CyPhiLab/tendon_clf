# spirob_zmq

ZeroMQ version of the `ros/spirob_ros` nodes. Runs on Linux, macOS and Windows.

## Setup

```bash
cd spirob_zmq
git submodule update --init ../external/spirob_mujoco
pip install -r requirements.txt
```

## Run

```bash
cd ~/huy/tendon_clf
source spirob_zmq/.venv/bin/activate
python -m spirob_zmq.launch                   # simulated plant + EKF + controller + viewer
python -m spirob_zmq.launch --mode no_ekf     # controller on the true state
python -m spirob_zmq.launch --headless --duration 20
python -m spirob_zmq.launch -p control_node.target_pos=[-0.3,0.0,0.25] -p timing_report_s=5
```

`-p key=value` sets a parameter on every node, `-p node.key=value` on one node.
Defaults are in `robots.py`.

The control input `u` (and `app_ctrl`) is the **motor current in amps**. The model's actuators are
ideal current sources (`<motor gear="31.75">`, the AK drives in current mode), with the 24 V bus
limit applied in code (`common.VoltageLimit`, `common.step_current`).

The estimator starts from the pose fitted to the markers (`INITIAL_STATE`), not from the keyframe,
so runs can start from any pose. In simulation, `-p virtual_measurement_node.init_pose=random`
starts the plant curled.

## Tools

```bash
python -m spirob_zmq.topic echo /spirob/robot_state
python -m spirob_zmq.topic hz /spirob/robot_state
python -m spirob_zmq.topic record run.jsonl
python -m spirob_zmq.lockstep --duration 5 --out run.jsonl
python -m spirob_zmq.ekf_report run.jsonl --plot ekf.png   # includes estimator errors and, if recorded, motor current feedback vs commanded current
python -m spirob_zmq.replay_hw ekf_results/real/<run>.jsonl --plot replay.png   # model vs recorded motor angles, driven by the recorded currents
python tests/test_startup.py && python tests/test_real_measurement_node.py      # startup and marker tracking, offline
```

## Hardware

```bash
python -m spirob_zmq.calibrate_frame                     # robot at rest: Vicon -> MuJoCo frame, rerun if the Vicon origin or base moves
python -m spirob_zmq.launch --mode real_ekf -p hardware_node.dry_run=false
```

`real_measurement_node` waits until the markers are still, maps them into the MuJoCo frame (from
`base_marker_sites` if static base markers are set up, else `ekf_results/vicon_frame.json`), labels
them by fitting the pose, and then tracks them frame to frame; a segment lost entirely is found again
by its triangle's shape and the chain geometry. Markers on the robot that the estimator does not use
must be listed in `unpublished_sites` (robots.py): they are tracked but not published, and left
untracked a neighbouring triangle is taken for a measured one. `-p hardware_node.pretension_current=-0.1`
holds a small current until the controller starts, to take up tendon slack.
