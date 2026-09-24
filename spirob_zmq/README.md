# spirob_zmq

ZeroMQ version of the `ros/spirob_ros` nodes. Runs on Linux, macOS and Windows.

## Setup

```bash
git submodule update --init external/spirob_mujoco
pip install -r requirements.txt
```

## Run

```bash
python -m spirob_zmq.launch                   # simulated plant + EKF + controller + viewer
python -m spirob_zmq.launch --mode no_ekf     # controller on the true state
python -m spirob_zmq.launch --headless --duration 20
python -m spirob_zmq.launch -p control_node.target_pos=[-0.3,0.0,0.6] -p timing_report_s=5
```

`-p key=value` sets a parameter on every node, `-p node.key=value` on one node.
Defaults are in `robots.py`.

## Tools

```bash
python -m spirob_zmq.topic echo /spirob/robot_state
python -m spirob_zmq.topic hz /spirob/robot_state
python -m spirob_zmq.topic record run.jsonl
python -m spirob_zmq.lockstep --duration 5 --out run.jsonl
python -m spirob_zmq.ekf_report run.jsonl --plot ekf.png
```

## Hardware

`-p hardware_node.dry_run=false`. Not yet tested on hardware.
