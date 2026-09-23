# spirob_zmq — ZeroMQ port of `ros/spirob_ros`

Same nodes, topics, and message fields as the ROS 2 package, but it needs only
`pyzmq`, so it runs on Linux, macOS, and Windows with no ROS install.
The node code mirrors the ROS nodes line for line. Only the plumbing changed.

## Install

```bash
pip install -r requirements.txt   # includes pyzmq
```

## Run (from the repo root)

```bash
python -m spirob_zmq.launch                    # EKF architecture  (= spirob.launch.py)
python -m spirob_zmq.launch --mode no_ekf      # no EKF            (= spirob_no_ekf.launch.py)
python -m spirob_zmq.launch --headless --duration 20   # no viewer, stop after 20 s
```

On macOS the launcher runs the viewer node under `mjpython`, which MuJoCo's
passive viewer requires. `mjpython` ships with `pip install mujoco`.

Parameters (the equivalent of ROS `parameters=[...]`):

```bash
python -m spirob_zmq.launch -p noise_std=1e-4                 # every node
python -m spirob_zmq.launch -p control_node.K=300 -p hardware_node.dry_run=false
python -m spirob_zmq.launch --config params.json   # {"model_path": "...", "control_node": {"K": 300}}
```

`model_path` defaults to `mujoco_models/spirob/spirob_control.xml` in this repo,
so the hard-coded `/home/zach/...` path from the ROS launch files is no longer needed.

You can also run any node on its own, for example:

```bash
python -m spirob_zmq.broker &
python -m spirob_zmq.nodes.state_estimation_node -p rate_hz=50
```

## Inspect topics (the equivalent of `ros2 topic`)

```bash
python -m spirob_zmq.topic echo /spirob/robot_state
python -m spirob_zmq.topic hz   /spirob/robot_state
python -m spirob_zmq.topic record run.jsonl          # all topics -> JSON lines
```

## Evaluating the estimator

`virtual_measurement_node` also publishes the plant's actual state on
`/spirob/true_state`. Two tools use it:

```bash
# Deterministic run in simulated time (no sockets, reproducible, no viewer)
python -m spirob_zmq.lockstep --duration 3 --out run.jsonl -p noise_std=1e-4 -p measurement_noise=1e-4
# Score estimate vs truth (also works on `topic record` output from a real-time run)
python -m spirob_zmq.ekf_report run.jsonl --plot ekf.png
```

EKF parameters of note: `jacobian=analytic|fd` (analytic ~1 ms and the default;
`fd` is the original `mjd_transitionFD` path at ~20 ms) and `position_jacobian_every`.

### Differences from the ROS nodes (bug fixes)

- The EKF and virtual measurement refresh kinematics after `mj_step`, so site
  positions and Jacobians belong to the new state rather than the previous one.
- All nodes take stiffness, damping, and gravity from one shared place
  (`joint_stiffness`, `joint_damping`, `gravity` params). Before this fix the
  controller used 0.3 stiffness while the plant and EKF used 0.
- The virtual plant is driven by `/spirob/motor_state` (the applied input, which
  is also what the EKF uses) instead of `/spirob/motor_command`.
- EKF update uses a linear solve and the Joseph-form covariance update.

## How it maps to ROS

| ROS 2                              | spirob_zmq                                      |
|------------------------------------|-------------------------------------------------|
| DDS discovery                      | `broker.py` (XSUB/XPUB forwarder on tcp 5555/5556) |
| `rclpy.Node`, timers, subscriptions | `core.Node` (single-threaded `spin()`, like rclpy) |
| `spirob_interfaces/msg/*`          | JSON dicts with the same field names (`header.stamp` → `stamp`, unix seconds) |
| subscription queue depth 1         | `create_subscription(..., latest_only=True)`    |
| `ros2 launch`                      | `python -m spirob_zmq.launch`                   |

To run across machines, start the broker on one host and set
`SPIROB_ZMQ_PUB=tcp://<host>:5555` and `SPIROB_ZMQ_SUB=tcp://<host>:5556` on every machine.
