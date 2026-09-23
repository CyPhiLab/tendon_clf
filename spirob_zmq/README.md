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
