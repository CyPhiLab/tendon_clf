# spirob_zmq — ZeroMQ port of `ros/spirob_ros`

Same nodes, topics, and message fields as the ROS 2 package, but it needs only
`pyzmq`, so it runs on Linux, macOS, and Windows with no ROS install.
The node code mirrors the ROS nodes line for line. Only the plumbing changed.

## Install

```bash
git submodule update --init external/spirob_mujoco   # the robot model
pip install -r requirements.txt                      # includes pyzmq
```

Tested with mujoco 3.14. It needs 3.10 or newer: it uses `MjSpec`, `mj_jacDot`,
and the 3.10 `mj_fullM` signature.

## Robots

`-p robot=...` selects a profile from `robots.py`. The profile sets the model,
start pose, target, ctrl limits and controller gains; any `-p` override wins.

| robot | model | notes |
|---|---|---|
| `spirob_horz` (default) | `external/spirob_mujoco` via `mujoco_models/spirob/spirob_horz_control.xml` | dcmotor (ctrl in volts, back-EMF, 12 N m limit), implicitfast at 1 ms, base raised to 0.55 m, starts gravity-settled. Settings follow SPIROB_HORZ_NOTES.md on `claude/port-progress-shc9sv`. |
| `spirob` | `mujoco_models/spirob/spirob_control.xml` | the original vertical model and ROS controller objective |

Nodes step the model at its own timestep, several substeps per tick, instead
of overwriting it with 1/rate_hz. On the horizontal arm a step costs ~0.6 ms,
~60% of it mesh-mesh collision between adjacent segments (the model disables
`filterparent`), so the simulated plant runs at ~0.6x real time on a 4-core
machine. Use `lockstep` for accurate evaluation.

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

The EKF is measurement-driven: each `site_measurement` triggers predict,
update and publish over the interval since the previous measurement's `stamp`
(the virtual plant stamps its simulated time). The prediction is `mj_step` at
`prediction_timestep` (default: the model's), and the covariance uses
`mjd_transitionFD`, refreshed every `jacobian_every` updates (default 10) on a
background thread (`jacobian_thread`, off in lockstep for determinism). About
7 ms per measurement on the horizontal arm, almost all of it the 10 `mj_step`s.

The controller eliminates u through the ID constraint and solves a dense
49-variable QP with DAQP: ~0.1 ms solve, ~1.3 ms tick (0.6 ms of that is
`mj_forward`). `hardware_node` maps ctrl to motor current through the same
actuator law, clamped to `max_current`. **Its non-dry-run path (feedback units,
pole pairs, sign convention) has not been checked on hardware yet.**

### Differences from the ROS nodes (bug fixes)

- The EKF and virtual measurement refresh kinematics after `mj_step`, so site
  positions and Jacobians belong to the new state rather than the previous one.
- All nodes take stiffness, damping, and gravity from one shared place
  (`joint_stiffness`, `joint_damping`, `gravity` params). Before this fix the
  controller used 0.3 stiffness while the plant and EKF used 0.
- The virtual plant is driven by `/spirob/motor_state` (the applied input, which
  is also what the EKF uses) instead of `/spirob/motor_command`.
- EKF update uses a linear solve and the Joseph-form covariance update, and is
  driven by measurement stamps instead of its own timer.

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
