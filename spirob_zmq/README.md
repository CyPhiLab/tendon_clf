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

The Spirob viewer starts side-on to show the curl in the Y-Z plane. Startup
angles can be overridden with `-p simulation_node.viewer_azimuth=0` and
`-p simulation_node.viewer_elevation=-20` (degrees). These are applied only
when the viewer opens; mouse rotation remains available during the run.
`viewer_pan_right=0.08` shifts the startup framing so the robot appears farther
right (world-space pan distance in meters). Override with
`-p simulation_node.viewer_pan_right=0` to disable the offset.
The configured `site_name` is shown in green in the viewer (the red sphere is
the target). Only the viewer's copy of the site appearance and visibility is
changed; the site's position and physics are unchanged. If overriding
`control_node.site_name` alone, also set `simulation_node.site_name` to match,
or use the shared `-p site_name=...` override for both.

The control input `u` (and `app_ctrl`) is the **motor current in amps**. The model's actuators are
ideal current sources (`<motor gear="31.75">`, the AK drives in current mode), with the 24 V bus
limit applied in code (`common.VoltageLimit`, `common.step_current`).

The estimator starts from the pose fitted to the markers (`INITIAL_STATE`), not from the keyframe,
so runs can start from any pose. In simulation, `-p virtual_measurement_node.init_pose=random`
starts the plant curled.

### EKF covariance

For the current model, `x = [q, dq]`. Startup (including `init_from=keyframe`),
measured-pose initialization, and divergence resets all use
`P = diag([init_q_std**2] * nq + [init_dq_std**2] * nv)`.
The defaults are `init_q_std=0.1` rad and `init_dq_std=0.001` rad/s.

Process noise comes from independent motor-current errors with
`process_current_std=0.05` A (one standard deviation, not a hard bound).
`discrete_jacobian` returns both discrete state and input Jacobians `(A, B)`
from MuJoCo's `mjd_transitionFD`. No additional timestep factor is needed.
The same input error is held throughout each prediction interval between
measurements, and errors are independent between intervals:

```text
Sigma_u = process_current_std**2 * I
G_0 = 0
G_(j+1) = A @ G_j + B
Q_interval = G_n @ Sigma_u @ G_n.T
P_pred = A**n @ P @ (A**n).T + Q_interval
```

On top of that, `process_q_std` (rad/sqrt(s), default 0) and `process_dq_std` (rad/s/sqrt(s);
1.0 in `robots.py`) add independent noise on every joint angle and velocity,
`Q_interval += diag(process_q_std**2 * dt, process_dq_std**2 * dt)` with `dt = n * timestep`.
The motor currents alone give a rank-3 `Q`: the covariance then collapses and the filter follows the
model instead of the markers. On hardware the model drifts away from a robot that friction holds
still; the velocity noise lets the updates stop that drift. `measurement_noise` (5 mm in `robots.py`)
has to cover the marker error that no pose of the model removes (about 5 mm on the 20261103 runs),
not just the Vicon noise; below that the filter chases it and the estimate jitters.

`joint_frictionloss` (N m, not set by default) overrides the model's joint friction, for example
`-p state_estimation_node.joint_frictionloss=0.02` for the estimator only. Replays of the 20261103
runs with 0.02 cut the estimate's marker error at startup and its velocity while the robot is still by more than half.

The estimator subscribes to `SITE_MEASUREMENT` latest-only: if an update takes longer than the
measurement interval, stale measurements are skipped (the prediction spans the gap) rather than
queued. Queued, the estimate fell further behind every update; that was why `jacobian_every=1`
(about 20 ms per update on the lab PC, Vicon frames every 21 ms) ran badly. `jacobian_every`
now only trades CPU for linearization freshness; the estimate barely depends on it.

Here `A**n` denotes a matrix power. The latest matched `(A, B)` pair is held
fixed over the interval, retaining the existing Jacobian refresh cadence.
The nominal state still uses MuJoCo and the existing voltage-limited current
stepping. The Jacobians describe MuJoCo's transition at nominal `u`, not the
derivative of the external voltage limiter.

The acceleration-relative model and its floor are removed; `process_accel_relative_std`,
`process_accel_std_min`, `process_act_noise`, `process_pos_noise`, `process_vel_noise`,
and `initial_covariance` are no longer used (`process_q_std` and `process_dq_std` replace the joint terms). Models with activation states
(`x = [q, dq, act]`) use `init_act_std=0.0` for their initial activation uncertainty;
their process uncertainty also comes through `B`.

Measurement corrections are capped independently for each joint:
`abs(q_corrected[i] - q_predicted[i]) <= max_joint_q_correction` (default `0.2` rad).
The existing optional `max_q_correction` L2 block limit still applies first.
The affected gain rows are scaled before the Joseph covariance update, and
covariance symmetry is enforced after prediction and correction.
Every update publishes `correction_q_capped_count` and zero-based
`correction_q_capped_indices` on `/spirob/ekf_diagnostics`, together with the other
prediction/update diagnostics. Every correction update logs its update number,
capped-joint count (including zero), total joint count, and cap in the terminal.

## Tools

The controller publishes `/spirob/tracking_state` for its configured `site_name`:
one sample per processed valid state and one per marker measurement. Records
contain the original sample timestamp, source (`estimated`, `vicon`, or
`simulated`), marker name, MuJoCo-frame XYZ position in meters, target position,
and error (`position - target`) with its norm. Missing marker observations are
recorded as invalid with null position/error values, not filled from the estimate.
Estimated samples use forward kinematics of the state received by the controller;
they are available even during its startup delay.

Normal launch recording includes this topic in `ekf_results/{real,sim}/<run>.jsonl`.
After shutdown, `ekf_report` also writes `<run>_tracking.csv` next to that JSONL,
with separate rows per source, sample/recording timestamps, elapsed seconds,
XYZ/target/error columns, and validity. Sources are not interpolated or paired.
The tracking plot uses the recorded target and site metadata for these new runs.
Older runs still use the current profile for plotting and do not produce this CSV.
Recording must be enabled; interrupted postprocessing can be rerun with
`python -m spirob_zmq.ekf_report <run>.jsonl --plot <run>.png`.

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
python -m spirob_zmq.launch --mode real_ekf -p hardware_node.dry_run=false --duration 20
```

`real_measurement_node` waits until the markers are still, maps them into the MuJoCo frame (from
`base_marker_sites` if static base markers are set up, else `ekf_results/vicon_frame.json`), labels
them by fitting the pose, and then tracks them frame to frame; a segment lost entirely is found again
by its triangle's shape and the chain geometry. Markers on the robot that the estimator does not use
must be listed in `unpublished_sites` (robots.py): they are tracked but not published, and left
untracked a neighbouring triangle is taken for a measured one. `-p hardware_node.pretension_current=-0.1`
holds a small current until the controller starts, to take up tendon slack.

Open-loop keyboard teleop (no estimator or controller): `q/a`, `w/s`, `e/d` pull more/less on
motors 1-3, `+/-` change the step, space releases all, `x` quits (and sends 0 A).

```bash
python -m spirob_zmq.launch --mode teleop                                     # simulated plant + viewer
python -m spirob_zmq.launch --mode real_teleop --no-record                     # drives the motors
python -m spirob_zmq.launch --mode real_teleop -p hardware_node.dry_run=true    # no CAN bus
python -m spirob_zmq.launch --mode real_teleop -p teleop_node.current_limit=1.0
```

`real_teleop` defaults to `hardware_node.dry_run=false`. Currents are clamped to
`[-current_limit, 0]` (default 1.5 A; negative pulls).
