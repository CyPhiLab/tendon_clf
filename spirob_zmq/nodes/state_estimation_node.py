"""EKF fusing applied motor control (prediction) with site positions (update).

Measurement-driven: each /spirob/site_measurement triggers predict + update +
publish. The prediction spans the time since the previous measurement, taken
from the measurements' ``stamp`` (the time the measurement refers to; the
virtual plant stamps its simulated time), so the estimate stays aligned with
the measurements however the processes are scheduled, and a dropped
measurement just means a longer prediction.

Everything model-related comes from MuJoCo: the prediction is ``mj_step`` at
``prediction_timestep``, and the covariance is propagated with
``mjd_transitionFD``. A finite-difference Jacobian costs ~30 ms on the
horizontal arm (every position perturbation re-runs collision), but the
estimate is insensitive to how fresh it is (lockstep: refreshing every update,
every 50, or never all give ~0.02 mm ee error), so it is refreshed every
``jacobian_every`` updates on a background thread (MuJoCo releases the GIL),
and predictions use the latest one available.
"""

import copy
import threading

import mujoco
import numpy as np

from spirob_zmq.common import (MOTOR_STATE, ROBOT_STATE, SITE_MEASUREMENT, load_model,
                               rest_state)
from spirob_zmq.core import Node, run_node


class StateEstimationNode(Node):
    def __init__(self, params=None):
        super().__init__('state_estimation_node', params)

        # Mujoco model used for prediction and linearization
        self.model = load_model(self)
        self.data = mujoco.MjData(self.model)
        self.nq = self.model.nq
        self.nv = self.model.nv
        if self.nq != self.nv:
            self.get_logger().warn(
                'nq != nv  the constant-velocity, covariance model used here assumes nq == nv.')

        # Motor ids, in the same order as MotorState.app_ctrl
        self.motor_ids = list(self.declare_parameter('motor_ids', [0, 1, 2]))

        # Sites used as EKF measurements
        site_names = self.declare_parameter(
            'site_names', ['ee_seg5', 'ee_seg10', 'ee_seg15', 'ee_seg20', 'ee'])
        self.site_ids = [self.model.site(name).id for name in site_names]
        self.ee_site_id = self.model.site('ee').id
        self.site_meas = 3 * len(self.site_ids)

        # The robot starts at rest in the same state as the plant (straight, or
        # gravity-settled), computed at the model's own timestep
        rest_state(self, self.model, self.data)

        # Prediction timestep. Defaults to the model's; a coarser step makes
        # each prediction cheaper at some cost in accuracy.
        h = self.declare_parameter('prediction_timestep')
        if h is not None:
            self.model.opt.timestep = h
        self.h = self.model.opt.timestep

        # Nominal measurement rate; process noise is specified per 1/rate_hz
        self.rate_hz = self.declare_parameter('rate_hz', 100.0)

        # EKF noise parameters (meters and meters/sec, per xyz component, iid Gaussian)
        pos_noise = self.declare_parameter('process_pos_noise', 1e-5)
        vel_noise = self.declare_parameter('process_vel_noise', 1e-5)
        meas_noise = self.declare_parameter('measurement_noise', 1e-5)

        nx = self.nq + self.nv

        # Process noise covariance Q (per nominal period) and measurement noise covariance R
        self.Q = np.diag([pos_noise ** 2] * self.nq + [vel_noise ** 2] * self.nv)
        self.R = (meas_noise ** 2) * np.eye(self.site_meas)

        # EKF state and covariance, x = [q; dq]
        self.x = np.concatenate([self.data.qpos, self.data.qvel])
        self.P = np.eye(nx) * self.declare_parameter('initial_covariance', 1e-8)

        # Linearization of one prediction step, refreshed every jacobian_every
        # updates; on a background thread unless jacobian_thread is false
        # (lockstep turns it off so runs are deterministic)
        self.jacobian_every = self.declare_parameter('jacobian_every', 10)
        self.jacobian_thread = self.declare_parameter('jacobian_thread', True)
        self.fd_eps = self.declare_parameter('fd_eps', 1e-6)
        self.fd_centered = self.declare_parameter('fd_centered', False)
        # The linearization gets its own model and data, so a background
        # refresh never shares mutable MuJoCo state with the prediction
        self.lin_model = copy.deepcopy(self.model)
        self.lin_data = mujoco.MjData(self.lin_model)
        self.F_step = self.discrete_jacobian(self.x, np.zeros(self.model.nu))
        self.n_updates = 0
        self._jac_request = None
        self._jac_wakeup = threading.Condition()
        self._jac_worker = None
        if self.jacobian_thread:
            self._jac_worker = threading.Thread(target=self._jacobian_loop, daemon=True)
            self._jac_worker.start()

        # Latest applied control (from hardware_node), defaults to zero until the first MotorState arrives.
        self.app_u = np.zeros(self.model.nu)

        self.last_stamp = None
        self.have_estimate = False

        # Subscriptions
        self.create_subscription(MOTOR_STATE, self._on_command)
        self.create_subscription(SITE_MEASUREMENT, self._on_measurement)

        # Publisher
        self.state_pub = self.create_publisher(ROBOT_STATE)

    def _on_command(self, msg):
        motor_id = msg['motor_id']
        if motor_id not in self.motor_ids:
            return
        i = self.motor_ids.index(motor_id)
        self.app_u[i] = msg['app_ctrl']

    def _on_measurement(self, msg):
        z = np.array(msg['data'], dtype=float)
        if z.shape[0] != self.site_meas:
            self.get_logger().warn(
                f'site_measurement has {z.shape[0]} entries, expected {self.site_meas}; ignoring')
            return
        stamp = float(msg['stamp'])
        if self.last_stamp is None:
            # First measurement: the robot is at rest, nothing to predict yet
            n = 0
        else:
            dt = stamp - self.last_stamp
            if dt <= 0.0:
                self.get_logger().warn(f'measurement stamp went back by {-dt:.4f} s; ignoring',
                                       throttle_duration_sec=1.0)
                return
            n = max(1, int(round(dt / self.h)))
        self.last_stamp = stamp
        self.predict(n)
        self.update(z)
        self.publish(stamp)

    def predict(self, n):
        """Propagate x and P over n prediction steps with the latest applied control."""
        if n == 0:
            return
        nq = self.nq
        if self.n_updates % self.jacobian_every == 0:
            self.request_jacobian()
        F = np.linalg.matrix_power(self.F_step, n)

        self.data.qpos[:] = self.x[:nq]
        self.data.qvel[:] = self.x[nq:]
        self.data.ctrl[:] = self.app_u
        for _ in range(n):
            mujoco.mj_step(self.model, self.data)
        self.x = np.concatenate([self.data.qpos, self.data.qvel])
        self.P = F @ self.P @ F.T + self.Q * (n * self.h * self.rate_hz)

    def update(self, z):
        nq, nv = self.nq, self.nv
        # Kinematics at the predicted state (mj_step leaves xpos at the pre-step state)
        self.data.qpos[:] = self.x[:nq]
        mujoco.mj_kinematics(self.model, self.data)
        mujoco.mj_comPos(self.model, self.data)
        z_pred = np.concatenate([self.data.site_xpos[site_id] for site_id in self.site_ids])

        H = np.zeros((self.site_meas, nq + nv))
        jac = np.zeros((3, nv))
        for i, site_id in enumerate(self.site_ids):
            mujoco.mj_jacSite(self.model, self.data, jac, None, site_id)
            H[3 * i:3 * i + 3, :nq] = jac

        y = z - z_pred
        S = H @ self.P @ H.T + self.R
        # K = P H^T S^-1, via a solve instead of an explicit inverse (S is symmetric)
        K = np.linalg.solve(S, H @ self.P).T
        self.x = self.x + K @ y
        # Joseph form keeps P symmetric positive semi-definite despite round-off
        I_KH = np.eye(nq + nv) - K @ H
        self.P = I_KH @ self.P @ I_KH.T + K @ self.R @ K.T
        self.n_updates += 1
        self.have_estimate = True

    def publish(self, stamp):
        nq, nv = self.nq, self.nv
        self.data.qpos[:] = self.x[:nq]
        self.data.qvel[:] = self.x[nq:]
        mujoco.mj_kinematics(self.model, self.data)
        mujoco.mj_comPos(self.model, self.data)
        jac = np.zeros((3, nv))
        mujoco.mj_jacSite(self.model, self.data, jac, None, self.ee_site_id)

        # Stamped with the measurement's time: this is the estimate at that time
        self.state_pub.publish({
            'stamp': stamp,
            'q': self.x[:nq],
            'dq': self.x[nq:],
            'task_pos': self.data.site_xpos[self.ee_site_id],
            'task_vel': jac @ self.x[nq:],
            'is_valid': self.have_estimate,
        })

    def request_jacobian(self):
        """Refresh F_step at the current estimate: inline, or by handing a
        snapshot to the worker (dropped if it is still busy with the last one)."""
        snapshot = (self.x.copy(), self.app_u.copy(), self.data.qacc_warmstart.copy())
        if self._jac_worker is None:
            self.F_step = self.discrete_jacobian(*snapshot)
            return
        with self._jac_wakeup:
            if self._jac_request is None:
                self._jac_request = snapshot
                self._jac_wakeup.notify()

    def _jacobian_loop(self):
        while True:
            with self._jac_wakeup:
                while self._jac_request is None and self.ok():
                    self._jac_wakeup.wait()
                if not self.ok():
                    return
                snapshot = self._jac_request
            self.F_step = self.discrete_jacobian(*snapshot)
            with self._jac_wakeup:
                self._jac_request = None

    def destroy_node(self):
        super().destroy_node()
        if self._jac_worker is not None:
            with self._jac_wakeup:
                self._jac_wakeup.notify()
            self._jac_worker.join(timeout=1.0)

    def discrete_jacobian(self, x, u, warmstart=None):
        """d x_{k+1} / d x_k of one prediction step at (x, u), from MuJoCo's
        mjd_transitionFD (finite differences of mj_step, constraints included).
        Uses its own MjData so the prediction's solver warmstart is untouched."""
        m, d = self.lin_model, self.lin_data
        d.qpos[:] = x[:self.nq]
        d.qvel[:] = x[self.nq:]
        d.ctrl[:] = u
        if warmstart is not None:
            d.qacc_warmstart[:] = warmstart
        mujoco.mj_forward(m, d)
        A = np.zeros((2 * m.nv, 2 * m.nv))
        mujoco.mjd_transitionFD(m, d, self.fd_eps, int(self.fd_centered), A, None, None, None)
        return A


def main(argv=None):
    run_node(StateEstimationNode, argv)


if __name__ == '__main__':
    main()
