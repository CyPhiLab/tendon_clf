"""EKF: MuJoCo prediction, mocap marker update, run once per measurement."""

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
        self.na = self.model.na
        if self.nq != self.nv:
            self.get_logger().warn(
                'nq != nv  the constant-velocity, covariance model used here assumes nq == nv.')

        # Motor ids, in the same order as MotorState.app_ctrl
        self.motor_ids = list(self.declare_parameter('motor_ids', [0, 1, 2]))

        # Sites used as EKF measurements
        site_names = self.declare_parameter(
            'site_names')
        self.site_ids = [self.model.site(name).id for name in site_names]
        self.ee_site_id = self.model.site('ee').id
        self.site_meas = 3 * len(self.site_ids)

        rest_state(self, self.model, self.data)

        h = self.declare_parameter('prediction_timestep')
        if h is not None:
            self.model.opt.timestep = h
        self.h = self.model.opt.timestep

        self.rate_hz = self.declare_parameter('rate_hz', 100.0)

        # EKF noise parameters (meters and meters/sec, per xyz component, iid Gaussian)
        pos_noise = self.declare_parameter('process_pos_noise', 1e-5)
        vel_noise = self.declare_parameter('process_vel_noise', 1e-5)
        act_noise = self.declare_parameter('process_act_noise', 1e-3)
        meas_noise = self.declare_parameter('measurement_noise', 1e-5)

        nx = self.nq + self.nv + self.na

        # Process noise covariance Q and measurement noise covariance R
        self.Q = np.diag([pos_noise ** 2] * self.nq + [vel_noise ** 2] * self.nv
                         + [act_noise ** 2] * self.na)
        self.R = (meas_noise ** 2) * np.eye(self.site_meas)

        # EKF state and covariance, x = [q; dq; act]
        self.x = self._get_state(self.data)
        self.P = np.eye(nx) * self.declare_parameter('initial_covariance', 1e-8)

        self.jacobian_every = self.declare_parameter('jacobian_every', 10)
        self.jacobian_thread = self.declare_parameter('jacobian_thread', True)
        self.fd_eps = self.declare_parameter('fd_eps', 1e-6)
        self.fd_centered = self.declare_parameter('fd_centered', False)
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
        if n == 0:
            return
        if self.n_updates % self.jacobian_every == 0:
            self.request_jacobian()
        F = np.linalg.matrix_power(self.F_step, n)

        self._set_state(self.data, self.x)
        self.data.ctrl[:] = self.app_u
        for _ in range(n):
            mujoco.mj_step(self.model, self.data)
        self.x = self._get_state(self.data)
        self.P = F @ self.P @ F.T + self.Q * (n * self.h * self.rate_hz)

    def update(self, z):
        nq, nv = self.nq, self.nv
        self._set_state(self.data, self.x)
        mujoco.mj_kinematics(self.model, self.data)
        mujoco.mj_comPos(self.model, self.data)
        z_pred = np.concatenate([self.data.site_xpos[site_id] for site_id in self.site_ids])

        H = np.zeros((self.site_meas, len(self.x)))
        jac = np.zeros((3, nv))
        for i, site_id in enumerate(self.site_ids):
            mujoco.mj_jacSite(self.model, self.data, jac, None, site_id)
            H[3 * i:3 * i + 3, :nq] = jac

        y = z - z_pred
        S = H @ self.P @ H.T + self.R
        K = np.linalg.solve(S, H @ self.P).T
        self.x = self.x + K @ y
        I_KH = np.eye(len(self.x)) - K @ H
        self.P = I_KH @ self.P @ I_KH.T + K @ self.R @ K.T
        self.n_updates += 1
        self.have_estimate = True

    def publish(self, stamp):
        nq, nv = self.nq, self.nv
        self._set_state(self.data, self.x)
        mujoco.mj_kinematics(self.model, self.data)
        mujoco.mj_comPos(self.model, self.data)
        jac = np.zeros((3, nv))
        mujoco.mj_jacSite(self.model, self.data, jac, None, self.ee_site_id)

        self.state_pub.publish({
            'stamp': stamp,
            'q': self.x[:nq],
            'dq': self.x[nq:nq + nv],
            'task_pos': self.data.site_xpos[self.ee_site_id],
            'task_vel': jac @ self.x[nq:nq + nv],
            'is_valid': self.have_estimate,
        })

    def _get_state(self, d):
        return np.concatenate([d.qpos, d.qvel, d.act])

    def _set_state(self, d, x):
        nq, nv = self.nq, self.nv
        d.qpos[:] = x[:nq]
        d.qvel[:] = x[nq:nq + nv]
        d.act[:] = x[nq + nv:]

    def request_jacobian(self):
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
        m, d = self.lin_model, self.lin_data
        self._set_state(d, x)
        d.ctrl[:] = u
        if warmstart is not None:
            d.qacc_warmstart[:] = warmstart
        mujoco.mj_forward(m, d)
        A = np.zeros((2 * m.nv + m.na, 2 * m.nv + m.na))
        mujoco.mjd_transitionFD(m, d, self.fd_eps, int(self.fd_centered), A, None, None, None)
        return A


def main(argv=None):
    run_node(StateEstimationNode, argv)


if __name__ == '__main__':
    main()
