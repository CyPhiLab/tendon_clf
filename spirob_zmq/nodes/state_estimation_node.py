"""EKF: MuJoCo prediction, mocap marker update, run once per measurement."""

import copy
import threading

import mujoco
import numpy as np

from spirob_zmq.common import (INITIAL_STATE, MOTOR_STATE, ROBOT_STATE, SITE_MEASUREMENT, VoltageLimit,
                               load_model, rest_state, step_current)
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
        self.motor_ids = list(self.declare_parameter('motor_ids', [10, 1, 2]))

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
        pos_noise = self.declare_parameter('process_pos_noise', 1e-2)
        vel_noise = self.declare_parameter('process_vel_noise', 1e-1)
        act_noise = self.declare_parameter('process_act_noise', 1e-3)
        meas_noise = self.declare_parameter('measurement_noise', 1e-3)
        # Per-marker chi-square gate on the innovation (3 dof, 99.9 %); 0 disables it
        self.innovation_gate = self.declare_parameter('innovation_gate', 16.27)

        # Limit how far the measurement update can move each corrected EKF state block.
        # The limits are L2 norms in each block's units: joint-position units for q,
        # joint-velocity units for dq, and MuJoCo activation units for act. None disables
        # a block's limit. Scaling the matching gain rows keeps the covariance update
        # consistent with the bounded state correction.
        self.max_q_correction = self.declare_parameter('max_q_correction', 0.2)
        self.max_dq_correction = self.declare_parameter('max_dq_correction', 1.0)
        self.max_act_correction = self.declare_parameter('max_act_correction', 0.1)
        # A covariance entry beyond this (or non-finite) means the filter has diverged: reset P
        self.max_covariance = self.declare_parameter('max_covariance', 1e3)

        nx = self.nq + self.nv + self.na

        # Process noise covariance Q and measurement noise covariance R
        self.Q = np.diag([pos_noise ** 2] * self.nq + [vel_noise ** 2] * self.nv
                         + [act_noise ** 2] * self.na)
        self.R = (meas_noise ** 2) * np.eye(self.site_meas)

        # EKF state and covariance, x = [q; dq; act]
        self.x = self._get_state(self.data)
        self.P = np.eye(nx) * self.declare_parameter('initial_covariance', 1e-6)

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

        # Latest applied motor current (from hardware_node), zero until the first MotorState arrives.
        self.app_u = np.zeros(self.model.nu)
        self.vlimit = VoltageLimit(self)

        self.last_stamp = None
        self.have_estimate = False

        # 'measurement': start from the pose fitted to the first still Vicon snapshot (INITIAL_STATE),
        # ignoring measurements until it arrives. 'keyframe': start from the settled init keyframe.
        self.init_from = self.declare_parameter('init_from', 'measurement')
        self.init_q_std = self.declare_parameter('init_q_std', 0.05)
        self.init_dq_std = self.declare_parameter('init_dq_std', 0.05)
        self.initialized = self.init_from == 'keyframe'

        # Subscriptions
        self.create_subscription(MOTOR_STATE, self._on_command)
        self.create_subscription(SITE_MEASUREMENT, self._on_measurement)
        self.create_subscription(INITIAL_STATE, self._on_initial_state)

        # Publisher
        self.state_pub = self.create_publisher(ROBOT_STATE)

    def _filter_valid_measurements(self, z, site_ids):
        """Filter out NaN measurements, returning valid z and site_ids.
        
        Returns:
            (z_valid, valid_site_ids) or (None, None) if all measurements are NaN.
        """
        z_reshaped = z.reshape(len(site_ids), 3)
        valid_mask = ~np.any(np.isnan(z_reshaped), axis=1)
        
        if not np.any(valid_mask):
            return None, None
        
        valid_site_ids = [site_ids[i] for i in np.where(valid_mask)[0]]
        z_valid = z_reshaped[valid_mask].reshape(-1)
        
        return z_valid, valid_site_ids

    def _on_command(self, msg):
        motor_id = msg['motor_id']
        if motor_id not in self.motor_ids:
            return
        i = self.motor_ids.index(motor_id)
        self.app_u[i] = msg['app_ctrl']

    def _on_initial_state(self, msg):
        if self.initialized:
            return
        self.x = np.concatenate([np.asarray(msg['q'], dtype=float), np.zeros(self.nv), np.zeros(self.na)])
        self._reset_covariance()
        self.last_stamp = None
        self.F_step = self.discrete_jacobian(self.x, self.app_u)
        self.initialized = True
        self.get_logger().info(f'initialized from measured pose (fit rms {1e3 * msg.get("rms", float("nan")):.1f} mm)')

    def _on_measurement(self, msg):
        if not self.initialized:
            self.get_logger().info('waiting for the initial state', throttle_duration_sec=2.0)
            return
        z = np.array(msg['data'], dtype=float)
        if z.shape[0] != self.site_meas:
            self.get_logger().warn(
                f'site_measurement has {z.shape[0]} entries, expected {self.site_meas}; ignoring')
            return
        
        # Filter out NaN measurements
        z_valid, valid_site_ids = self._filter_valid_measurements(z, self.site_ids)
        if z_valid is None:
            self.get_logger().warn('All site measurements are NaN; ignoring')
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
        x_prev = self.x.copy()
        self.predict(n)
        self._check_covariance('prediction')
        try:
            self.update(z_valid, valid_site_ids)
        except np.linalg.LinAlgError as exc:
            self.get_logger().error(f'update failed ({exc}); skipping this measurement', throttle_duration_sec=1.0)
        if not np.isfinite(self.x).all():
            self.get_logger().error('state became non-finite; keeping the previous state', throttle_duration_sec=1.0)
            self.x = x_prev
        self._check_covariance('update')
        self.publish(stamp)

    def _reset_covariance(self):
        nq, nv, na = self.nq, self.nv, self.na
        self.P = np.diag([self.init_q_std ** 2] * nq + [self.init_dq_std ** 2] * nv
                         + [self.Q[i, i] for i in range(nq + nv, nq + nv + na)])

    def _check_covariance(self, where):
        """Reset P if it overflowed or blew up, instead of running on with a meaningless gain."""
        finite = np.isfinite(self.P).all()
        biggest = np.abs(self.P).max() if finite else np.inf
        if biggest > self.max_covariance:
            self.get_logger().error(f'covariance diverged after {where} (max |P| {biggest:.3g}); resetting it. '
                                    'Check the model damping and process noise.', throttle_duration_sec=1.0)
            self._reset_covariance()

    def predict(self, n):
        if n == 0:
            return
        if self.n_updates % self.jacobian_every == 0:
            self.request_jacobian()
        F = np.linalg.matrix_power(self.F_step, n)

        self._set_state(self.data, self.x)
        step_current(self.model, self.data, self.app_u, n, self.vlimit)
        self.x = self._get_state(self.data)
        
        # Cap process noise scaling to prevent unbounded covariance growth during measurement gaps
        # Prevents divergence when measurements are delayed or missing for extended periods
        Q_scale = min(n * self.h * self.rate_hz, 10.0)  # Cap at 10x base uncertainty
        self.P = F @ self.P @ F.T + self.Q * Q_scale
        
    def update(self, z, site_ids):
        nq, nv = self.nq, self.nv
        self._set_state(self.data, self.x)
        mujoco.mj_kinematics(self.model, self.data)
        mujoco.mj_comPos(self.model, self.data)
        
        # Compute predicted measurement
        z_pred = np.concatenate([self.data.site_xpos[site_id] for site_id in site_ids])
        # MuJoCo-predicted (prior) positions of all markers, before the Vicon correction
        self.site_pred = self.data.site_xpos[self.site_ids].reshape(-1).copy()
        
        # Build Jacobian H for the sites
        n_sites = len(site_ids)
        meas_size = 3 * n_sites
        H = np.zeros((meas_size, len(self.x)))
        jac = np.zeros((3, nv))
        
        for j, site_id in enumerate(site_ids):
            mujoco.mj_jacSite(self.model, self.data, jac, None, site_id)
            H[3 * j:3 * j + 3, :nq] = jac
        
        # Measurement covariance for the sites
        R = self.R[np.ix_(np.arange(3 * len(site_ids)), np.arange(3 * len(site_ids)))]
        
        y = z - z_pred
        S = H @ self.P @ H.T + R

        # Drop markers whose innovation is implausible for this S (wrong Vicon match, reflection).
        # If most markers disagree, the state is off rather than the markers, so keep them all.
        if self.innovation_gate:
            d2 = []
            for j in range(n_sites):
                marker_y = y[3 * j:3 * j + 3]
                marker_S = S[3 * j:3 * j + 3, 3 * j:3 * j + 3]
                # Degenerate covariance can occur when the propagated P loses rank
                # (for example after very small process noise or long constrained motion).
                # Symmetrize S and add a tiny scale-aware diagonal floor before taking its
                # pseudo-inverse. This keeps residuals in nominally zero-variance directions
                # from being silently ignored, while avoiding a singular 3x3 solve. S is
                # theoretically symmetric, but floating-point operations can introduce a
                # tiny asymmetry.
                marker_S = (marker_S + marker_S.T) / 2.0
                marker_floor = max(1e-12, 1e-9 * np.max(np.abs(np.diag(marker_S))))
                marker_S += marker_floor * np.eye(3)
                marker_S_inv = np.linalg.pinv(marker_S, hermitian=True)
                d2.append(marker_y @ marker_S_inv @ marker_y)
            d2 = np.asarray(d2)
            bad = d2 > self.innovation_gate
            if bad.any() and bad.sum() <= n_sites // 2:
                keep = np.repeat(~bad, 3)
                H, R, y = H[keep], R[np.ix_(keep, keep)], y[keep]
                S = H @ self.P @ H.T + R
                self.get_logger().warn(f'rejected {int(bad.sum())} marker(s) with innovation d2 '
                                       f'{np.round(d2[bad], 1)}', throttle_duration_sec=1.0)
        # Regularization for numerical stability to prevent ill-conditioning
        S_reg = S + 1e-8 * np.eye(S.shape[0])
        K = np.linalg.solve(S_reg, H @ self.P).T
        # Limit each block by its vector norm rather than clipping coordinates
        # independently. Scaling the corresponding rows of K means the covariance update
        # below uses the same effective gain as the bounded state correction.
        # This is a guard against exceptional corrections, not a replacement for sensible
        # noise tuning or the marker outlier gate.
        self._limit_gain(K, y)
        correction = K @ y
        self.x = self.x + correction
        I_KH = np.eye(len(self.x)) - K @ H
        self.P = I_KH @ self.P @ I_KH.T + K @ R @ K.T
        
        # Enforce covariance symmetry: numerical errors can make P slightly asymmetric
        # Symmetric matrices are mathematically required for positive-definiteness
        self.P = (self.P + self.P.T) / 2.0
        
        self.n_updates += 1
        self.have_estimate = True

    def _limit_gain(self, K, innovation):
        """Bound the EKF measurement correction applied to each state block.

        If a block's correction exceeds its configured L2 limit, scale the corresponding
        gain rows so the corrected state moves by at most that amount. The covariance
        update then uses the same bounded gain. None leaves a block unbounded.
        """
        nq, nv = self.nq, self.nv
        blocks = (
            (slice(0, nq), self.max_q_correction),
            (slice(nq, nq + nv), self.max_dq_correction),
            (slice(nq + nv, nq + nv + self.na), self.max_act_correction),
        )
        for block, limit in blocks:
            if limit is None:
                continue
            limit = float(limit)
            if limit <= 0.0:
                raise ValueError('correction limits must be positive, or None to disable')
            step = K[block, :] @ innovation
            norm = np.linalg.norm(step)
            if norm > limit:
                K[block, :] *= limit / norm

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
            # Estimated marker positions (same order as site_names), for plotting vs Vicon
            'site_pos': self.data.site_xpos[self.site_ids].reshape(-1),
            'site_pos_pred': getattr(self, 'site_pred', self.data.site_xpos[self.site_ids].reshape(-1)),
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
