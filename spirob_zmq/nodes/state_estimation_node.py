
"""EKF: MuJoCo prediction, mocap marker update, once per measurement."""

import copy
import threading

import mujoco
import numpy as np

from spirob_zmq.common import (
    EKF_DIAGNOSTICS,
    INITIAL_STATE,
    MOTOR_STATE,
    ROBOT_STATE,
    SITE_MEASUREMENT,
    VoltageLimit,
    load_model,
    rest_state,
    step_current,
)
from spirob_zmq.core import Node, run_node
from spirob_zmq import drive


class StateEstimationNode(Node):
    def __init__(self, params=None):
        super().__init__('state_estimation_node', params)

        # MuJoCo prediction model
        self.model = load_model(self)
        self.data = mujoco.MjData(self.model)

        self.nq = self.model.nq
        self.nv = self.model.nv
        self.na = self.model.na
        self.nx = self.nq + self.nv + self.na
        # Robot coordinates (drive_model='motor' adds the drives' DOFs after them)
        self.n_robot = drive.robot_nq(self.model)
        # drive_model='motor': the drive states (carriage, motors, LuGre bristles) are predicted open loop
        # and kept out of the covariance. The markers do not see them, and across stick-slip their
        # finite-difference linearization is meaningless (it blows the covariance up within a slip).
        self._robot_mask = np.ones(self.nx)
        if self.n_robot < self.nq:
            self._robot_mask[self.n_robot:self.nq] = 0.0
            self._robot_mask[self.nq + self.n_robot:] = 0.0
        self._robot_mask2 = np.outer(self._robot_mask, self._robot_mask)

        if self.nq != self.nv:
            raise ValueError(
                'This EKF requires nq == nv. '
                'Quaternion coordinates need tangent-space handling.')

        self.motor_ids = list(self.declare_parameter('motor_ids', [10, 1, 2]))

        site_names = self.declare_parameter('site_names')
        self.site_ids = [self.model.site(name).id for name in site_names]
        self.site_index = {site_id: i for i, site_id in enumerate(self.site_ids)}
        self.ee_site_id = self.model.site('ee').id
        self.site_meas = 3 * len(self.site_ids)

        rest_state(self, self.model, self.data)

        h = self.declare_parameter('prediction_timestep')
        if h is not None:
            self.model.opt.timestep = h

        self.h = self.model.opt.timestep
        self.rate_hz = self.declare_parameter('rate_hz', 100.0)

        self.process_current_std = self.declare_parameter('process_current_std', 0.1)
        # Model error not explained by the motor currents (stiffness, friction, geometry): white noise
        # on every joint angle (rad/sqrt(s)) and velocity (rad/s/sqrt(s)). Without it the covariance
        # collapses onto the 3 current directions and the filter stops following the markers.
        self.process_q_std = self.declare_parameter('process_q_std', 0.0)
        self.process_dq_std = self.declare_parameter('process_dq_std', 0.0)
        meas_noise = self.declare_parameter('measurement_noise', 7.5e-3)

        # Diagnostic option
        self.disable_velocity_correction = self.declare_parameter('disable_velocity_correction', False)

        # Optional per-update correction limits (L2 norms).
        # Set to None to disable individual limits.
        self.max_q_correction = self.declare_parameter('max_q_correction', None)
        self.max_joint_q_correction = self.declare_parameter('max_joint_q_correction', 0.2)
        self.max_dq_correction = self.declare_parameter('max_dq_correction', None)
        self.max_act_correction = self.declare_parameter('max_act_correction', None)
        self.max_covariance = self.declare_parameter('max_covariance', 6e1)

        self.R = meas_noise ** 2 * np.eye(self.site_meas)

        self.x = self._get_state(self.data)
        self.init_q_std = self.declare_parameter('init_q_std', 1e-1)
        self.init_dq_std = self.declare_parameter('init_dq_std', 1e-3)
        self.init_act_std = self.declare_parameter('init_act_std', 0.0)

        for name in (
            'process_current_std', 'process_q_std', 'process_dq_std', 'init_q_std', 'init_dq_std',
            'init_act_std', 'max_joint_q_correction',
        ):
            value = getattr(self, name)
            if not np.isfinite(value) or value < 0:
                raise ValueError(f'{name} must be finite and nonnegative')

        self.Q = np.zeros((self.nx, self.nx))
        self._reset_covariance()

        self.jacobian_every = max(1, int(self.declare_parameter('jacobian_every', 10)))
        self.jacobian_thread = self.declare_parameter('jacobian_thread', False)
        self.fd_eps = self.declare_parameter('fd_eps', 1e-6)
        self.fd_centered = self.declare_parameter('fd_centered', False)
        self.lin_model = copy.deepcopy(self.model)
        if self.n_robot < self.nq:
            # drive_model='motor' holds the hinges with a very stiff friction constraint; finite
            # differences across its stick/slip switch give a transition Jacobian that blows the
            # covariance up after every slip. Linearize with MuJoCo's default friction impedance.
            self.lin_model.dof_solref[:self.n_robot] = [0.02, 1.0]
            self.lin_model.dof_solimp[:self.n_robot, :3] = [0.9, 0.95, 0.001]
        self.lin_data = mujoco.MjData(self.lin_model)

        # Keep the linearization timestep consistent with prediction.
        self.lin_model.opt.timestep = self.h
        self._step_jacobians = self.discrete_jacobian(self.x, np.zeros(self.model.nu))
        self.n_updates = 0
        self._jac_request = None
        self._jac_wakeup = threading.Condition()
        self._jac_worker = None

        if self.jacobian_thread:
            self._jac_worker = threading.Thread(target=self._jacobian_loop, daemon=True,)
            self._jac_worker.start()

        self.app_u = np.zeros(self.model.nu)
        self.vlimit = VoltageLimit(self)

        self.last_stamp = None
        self.have_estimate = False
        self.last_diagnostics = {}
        self.prediction_diagnostics = {'n_steps': 0}

        self.init_from = self.declare_parameter('init_from', 'measurement')
        self.initialized = self.init_from == 'keyframe'

        self.create_subscription(MOTOR_STATE, self._on_command)
        # Only the newest measurement: if an update ever takes longer than the measurement interval,
        # skip the stale ones (the prediction covers any gap) instead of queueing them and lagging
        self.create_subscription(SITE_MEASUREMENT, self._on_measurement, latest_only=True)
        self.create_subscription(INITIAL_STATE, self._on_initial_state)

        self.state_pub = self.create_publisher(ROBOT_STATE)
        self.diagnostics_pub = self.create_publisher(EKF_DIAGNOSTICS)

        self.get_logger().info(
            'EKF initialized: '
            f'nx={self.nx}, '
            f'markers={len(self.site_ids)}, '
            f'disable_velocity_correction='
            f'{self.disable_velocity_correction}, '
            'innovation_gating=disabled'
        )

    @staticmethod
    def _finite_or_none(value):
        return float(value) if np.isfinite(value) else None

    def _filter_valid_measurements(self, z, site_ids):
        """Remove markers containing NaN or infinity."""
        points = z.reshape(len(site_ids), 3)
        mask = np.all(np.isfinite(points), axis=1)

        if not np.any(mask):
            return None, None

        valid_ids = [site_ids[i] for i in np.flatnonzero(mask)]
        return points[mask].reshape(-1), valid_ids

    def _on_command(self, msg):
        motor_id = msg['motor_id']

        if motor_id not in self.motor_ids:
            return

        i = self.motor_ids.index(motor_id)
        self.app_u[i] = msg['app_ctrl']

    def _on_initial_state(self, msg):
        if self.initialized:
            return

        q = np.asarray(msg['q'], dtype=float)

        if q.shape not in ((self.nq,), (self.n_robot,)) or not np.isfinite(q).all():
            self.get_logger().error('Invalid initial joint configuration')
            return
        # Robot pose only (the markers' fit): drives at rest tension
        q = drive.full_qpos(self.model, q)

        self.x = np.concatenate([q, np.zeros(self.nv), np.zeros(self.na),])
        self._reset_covariance()
        self.last_stamp = None

        self._step_jacobians = self.discrete_jacobian(self.x, self.app_u)
        self.initialized = True

        rms = msg.get('rms', float('nan'))
        self.get_logger().info(
            f'Initialized from measured pose '
            f'(fit RMS {1e3 * rms:.1f} mm)'
        )

    def _on_measurement(self, msg):
        if not self.initialized:
            self.get_logger().info(
                'Waiting for initial state',
                throttle_duration_sec=2.0,
            )
            return

        z = np.asarray(msg['data'], dtype=float)

        if z.size != self.site_meas:
            self.get_logger().warn(
                f'Measurement has {z.size} entries; '
                f'expected {self.site_meas}'
            )
            return

        stamp = float(msg['stamp'])

        if not np.isfinite(stamp):
            return

        z_valid, valid_ids = self._filter_valid_measurements(z, self.site_ids)

        if z_valid is None:
            self.get_logger().warn('No valid marker measurements')
            return

        if self.last_stamp is None:
            n = 0
        else:
            dt = stamp - self.last_stamp

            if dt <= 0.0:
                self.get_logger().warn('Non-increasing measurement timestamp')
                return

            n = max(1, int(round(dt / self.h)))

        self.last_stamp = stamp

        x_prev = self.x.copy()
        P_prev = self.P.copy()

        self.predict(n)

        if not np.isfinite(self.x).all():
            self.get_logger().error('Non-finite prediction; restoring previous state')
            self.x = x_prev
            self.P = P_prev
            return

        self._check_covariance('prediction')

        try:
            self.update(z_valid, valid_ids)
        except (np.linalg.LinAlgError, ValueError) as exc:
            self.get_logger().error(
                f'EKF update failed: {exc}',
                throttle_duration_sec=1.0,
            )

        if not np.isfinite(self.x).all():
            self.get_logger().error('Non-finite corrected state; restoring previous state')
            self.x = x_prev
            self.P = P_prev

        self._check_covariance('update')
        self.publish(stamp)

    def _reset_covariance(self):
        self.P = np.diag(
            [self.init_q_std ** 2] * self.nq
            + [self.init_dq_std ** 2] * self.nv
            + [self.init_act_std ** 2] * self.na
        ) * self._robot_mask2

    def _process_covariance(self, input_sensitivity):
        """Covariance for independent motor errors held over one measurement interval."""
        return self.process_current_std ** 2 * (input_sensitivity @ input_sensitivity.T)

    def _check_covariance(self, where):
        finite = np.isfinite(self.P).all()

        biggest = (
            np.max(np.abs(self.P))
            if finite else np.inf
        )

        if biggest > self.max_covariance:
            self.get_logger().error(
                f'Covariance diverged after {where}: '
                f'max |P| = {biggest:.3g}. Resetting P.',
                throttle_duration_sec=1.0,
            )
            self._reset_covariance()

    def predict(self, n):
        if n == 0:
            self.prediction_diagnostics = {'n_steps': 0}
            return

        if self.n_updates % self.jacobian_every == 0:
            self.request_jacobian()

        # Replace/read the pair together so asynchronous A and B cannot be mixed.
        F, B = self._step_jacobians
        F_total = np.eye(self.nx)

        B_total = np.zeros_like(B)
        p_before = np.diag(self.P).copy()

        self._set_state(self.data, self.x)

        for _ in range(n):
            step_current(
                self.model,
                self.data,
                self.app_u,
                1,
                self.vlimit,
            )
            # The same delta_u drives every substep: delta_x = B_total @ delta_u.
            B_total = F @ B_total + B
            F_total = F @ F_total

        self.x = self._get_state(self.data)

        self.Q = self._process_covariance(B_total)
        dt = n * self.h
        self.Q[:self.nq, :self.nq] += self.process_q_std ** 2 * dt * np.eye(self.nq)
        self.Q[self.nq:self.nq + self.nv, self.nq:self.nq + self.nv] += self.process_dq_std ** 2 * dt * np.eye(self.nv)
        self.P = (F_total @ self.P @ F_total.T + self.Q) * self._robot_mask2
        self.P = 0.5 * (self.P + self.P.T)

        self.prediction_diagnostics = {
            'n_steps': n,
            'transition_jacobian_fro_norm':
                self._finite_or_none(np.linalg.norm(F)),
            'covariance_diag_max_before_prediction':
                self._finite_or_none(np.max(p_before)),
            'covariance_diag_max_after_prediction':
                self._finite_or_none(np.max(np.diag(self.P))),
            'predicted_q_max_abs':
                self._finite_or_none(
                    np.max(np.abs(self.x[:self.nq]))
                ),
            'predicted_dq_max_abs':
                self._finite_or_none(
                    np.max(np.abs(
                        self.x[self.nq:self.nq + self.nv]
                    ))
                ),
            'predicted_state_finite':
                bool(np.isfinite(self.x).all()),
            'covariance_finite_after_prediction':
                bool(np.isfinite(self.P).all()),
        }

    def _limit_gain_block(self, K, y, sl, limit):
        """Scale gain rows to bound a correction block."""
        if limit is None:
            return

        limit = float(limit)

        if limit < 0:
            raise ValueError('Correction limits must be nonnegative')

        block = K[sl, :] @ y
        norm = np.linalg.norm(block)

        if norm > limit and norm > 0.0:
            K[sl, :] *= limit / norm

    def update(self, z, site_ids):
        nq, nv = self.nq, self.nv

        self._set_state(self.data, self.x)
        mujoco.mj_kinematics(self.model, self.data)
        mujoco.mj_comPos(self.model, self.data)

        z_pred = np.concatenate([self.data.site_xpos[site_id] for site_id in site_ids])

        self.site_pred = (self.data.site_xpos[self.site_ids].reshape(-1).copy())

        n_sites = len(site_ids)
        meas_size = 3 * n_sites

        H = np.zeros((meas_size, self.nx))
        jac = np.zeros((3, nv))

        for j, site_id in enumerate(site_ids):
            mujoco.mj_jacSite(self.model, self.data, jac, None, site_id,)
            H[3*j:3*j+3, :nq] = jac

        # Correct measurement covariance submatrix, preserving original marker ordering.
        indices = []

        for site_id in site_ids:
            i = self.site_index[site_id]
            indices.extend([3*i, 3*i+1, 3*i+2])

        R = self.R[np.ix_(indices, indices)]

        # Innovation: no magnitude gating or rejection.
        y = z - z_pred

        S = H @ self.P @ H.T + R
        S = 0.5 * (S + S.T)

        # Kalman gain
        K = np.linalg.solve(S, H @ self.P).T

        if self.disable_velocity_correction:
            K[nq:nq+nv, :] = 0.0

        # Optional bounded state corrections
        self._limit_gain_block(
            K, y,
            slice(0, nq),
            self.max_q_correction,
        )
        self._limit_gain_block(
            K, y,
            slice(nq, nq+nv),
            self.max_dq_correction,
        )
        self._limit_gain_block(
            K, y,
            slice(nq+nv, self.nx),
            self.max_act_correction,
        )

        correction = K @ y
        capped = np.abs(correction[:nq]) > self.max_joint_q_correction
        capped_indices = np.flatnonzero(capped)
        if capped_indices.size:
            scales = self.max_joint_q_correction / np.abs(correction[capped_indices])
            K[capped_indices, :] *= scales[:, None]
            correction[capped_indices] = (
                np.sign(correction[capped_indices]) * self.max_joint_q_correction
            )
        self.get_logger().info(
            f'EKF update {self.n_updates + 1}: '
            f'joint corrections capped for {capped_indices.size}/{nq} joints '
            f'(limit {self.max_joint_q_correction:g} rad)'
        )

        self.x = self.x + correction

        # Joseph covariance update using the applied gain.
        I_KH = np.eye(self.nx) - K @ H

        self.P = (I_KH @ self.P @ I_KH.T + K @ R @ K.T)
        self.P = 0.5 * (self.P + self.P.T)

        q_slice = slice(0, nq)
        dq_slice = slice(nq, nq+nv)
        act_slice = slice(nq+nv, self.nx)

        diag_p = np.diag(self.P)
        diag_s = np.diag(S)

        self.last_diagnostics = {
            'correction_q_capped_count': int(capped_indices.size),
            'correction_q_capped_indices': capped_indices.tolist(),
            'innovation_rms_m':
                self._finite_or_none(
                    np.sqrt(np.mean(y ** 2))
                ),
            'innovation_max_m':
                self._finite_or_none(np.max(np.abs(y))),
            'innovation_per_site_m': [
                self._finite_or_none(v)
                for v in np.linalg.norm(
                    y.reshape(-1, 3), axis=1
                )
            ],
            'correction_q_norm':
                self._finite_or_none(
                    np.linalg.norm(correction[q_slice])
                ),
            'correction_dq_norm':
                self._finite_or_none(
                    np.linalg.norm(correction[dq_slice])
                ),
            'correction_act_norm':
                self._finite_or_none(
                    np.linalg.norm(correction[act_slice])
                ),
            'correction_q_max_abs':
                self._finite_or_none(
                    np.max(np.abs(correction[q_slice]))
                ),
            'correction_dq_max_abs':
                self._finite_or_none(
                    np.max(np.abs(correction[dq_slice]))
                ),
            'correction_q': correction[q_slice].copy(),
            'correction_dq': correction[dq_slice].copy(),
            'innovation_cov_diag_min':
                self._finite_or_none(np.min(diag_s)),
            'innovation_cov_diag_max':
                self._finite_or_none(np.max(diag_s)),
            'covariance_diag_min':
                self._finite_or_none(np.min(diag_p)),
            'covariance_diag_max':
                self._finite_or_none(np.max(diag_p)),
            'state_finite':
                bool(np.isfinite(self.x).all()),
            'covariance_finite':
                bool(np.isfinite(self.P).all()),
        }

        if (
            not self.last_diagnostics['state_finite']
            or not self.last_diagnostics['covariance_finite']
        ):
            self.get_logger().error(
                f'Non-finite EKF update: '
                f'{self.last_diagnostics}'
            )

        self.n_updates += 1
        self.have_estimate = True

    def publish(self, stamp):
        nq, nv = self.nq, self.nv

        self._set_state(self.data, self.x)
        mujoco.mj_kinematics(self.model, self.data)
        mujoco.mj_comPos(self.model, self.data)

        jac = np.zeros((3, nv))
        mujoco.mj_jacSite(self.model, self.data, jac, None, self.ee_site_id,)

        nr = self.n_robot
        self.state_pub.publish({
            'stamp': stamp,
            'q': self.x[:nr].copy(),
            'dq': self.x[nq:nq+nr].copy(),
            'task_pos':
                self.data.site_xpos[self.ee_site_id].copy(),
            'task_vel':
                jac @ self.x[nq:nq+nv],
            'site_pos':
                self.data.site_xpos[self.site_ids]
                .reshape(-1).copy(),
            'site_pos_pred':
                getattr(
                    self,
                    'site_pred',
                    self.data.site_xpos[self.site_ids]
                    .reshape(-1),
                ).copy(),
            'is_valid': self.have_estimate,
            **({'drive_q': self.x[nr:nq].copy()} if nq > nr else {}),
        })
        self.diagnostics_pub.publish({
            'stamp': stamp,
            **self.prediction_diagnostics,
            **self.last_diagnostics,
        })

    def _get_state(self, d):
        return np.concatenate([
            d.qpos.copy(),
            d.qvel.copy(),
            d.act.copy(),
        ])

    def _set_state(self, d, x):
        d.qpos[:] = x[:self.nq]
        d.qvel[:] = x[self.nq:self.nq+self.nv]
        d.act[:] = x[self.nq+self.nv:]

    def request_jacobian(self):
        snapshot = (
            self.x.copy(),
            self.app_u.copy(),
            self.data.qacc_warmstart.copy(),
        )

        if self._jac_worker is None:
            self._step_jacobians = self.discrete_jacobian(*snapshot)
            return

        with self._jac_wakeup:
            # Replace pending requests with the newest snapshot.
            self._jac_request = snapshot
            self._jac_wakeup.notify()

    def _jacobian_loop(self):
        while True:
            with self._jac_wakeup:
                while (self._jac_request is None and self.ok()):
                    self._jac_wakeup.wait()

                if not self.ok():
                    return

                snapshot = self._jac_request
                self._jac_request = None

            jacobians = self.discrete_jacobian(*snapshot)

            with self._jac_wakeup:
                self._step_jacobians = jacobians

    def discrete_jacobian(self, x, u, warmstart=None):
        m = self.lin_model
        d = self.lin_data

        self._set_state(d, x)
        d.ctrl[:] = u

        if warmstart is not None:
            d.qacc_warmstart[:] = warmstart
        else:
            d.qacc_warmstart[:] = 0.0

        mujoco.mj_forward(m, d)

        A = np.zeros((self.nx, self.nx))
        B = np.zeros((self.nx, m.nu))

        mujoco.mjd_transitionFD(m, d, self.fd_eps, int(self.fd_centered), A, B, None, None,)

        return A * self._robot_mask2, B * self._robot_mask[:, None]

    def destroy_node(self):
        super().destroy_node()

        if self._jac_worker is not None:
            with self._jac_wakeup:
                self._jac_wakeup.notify_all()

            self._jac_worker.join(timeout=1.0)


def main(argv=None):
    run_node(StateEstimationNode, argv)


if __name__ == '__main__':
    main()
