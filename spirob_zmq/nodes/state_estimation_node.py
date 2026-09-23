"""EKF fusing applied motor control (prediction) with site positions (update)."""

import mujoco
import numpy as np

from spirob_zmq.common import MOTOR_STATE, ROBOT_STATE, SITE_MEASUREMENT, load_model
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

        # Control loop rate
        self.rate_hz = self.declare_parameter('rate_hz', 100.0)
        self.dt = 1.0 / self.rate_hz
        self.model.opt.timestep = self.dt

        # EKF noise parameters (meters and meters/sec, per xyz component, iid Gaussian)
        pos_noise = self.declare_parameter('process_pos_noise', 1e-5)
        vel_noise = self.declare_parameter('process_vel_noise', 1e-5)
        meas_noise = self.declare_parameter('measurement_noise', 1e-5)

        nx = self.nq + self.nv

        # Process noise covariance Q and measurement noise covariance R
        self.Q = np.diag([pos_noise ** 2] * self.nq + [vel_noise ** 2] * self.nv)
        self.R = (meas_noise ** 2) * np.eye(self.site_meas)

        # Discrete-time linearization of the dynamics, x_{k+1} = F_k x_k + B_k u_k
        self.F = np.eye(nx)
        self.F[:self.nq, self.nq:] = self.dt * np.eye(self.nq, self.nv)

        # How F is computed:
        #   'analytic' (default): linearize MuJoCo's Euler step of the smooth
        #       dynamics (~1 ms). Ignores constraints (joint frictionloss, contacts).
        #   'fd': mjd_transitionFD on the full model, constraints included (~20 ms).
        self.jacobian_mode = self.declare_parameter('jacobian', 'analytic')
        # d(gravity + tendon force)/dq is the expensive part of the analytic F and
        # changes slowly with configuration, so it is only refreshed every N ticks.
        self.position_jacobian_every = self.declare_parameter('position_jacobian_every', 10)
        m = self.model
        if self.jacobian_mode == 'analytic' and (
                m.opt.integrator != mujoco.mjtIntegrator.mjINT_EULER
                or m.actuator_gaintype.any() or m.actuator_biastype.any() or m.actuator_dyntype.any()):
            self.get_logger().warn(
                "analytic jacobian assumes the Euler integrator and plain motor actuators; using 'fd'")
            self.jacobian_mode = 'fd'
        self.lin_data = mujoco.MjData(self.model)
        self._dfdq = None
        self._ticks = 0

        # EKF state and covariance, x = [q; dq]
        self.x = np.zeros(nx)
        self.P = np.eye(nx) * 1e-8

        # Latest applied control (from hardware_node), defaults to zero until the first MotorState arrives.
        self.app_u = np.zeros(self.model.nu)

        # Latest site measurement, consumed (and cleared) by the next tick
        self.latest_measurement = None
        self.have_estimate = False

        # Subscriptions
        self.create_subscription(MOTOR_STATE, self._on_command)
        self.create_subscription(SITE_MEASUREMENT, self._on_measurement)

        # Publisher
        self.state_pub = self.create_publisher(ROBOT_STATE)

        # Timer: predict + (optionally) update, then publish
        self.timer = self.create_timer(self.dt, self._tick)

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
        self.latest_measurement = z

    def _tick(self):
        nq, nv = self.nq, self.nv

        # F_k computed at x_k-1, u_k-1
        if self.jacobian_mode == 'fd':
            self.F = self.discrete_jacobian(self.x, self.app_u)[0]
        else:
            self.F = self.analytic_jacobian(self.x, self.app_u)
        self._ticks += 1

        # States at k-1
        self.data.qpos[:] = self.x[:nq]
        self.data.qvel[:] = self.x[nq:]
        self.data.ctrl[:] = self.app_u

        # Predicted states at k
        mujoco.mj_step(self.model, self.data)
        x_pred = np.concatenate([self.data.qpos, self.data.qvel])

        # mj_step leaves xpos/Jacobians at the pre-step configuration; refresh
        # them so z_pred and H are evaluated at x_pred, not at x_{k-1}.
        mujoco.mj_kinematics(self.model, self.data)
        mujoco.mj_comPos(self.model, self.data)

        # Predicted covariance estimate at k
        P_pred = self.F @ self.P @ self.F.T + self.Q

        # Update (only if a new measurement has arrived)
        z = self.latest_measurement
        if z is not None:
            self.latest_measurement = None
            z_pred = np.concatenate([self.data.site(site_id).xpos.copy() for site_id in self.site_ids])

            H = np.zeros((self.site_meas, nq + nv))
            for i, site_id in enumerate(self.site_ids):
                jac = np.zeros((3, nv))
                mujoco.mj_jacSite(self.model, self.data, jac, None, site_id)
                H[3 * i:3 * i + 3, :nq] = jac

            y = z - z_pred
            S = H @ P_pred @ H.T + self.R
            # K = P H^T S^-1, via a solve instead of an explicit inverse (S is symmetric)
            K = np.linalg.solve(S, H @ P_pred).T
            x_upd = x_pred + K @ y
            # Joseph form keeps P symmetric positive semi-definite despite round-off
            I_KH = np.eye(nq + nv) - K @ H
            P_upd = I_KH @ P_pred @ I_KH.T + K @ self.R @ K.T
        else:
            x_upd = x_pred
            P_upd = P_pred

        self.x = x_upd
        self.P = P_upd
        self.have_estimate = True

        # ---- Publish RobotState ----
        self.data.qpos[:] = self.x[:nq]
        self.data.qvel[:] = self.x[nq:]
        mujoco.mj_forward(self.model, self.data)

        jac = np.zeros((3, nv))
        mujoco.mj_jacSite(self.model, self.data, jac, None, self.ee_site_id)

        self.state_pub.publish({
            'stamp': self.now(),
            'q': self.x[:nq],
            'dq': self.x[nq:],
            'task_pos': self.data.site(self.ee_site_id).xpos,
            'task_vel': jac @ self.x[nq:],
            'is_valid': self.have_estimate,
        })

    def analytic_jacobian(self, x, u, eps=1e-6):
        """Linearize MuJoCo's Euler step of the smooth dynamics at (x, u).

        With implicit joint damping D (what MuJoCo's Euler integrator does):
            v' = v + h (M + hD)^-1 (f(q) - D v),   q' = q + h v'
        where f(q) = actuator (tendon) force - gravity - joint stiffness. Hence
            dv'/dq = h (M + hD)^-1 df/dq,  dv'/dv = I - h (M + hD)^-1 D,
            dq'/dq = I + h dv'/dq,         dq'/dv = h dv'/dv.
        M, D and the stiffness are exact; df/dq for gravity and tendon moment
        arms uses cheap position-only passes (no collision or constraint solve).
        Agrees with mjd_transitionFD on the constraint-free model to <1% (the
        rest is Coriolis and dM/dq at nonzero velocity). Constraint forces
        (joint frictionloss, contacts) are not included.
        """
        m, d = self.model, self.lin_data
        nq, nv, h = m.nq, m.nv, m.opt.timestep

        d.qpos[:] = x[:nq]
        d.qvel[:] = 0.0
        if self._dfdq is None or self._ticks % self.position_jacobian_every == 0:
            f0 = self._position_forces(u)
            dfdq = np.zeros((nv, nv))
            for j in range(nv):
                d.qpos[:] = x[:nq]
                d.qpos[j] += eps
                dfdq[:, j] = (self._position_forces(u) - f0) / eps
            self._dfdq = dfdq
            d.qpos[:] = x[:nq]
        mujoco.mj_kinematics(m, d)
        mujoco.mj_comPos(m, d)
        mujoco.mj_crb(m, d)
        M = np.zeros((nv, nv))
        mujoco.mj_fullM(m, d, M)

        dfdq = self._dfdq - np.diag(m.jnt_stiffness[m.dof_jntid])
        damping = m.dof_damping
        MhD = M + h * np.diag(damping)
        dv_dq = h * np.linalg.solve(MhD, dfdq)
        dv_dv = np.eye(nv) - h * np.linalg.solve(MhD, np.diag(damping))

        F = np.empty((2 * nv, 2 * nv))
        F[nv:, :nv] = dv_dq
        F[nv:, nv:] = dv_dv
        F[:nv, :nv] = np.eye(nv) + h * dv_dq
        F[:nv, nv:] = h * dv_dv
        return F

    def _position_forces(self, u):
        """Actuator force minus gravity at lin_data.qpos with zero velocity."""
        m, d = self.model, self.lin_data
        mujoco.mj_kinematics(m, d)
        mujoco.mj_comPos(m, d)
        mujoco.mj_tendon(m, d)
        mujoco.mj_transmission(m, d)
        gravity = np.zeros(m.nv)
        mujoco.mj_rne(m, d, 0, gravity)
        moment = np.zeros((m.nu, m.nv))
        mujoco.mju_sparse2dense(moment, d.actuator_moment, d.moment_rownnz,
                                d.moment_rowadr, d.moment_colind)
        return moment.T @ (m.actuator_gainprm[:, 0] * u) - gravity

    def discrete_jacobian(self, x, u):
        """
        Use MuJoCo's mjd_transitionFD to compute A, B at (x,u).
        This is 1st derivative wrt x, u of the discrete transition function x_{k+1}=f(x_k,u_k).
        By default it uses the dimension 2*nv (position and velocity).
        Adjust if your system dimension is different.
        """
        nq = self.model.nq
        nv = self.model.nv

        Nx = 2 * nv
        Nu = self.model.nu

        # Set the state for this linearization point
        self.data.qpos[:] = x[:nq]
        self.data.qvel[:] = x[nq:nq+nv]
        mujoco.mj_forward(self.model, self.data)
        self.data.ctrl[:] = u

        # We now call mjd_transitionFD
        A = np.zeros((Nx, Nx))
        B = np.zeros((Nx, Nu))
        eps = 1e-5
        flg_centered = 1
        mujoco.mjd_transitionFD(self.model, self.data, eps, flg_centered, A, B, None, None)
        return A, B


def main(argv=None):
    run_node(StateEstimationNode, argv)


if __name__ == '__main__':
    main()
