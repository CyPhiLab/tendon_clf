"""Soft ID-CLF-QP controller: robot_state -> motor_command.

The QP is the one from controllers/id_clf_qp.py on the claude/port-progress-shc9sv
branch (with the dcmotor terms from its spirob_horz port). Per-robot settings
come from robots.py; the ``spirob`` profile reproduces the original ROS
control_node's objective.

    min  w_task |J qdd + Jdot dq - mu_des|^2 + reg_qdd |qdd|^2 + reg_u |u|^2
         + reg_dl dl^2 + reg_null |N qdd - qdd_ref|^2
    s.t. CLF:   dV <= -V/e + dl
         ID:    u = pinv(B) (M qdd + h)
         lo(v) <= u <= hi(v)

with B = moment^T k_v and h = bias - passive + moment^T (k_e v_act)
[- qfrc_constraint], so M qdd + h = B u is MuJoCo's own dynamics (dcmotor gain
and back-EMF). lo/hi include the dcmotor's velocity-dependent force limit.

The ID equality defines u, so u is eliminated and the QP is solved densely in
z = [qdd; dl] (nv + 1 variables, 1 + 2 nu inequality rows) with DAQP. That is
the same problem the cvxpy version solved (same u to ~1e-10) at ~130 us instead
of ~4.4 ms, most of which was cvxpy's per-call canonicalization.
"""

import ctypes
import time

import daqp
import mujoco
import numpy as np
from scipy import linalg

from spirob_zmq.common import (MOTOR_COMMAND, ROBOT_STATE, ActuatorLaw, actuator_moment,
                               load_model)
from spirob_zmq.core import Node, run_node


class ControlNode(Node):
    def __init__(self, params=None):
        super().__init__('control_node', params)
        self.model = load_model(self)
        self.data = mujoco.MjData(self.model)
        self.law = ActuatorLaw(self.model)

        # CLF invariants
        self.task_dim = self.declare_parameter('task_dim', 3)
        m = self.task_dim
        self.F = np.zeros((2 * m, 2 * m))
        self.F[:m, m:] = np.eye(m)
        self.G = np.zeros((2 * m, m))
        self.G[m:, :] = np.eye(m)
        self.e = self.declare_parameter('e', 0.01)
        scale = linalg.block_diag(np.eye(m) / self.e, np.eye(m))
        self.Pe = scale.T @ linalg.solve_continuous_are(self.F, self.G, np.eye(2 * m), np.eye(m)) @ scale
        self.PeG = self.Pe @ self.G
        self.FTPe_PeF = self.F.T @ self.Pe + self.Pe @ self.F

        # Control gains
        self.K = self.declare_parameter('K', 200.0)
        self.Kd = 2.0 * np.sqrt(self.K)
        self.task_weight = self.declare_parameter('task_weight', 1.0)
        self.reg_qdd = self.declare_parameter('reg_qdd', 0.2)
        self.reg_u = self.declare_parameter('reg_u', 0.5)
        self.reg_null = self.declare_parameter('reg_null', 0.1)
        self.reg_dl = self.declare_parameter('reg_dl', 1000.0)
        self.null_gain = self.declare_parameter('null_gain', 50.0)

        # Model handling
        self.pinv_rcond = self.declare_parameter('pinv_rcond')
        self.include_constraint_forces = self.declare_parameter('include_constraint_forces', False)

        # Control input bounds (ctrl units: volts for a dcmotor); None uses the
        # model's ctrlrange
        self.u_min = self.declare_parameter('u_min')
        self.u_max = self.declare_parameter('u_max')

        # Motor ids, in the same order as the u vector
        self.motor_ids = self.declare_parameter('motor_ids', [0, 1, 2])

        # Control loop rate
        self.rate_hz = self.declare_parameter('rate_hz', 200.0)

        # Fixed task-space target
        self.target_pos = np.array(self.declare_parameter('target_pos'), dtype=float)

        # Name of the end-effector site in the MuJoCo model
        site_name = self.declare_parameter('site_name', 'ee')
        self.site_id = self.model.site(site_name).id

        # Preallocated work arrays
        nv, nu = self.model.nv, self.model.nu
        self._jac6 = np.zeros((6, nv))
        self._jdot6 = np.zeros((6, nv))
        self._M = np.zeros((nv, nv))
        self._H = np.zeros((nv + 1, nv + 1))
        self._H[nv, nv] = 2.0 * self.reg_dl
        self._g = np.zeros(nv + 1)
        self._A = np.zeros((1 + 2 * nu, nv + 1))
        self._A[0, nv] = -1.0
        self._b = np.zeros(1 + 2 * nu)
        self._blower = np.full(1 + 2 * nu, -1e30)
        self._sense = np.zeros(1 + 2 * nu, dtype=ctypes.c_int)

        # Last command sent; used as ctrl when evaluating actuator/constraint forces
        self.last_u = np.zeros(nu)
        self.solve_times = []

        # Set once a valid RobotState has been received
        self.have_state = False

        # Subscribe to the fused state from state_estimation_node
        self.create_subscription(ROBOT_STATE, self._on_state, latest_only=True)

        # Create publisher for MotorCommand
        self.cmd_pub = self.create_publisher(MOTOR_COMMAND)

        # Create a timer that runs the controller and publishes commands
        self.timer = self.create_timer(1.0 / self.rate_hz, self._tick)

    def _on_state(self, msg):
        if not msg['is_valid']:
            return
        self.data.qpos[:] = msg['q']
        self.data.qvel[:] = msg['dq']
        self.have_state = True

    def _tick(self):
        if not self.have_state:
            return

        model, data = self.model, self.data
        nv, m = model.nv, self.task_dim

        # Refresh kinematics/dynamics at the current estimated state, with the
        # last command applied (actuator velocity terms and constraint forces).
        # A motor-current state is set to its steady state for that command,
        # then only the acceleration stage is recomputed.
        data.ctrl[:] = self.last_u
        mujoco.mj_forward(model, data)
        if self.law.has_act:
            data.act[:] = self.law.steady_act(self.last_u, data.actuator_velocity)
            mujoco.mj_forwardSkip(model, data, mujoco.mjtStage.mjSTAGE_VEL, 0)

        # End-effector Jacobian and its time derivative (task rows only)
        point = data.site_xpos[self.site_id]
        mujoco.mj_jacSite(model, data, self._jac6[:3], self._jac6[3:], self.site_id)
        mujoco.mj_jacDot(model, data, self._jdot6[:3], self._jdot6[3:], point,
                         model.site_bodyid[self.site_id])
        J, dJ_dt = self._jac6[:m], self._jdot6[:m]

        # Mass matrix
        M = self._M
        mujoco.mj_fullM(model, data, M)

        # M qdd + h = B u, exactly as MuJoCo computes it
        moment = actuator_moment(model, data)
        act_vel = data.actuator_velocity
        B = moment.T * self.law.k_v
        pinv_B = (np.linalg.pinv(B) if self.pinv_rcond is None
                  else np.linalg.pinv(B, rcond=self.pinv_rcond))
        h = data.qfrc_bias - data.qfrc_passive + moment.T @ (self.law.k_e * act_vel)
        if self.include_constraint_forces:
            h = h - data.qfrc_constraint
        # u = P qdd + p
        P = pinv_B @ M
        p = pinv_B @ h

        # Task-space error (twist; orientation rows, if any, are zero)
        dq = data.qvel
        twist = np.zeros(m)
        twist[:3] = self.target_pos - point
        Jdq = J @ dq
        mu_des = self.K * twist - self.Kd * Jdq
        task_const = dJ_dt @ dq - mu_des

        # Lyapunov function and its derivative: dV = clf_a qdd + clf_const
        eta = np.concatenate((-twist, Jdq))
        V = float(eta @ self.Pe @ eta)
        eta_T_PeG = eta @ self.PeG
        clf_a = 2.0 * eta_T_PeG @ J
        clf_const = float(eta @ self.FTPe_PeF @ eta + 2.0 * eta_T_PeG @ (dJ_dt @ dq))

        # Null-space projector and damping reference qdd_ref = -null_gain N dq
        N = np.eye(nv) - J.T @ np.linalg.solve(J @ J.T, J)
        qdd_ref = -self.null_gain * (N @ dq)

        lb, ub = self.law.ctrl_bounds(self.u_min, self.u_max, act_vel)

        # Dense QP in z = [qdd; dl]: min 1/2 z'Hz + g'z  s.t.  A z <= b
        # (N is a symmetric projector, so |N qdd - qdd_ref|^2 has Hessian N)
        H, g, A, b = self._H, self._g, self._A, self._b
        H[:nv, :nv] = 2.0 * (self.task_weight * J.T @ J + self.reg_u * P.T @ P + self.reg_null * N)
        H[np.arange(nv), np.arange(nv)] += 2.0 * self.reg_qdd
        g[:nv] = 2.0 * (self.task_weight * J.T @ task_const + self.reg_u * P.T @ p
                        - self.reg_null * qdd_ref)
        nu = model.nu
        A[0, :nv] = clf_a                     # dV + V/e <= dl
        A[1:1 + nu, :nv] = P                  # u <= ub
        A[1 + nu:, :nv] = -P                  # u >= lb
        b[0] = -V / self.e - clf_const
        b[1:1 + nu] = ub - p
        b[1 + nu:] = p - lb

        t_start = time.perf_counter()
        z, _, exitflag, _ = daqp.solve(H, g, A, b, self._blower, self._sense)
        t_solve = time.perf_counter() - t_start
        self.solve_times.append(t_solve)
        self.get_logger().debug(f'QP = {t_solve*1e6:.0f} us  V={V:.4g}  |e|={np.linalg.norm(twist[:3]):.4f}')

        if exitflag < 1:
            self.get_logger().warn(f'QP failed (DAQP exitflag {exitflag}) -- skipping this tick',
                                   throttle_duration_sec=1.0)
            return

        u_ctrl = np.clip(P @ z[:nv] + p, lb, ub)
        self.last_u = u_ctrl

        # Publish the motor command to hardware_node
        self.cmd_pub.publish({
            'stamp': self.now(),
            'motor_ids': list(self.motor_ids),
            'u': u_ctrl,
        })


def main(argv=None):
    run_node(ControlNode, argv)


if __name__ == '__main__':
    main()
