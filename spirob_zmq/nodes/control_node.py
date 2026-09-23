"""Soft ID-CLF-QP controller: robot_state -> motor_command.

The QP is the one from controllers/id_clf_qp.py on the claude/port-progress-shc9sv
branch, including its speedups (56d96aa): the cvxpy problem is built once with
Parameters for all per-tick data (DPP, canonicalized once). Solved with
Clarabel by default (see qp_solver below).
Per-robot settings (gains, task_dim, actuator handling) come from robots.py; the
``spirob`` profile reproduces the original ROS control_node's objective.

    min  w_task |J qdd + Jdot dq - mu_des|^2 + reg_qdd |qdd|^2 + reg_u |u|^2
         + reg_dl dl^2 + reg_null |N qdd - qdd_ref|^2
    s.t. CLF:   dV <= -V/e + dl
         ID:    pinv(B) (M qdd + h) = u
         lo(v) <= u <= hi(v)

with B = moment^T k_v and h = bias - passive + moment^T (k_e v_act)
[- qfrc_constraint], so M qdd + h = B u is MuJoCo's own dynamics (the dcmotor's
k_v gain and back-EMF, per SPIROB_HORZ_NOTES.md). lo/hi include the dcmotor's
velocity-dependent force limit.
"""

import time

import cvxpy as cp
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

        # Control input bounds (ctrl units: volts for a dcmotor)
        self.u_min = self.declare_parameter('u_min', -12.0)
        self.u_max = self.declare_parameter('u_max', 0.0)

        # QP solver. Clarabel (interior point) solves every tick of the spirob_horz
        # QP to optimality in ~3 ms; OSQP is similar in median but reports
        # "inaccurate" and occasionally fails on it (ill-conditioned: back-EMF
        # dominates and distal inertias are ~1e-4).
        self.qp_solver = self.declare_parameter('qp_solver', 'CLARABEL')

        # Motor ids, in the same order as the u vector
        self.motor_ids = self.declare_parameter('motor_ids', [0, 1, 2])

        # Control loop rate (a tick takes ~4-5 ms, so 500 Hz as in the ROS node is not reachable)
        self.rate_hz = self.declare_parameter('rate_hz', 200.0)

        # Fixed task-space target
        self.target_pos = np.array(self.declare_parameter('target_pos'), dtype=float)

        # Name of the end-effector site in the MuJoCo model
        site_name = self.declare_parameter('site_name', 'ee')
        self.site_id = self.model.site(site_name).id

        self._build_problem()
        # Warm-start cache for the QP solver, carried across ticks
        self.previous_solution = None
        # Last command sent; used as ctrl when evaluating actuator/constraint forces
        self.last_u = np.zeros(self.model.nu)
        self.solve_times = []

        # Set once a valid RobotState has been received
        self.have_state = False

        # Subscribe to the fused state from state_estimation_node
        self.create_subscription(ROBOT_STATE, self._on_state, latest_only=True)

        # Create publisher for MotorCommand
        self.cmd_pub = self.create_publisher(MOTOR_COMMAND)

        # Create a timer that runs the controller and publishes commands
        self.timer = self.create_timer(1.0 / self.rate_hz, self._tick)

    def _build_problem(self):
        """Build the cvxpy problem once. Every per-tick quantity is a Parameter and
        every Parameter-matrix product is in an affine constraint (via auxiliary
        variables), so the problem is DPP and is canonicalized only once."""
        nu, nv, m = self.model.nu, self.model.nv, self.task_dim

        u = cp.Variable(nu, name='u')
        qdd = cp.Variable(nv, name='qdd')
        dl = cp.Variable(1, name='dl')
        y_task = cp.Variable(m, name='y_task')
        y_null = cp.Variable(nv, name='y_null')
        # N qdd = qdd - J^T w with JJ^T w = J qdd, which avoids an (nv, nv) projector
        w = cp.Variable(m, name='w')

        p = {
            'jac': cp.Parameter((m, nv), name='jac'),
            'task_const': cp.Parameter(m, name='task_const'),   # Jdot dq - mu_des
            'JJT': cp.Parameter((m, m), name='JJT', symmetric=True),
            'qdd_ref': cp.Parameter(nv, name='qdd_ref'),
            'clf_coeff': cp.Parameter(nv, name='clf_coeff'),     # 2 eta^T Pe G J
            'clf_rhs': cp.Parameter(1, name='clf_rhs'),          # -V/e - const
            'pinvBM': cp.Parameter((nu, nv), name='pinvBM'),     # pinv(B) M
            'pinvBh': cp.Parameter(nu, name='pinvBh'),           # -pinv(B) h
            'lb': cp.Parameter(nu, name='lb'),
            'ub': cp.Parameter(nu, name='ub'),
        }
        objective = cp.Minimize(
            self.task_weight * cp.sum_squares(y_task)
            + self.reg_qdd * cp.sum_squares(qdd)
            + self.reg_u * cp.sum_squares(u)
            + self.reg_dl * cp.sum_squares(dl)
            + self.reg_null * cp.sum_squares(y_null)
        )
        constraints = [
            y_task == p['jac'] @ qdd + p['task_const'],
            y_null == qdd - p['jac'].T @ w - p['qdd_ref'],
            p['JJT'] @ w == p['jac'] @ qdd,
            p['clf_coeff'] @ qdd - dl <= p['clf_rhs'],
            p['pinvBM'] @ qdd - u == p['pinvBh'],
            p['lb'] <= u,
            u <= p['ub'],
        ]
        self.prob = cp.Problem(objective, constraints)
        self.u_var, self.qdd_var, self.dl_var = u, qdd, dl
        self.params = p

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
        nu, m = model.nu, self.task_dim

        # Refresh kinematics/dynamics at the current estimated state, with the
        # last command applied (actuator velocity terms and constraint forces)
        data.ctrl[:] = self.last_u
        mujoco.mj_forward(model, data)

        # End-effector Jacobian and its time derivative (task rows only)
        jac6 = np.zeros((6, model.nv))
        jdot6 = np.zeros((6, model.nv))
        point = data.site_xpos[self.site_id]
        body = model.site_bodyid[self.site_id]
        mujoco.mj_jacSite(model, data, jac6[:3], jac6[3:], self.site_id)
        mujoco.mj_jacDot(model, data, jdot6[:3], jdot6[3:], point, body)
        jac, dJ_dt = jac6[:m], jdot6[:m]

        # Mass matrix
        M = np.zeros((model.nv, model.nv))
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

        # Task-space error (twist; orientation rows, if any, are zero)
        dq = data.qvel
        twist = np.zeros(m)
        twist[:3] = self.target_pos - data.site_xpos[self.site_id]
        Jdq = jac @ dq
        mu_des = self.K * twist - self.Kd * Jdq

        # Lyapunov function and the qdd-independent part of its derivative
        eta = np.concatenate((-twist, Jdq))
        V = float(eta @ self.Pe @ eta)
        eta_T_PeG = eta @ self.PeG
        clf_const = float(eta @ self.FTPe_PeF @ eta + 2.0 * eta_T_PeG @ (dJ_dt @ dq))

        # Null-space damping reference: -null_gain * N dq
        JJT = jac @ jac.T
        qdd_ref = -self.null_gain * (dq - jac.T @ np.linalg.solve(JJT, Jdq))

        lb, ub = self.law.ctrl_bounds(self.u_min, self.u_max, act_vel)

        p = self.params
        p['jac'].value = jac
        p['task_const'].value = dJ_dt @ dq - mu_des
        p['JJT'].value = JJT
        p['qdd_ref'].value = qdd_ref
        p['clf_coeff'].value = 2.0 * eta_T_PeG @ jac
        p['clf_rhs'].value = np.array([-V / self.e - clf_const])
        p['pinvBM'].value = pinv_B @ M
        p['pinvBh'].value = -(pinv_B @ h)
        p['lb'].value = lb
        p['ub'].value = ub

        if self.previous_solution is not None:
            self.u_var.value, self.qdd_var.value, self.dl_var.value = self.previous_solution

        t_start = time.perf_counter()
        try:
            self.prob.solve(solver=self.qp_solver, warm_start=True, verbose=False)
        except Exception as exc:
            self.get_logger().warn(f'QP solve raised an exception: {exc}')
            return
        t_solve = time.perf_counter() - t_start
        self.solve_times.append(t_solve)
        self.get_logger().debug(f'QP = {t_solve*1000:.2f} ms  V={V:.4g}  |e|={np.linalg.norm(twist[:3]):.4f}')

        if self.u_var.value is None:
            self.get_logger().warn('QP failed to converge -- no solution, skipping this tick')
            return

        self.previous_solution = (self.u_var.value.copy(), self.qdd_var.value.copy(),
                                  self.dl_var.value.copy())

        u_ctrl = np.clip(self.u_var.value, lb, ub)
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
