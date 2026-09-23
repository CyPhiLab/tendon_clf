"""Soft ID-CLF-QP controller: robot_state -> motor_command."""

import time

import cvxpy as cp
import mujoco
import numpy as np
from scipy import linalg

from spirob_zmq.common import MOTOR_COMMAND, ROBOT_STATE, load_model
from spirob_zmq.core import Node, run_node


class ControlNode(Node):
    def __init__(self, params=None):
        super().__init__('control_node', params)
        self.model = load_model(self)
        self.data = mujoco.MjData(self.model)

        # CLF invariants
        self.task_dim = 6
        self.F = np.zeros((2*self.task_dim, 2*self.task_dim))
        self.F[:self.task_dim, self.task_dim:] = np.eye(self.task_dim, self.task_dim)
        self.G = np.zeros((2*self.task_dim, self.task_dim))
        self.G[self.task_dim:, :] = np.eye(self.task_dim)
        self.e = 0.05
        self.Pe = linalg.block_diag(np.eye(self.task_dim) / self.e, np.eye(self.task_dim)).T @ linalg.solve_continuous_are(self.F, self.G, np.eye(2*self.task_dim), np.eye(self.task_dim)) @ linalg.block_diag(np.eye(self.task_dim) / self.e, np.eye(self.task_dim))

        # Control gains 
        self.K = self.declare_parameter('K', 500.0)
        self.reg_qdd = self.declare_parameter('reg_qdd', 0.5)
        self.reg_u = self.declare_parameter('reg_u', 0.5)
        self.reg_dl = self.declare_parameter('reg_dl', 1000.0)

        # Control input bounds 
        self.u_min = self.declare_parameter('u_min', -1.0)
        self.u_max = self.declare_parameter('u_max', 0.0)

        # Motor ids, in the same order as the u vector
        self.motor_ids = self.declare_parameter('motor_ids', [0, 1, 2])

        # Control loop rate
        self.rate_hz = self.declare_parameter('rate_hz', 500.0)

        # Fixed task-space target 
        target = self.declare_parameter('target_pos', [0.2, 0.0, 0.2])
        self.target_pos = np.array(target, dtype=float)

        # Name of the end-effector site in the MuJoCo model
        site_name = self.declare_parameter('site_name', 'ee')
        self.site_id = self.model.site(site_name).id

        # Warm-start cache for the QP solver, carried across ticks
        self.previous_solution = None

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

        # Refresh kinematics/dynamics at the current EKF-corrected state
        mujoco.mj_forward(model, data)
        nv, nu = model.nv, model.nu

        # End-effector Jacobian and its time derivative
        jac = np.zeros((6, nv))
        mujoco.mj_jacSite(model, data, jac[:3], jac[3:], self.site_id)
        dJ_dt = compute_jacobian_derivative(model, data, self.site_id)

        # Mass matrix 
        M = np.zeros((nv, nv))
        M_inv = np.zeros((nv, nv))
        mujoco.mj_fullM(model, data, M)
        mujoco.mj_solveM(model, data, M_inv, np.eye(nv))

        # Input matrix B at the current configuration
        Bp = calculate_input_matrix_at_state(model, data)
        pinv_Bp = np.linalg.pinv(Bp)

        # Task-space error (twist)
        dq = data.qvel.reshape(-1, 1)
        ee_pos = data.site(self.site_id).xpos
        twist = np.zeros(6)
        twist[:3] = self.target_pos - ee_pos

        # 6. Decision variables 
        u = cp.Variable(shape=(nu, 1))
        qdd = cp.Variable(shape=(nv, 1))
        dl = cp.Variable(shape=(1, 1))

        ydd = dJ_dt @ dq + jac @ qdd
        tau = M @ qdd + data.qfrc_bias.reshape(-1, 1) - data.qfrc_passive.reshape(-1, 1)

        # Lyapunov function and its derivative
        twist = twist.reshape(-1, 1)
        eta = np.concatenate((-twist, jac @ dq), axis=0)
        V = (eta.T @ self.Pe @ eta).item()
        dV = (eta.T @ (self.F.T @ self.Pe + self.Pe @ self.F) @ eta
              + 2 * eta.T @ self.Pe @ self.G @ (dJ_dt @ dq + jac @ qdd))

        K = self.K
        objective = cp.Minimize(
            0.5 * cp.sum_squares(ydd - (K * twist - 2 * np.sqrt(K) * (jac @ dq)))
            + self.reg_qdd * cp.sum_squares(qdd)
            + self.reg_u * cp.sum_squares(u)
            + self.reg_dl * cp.sum_squares(dl)
        )

        if self.previous_solution is not None:
            u_prev = self.previous_solution['u']
        else:
            u_prev = np.zeros((nu, 1))

        constraints = [
            dV <= -1.0 / self.e * V + dl,
            pinv_Bp @ tau == u,
            u >= self.u_min,
            u <= self.u_max,
            # cp.abs(u - u_prev) <= 0.02,
        ]

        prob = cp.Problem(objective=objective, constraints=constraints)

        if self.previous_solution is not None:
            try:
                u.value = self.previous_solution['u']
                qdd.value = self.previous_solution['qdd']
                dl.value = self.previous_solution['dl']
            except Exception:
                pass

        t_start = time.time()
        try:
            prob.solve(solver=cp.SCS, verbose=False, warm_start=True)
        except Exception as exc:
            self.get_logger().warn(f'QP solve raised an exception: {exc}')
            return
        t_solve = time.time() - t_start
        self.get_logger().info(f"QP = {t_solve*1000:.2f} ms")

        if u.value is None:
            self.get_logger().warn('QP failed to converge -- no solution, skipping this tick')
            return

        self.previous_solution = {
            'u': u.value.copy(),
            'qdd': qdd.value.copy(),
            'dl': dl.value.copy(),
        }

        u_ctrl = np.clip(np.squeeze(u.value), self.u_min, self.u_max)
        # self.get_logger().debug(
        #     f'V={V:.4f} task_error={np.linalg.norm(twist[:3]):.4f} t_solve={t_solve*1000:.2f}ms')

        # Publish the motor command to hardware_node
        self.cmd_pub.publish({
            'stamp': self.now(),
            'motor_ids': list(self.motor_ids),
            'u': np.atleast_1d(u_ctrl),
        })

def calculate_input_matrix_at_state(model, data):
    nv, nu = model.nv, model.nu
    B = np.zeros((nv, nu))

    data_temp = mujoco.MjData(model)
    data_temp.qpos[:] = data.qpos
    data_temp.qvel[:] = data.qvel  # optional; usually 0 is fine too

    mujoco.mj_forward(model, data_temp)

    for i in range(nu):
        data_temp.ctrl[:] = 0.0
        data_temp.ctrl[i] = 1.0
        mujoco.mj_forward(model, data_temp)
        B[:, i] = data_temp.qfrc_actuator.copy()

    return B


def compute_jacobian_derivative(model, data, site_id, h=1e-6):
    """Finite-difference estimate of the time derivative of the site
    Jacobian, ported from spirob_id_clf_qp.py."""
    mujoco.mj_kinematics(model, data)
    mujoco.mj_comPos(model, data)

    J = np.zeros((6, model.nv))
    mujoco.mj_jacSite(model, data, J[:3], J[3:], site_id)

    qpos_backup = np.copy(data.qpos)
    mujoco.mj_integratePos(model, data.qpos, data.qvel, h)

    mujoco.mj_kinematics(model, data)
    mujoco.mj_comPos(model, data)

    Jh = np.zeros((6, model.nv))
    mujoco.mj_jacSite(model, data, Jh[:3], Jh[3:], site_id)

    Jdot = (Jh - J) / h

    data.qpos[:] = qpos_backup
    return Jdot


def main(argv=None):
    run_node(ControlNode, argv)


if __name__ == '__main__':
    main()
