"""ID-CLF-QP controller."""

import ctypes
import time

import daqp
import mujoco
import numpy as np
from scipy import linalg

from spirob_zmq.common import (MOTOR_COMMAND, MOTOR_STATE, ROBOT_STATE, SITE_MEASUREMENT, TRACKING_STATE, TARGET_STATE,
                               ActuatorLaw, VoltageLimit, actuator_moment,
                               load_model)
from spirob_zmq.core import Node, run_node


class ControlNode(Node):
    def __init__(self, params=None):
        super().__init__('control_node', params)
        self.model = load_model(self, drive_model='current')
        self.data = mujoco.MjData(self.model)
        self.law = ActuatorLaw(self.model)
        self.vlimit = VoltageLimit(self)

        # CLF invariants
        self.task_dim = self.declare_parameter('task_dim', 3)
        m = self.task_dim
        self.F = np.zeros((2 * m, 2 * m))
        self.F[:m, m:] = np.eye(m)
        self.G = np.zeros((2 * m, m))
        self.G[m:, :] = np.eye(m)
        self.e = self.declare_parameter('e', 0.1)
        scale = linalg.block_diag(np.eye(m) / self.e, np.eye(m))
        self.Pe = scale.T @ linalg.solve_continuous_are(self.F, self.G, np.eye(2 * m), np.eye(m)) @ scale
        self.PeG = self.Pe @ self.G
        self.FTPe_PeF = self.F.T @ self.Pe + self.Pe @ self.F

        # Control gains
        self.K = self.declare_parameter('K', 200.0)
        self.Kd = 2.0 * np.sqrt(self.K)
        self.task_weight = self.declare_parameter('task_weight', 1.0)
        self.reg_qdd = self.declare_parameter('reg_qdd', 2.0)
        self.reg_u = self.declare_parameter('reg_u', 24.2)
        self.reg_null = self.declare_parameter('reg_null', 0.1)
        self.reg_dl = self.declare_parameter('reg_dl', 1000.0)
        self.null_gain = self.declare_parameter('null_gain', 50.0)
        # Weight on the planned acceleration the actuators cannot produce (0: off)
        self.reg_dyn = float(self.declare_parameter('reg_dyn', 0.0))

        self.pinv_rcond = self.declare_parameter('pinv_rcond')
        self.include_constraint_forces = self.declare_parameter('include_constraint_forces', False)

        # Motor current bounds [A]; None uses the model's ctrlrange
        self.u_min = self.declare_parameter('u_min')
        self.u_max = self.declare_parameter('u_max')
        self.u_step_max = self.declare_parameter('u_step_max', 0.1)

        # Motor ids, in the same order as the u vector
        self.motor_ids = self.declare_parameter('motor_ids', [10, 1, 2])

        # Control loop rate
        self.rate_hz = self.declare_parameter('rate_hz', 200.0)

        # Fixed task-space target
        self.target_pos = np.array(self.declare_parameter('target_pos'), dtype=float)
        self.tracking = self.declare_parameter('tracking', 'setpoint')
        self.omega = float(self.declare_parameter('omega', 1.0))
        if self.tracking not in ('setpoint', 'trajectory'):
            raise ValueError('tracking must be setpoint or trajectory')
        if not np.isfinite(self.omega):
            raise ValueError('omega must be finite')
        if self.tracking == 'trajectory' and self.task_dim != 3:
            raise ValueError('trajectory tracking requires task_dim=3')
        if self.tracking == 'setpoint' and (
            self.target_pos.shape != (3,) or not np.isfinite(self.target_pos).all()
        ):
            raise ValueError('target_pos must contain three finite coordinates')
        self.trajectory_start_stamp = None
        # Start the trajectory from where the tracked site is: blend from it onto the reference over
        # trajectory_ramp_s seconds (0: jump straight onto the reference)
        self.trajectory_ramp_s = float(self.declare_parameter('trajectory_ramp_s', 0.0))
        self.trajectory_start_point = None
        # Outer-loop integral action on the tracked site's position error [1/s]: shifts the target the
        # CLF-QP tracks by integral_gain * integral(target - position) dt, at most integral_limit [m].
        # Integrates the measured marker when the tracked site is measured, else the estimate.
        self.integral_gain = float(self.declare_parameter('integral_gain', 0.0))
        self.integral_limit = float(self.declare_parameter('integral_limit', 0.05))
        self.integral_offset = np.zeros(3)
        self._integral_stamp = None
        self._integral_measured_time = None
        # Drive friction feedforward [A per motor], added to the QP's current in the direction of the tendon
        # velocity v the QP solution implies, as tanh(v / friction_ff_velocity). While a drive is stuck
        # (measured tendon speed below friction_ff_stick_velocity [m/s]) it gets friction_ff (static friction /
        # gear) to break loose; once moving, friction_ff_coulomb (sliding friction / gear). Pull to wind a
        # tendon in, backdrive current (up to friction_ff_u_max) to let it out. 0 turns it off.
        nu_ = self.model.nu
        vec = lambda key, default: np.broadcast_to(np.asarray(self.declare_parameter(key, default), float), (nu_,)).copy()
        self.friction_ff = vec('friction_ff', 0.0)
        self.friction_ff_coulomb = vec('friction_ff_coulomb', 0.0)
        self.friction_ff_velocity = float(self.declare_parameter('friction_ff_velocity', 1e-3))
        self.friction_ff_stick_velocity = float(self.declare_parameter('friction_ff_stick_velocity', 2e-3))
        self.friction_ff_u_max = float(self.declare_parameter('friction_ff_u_max', 0.4))
        # Direction of the feedforward: 'qp' (the QP solution's next tendon velocity, which carries the
        # estimated velocity's noise) or 'task' (the task-space demand v_ref + friction_ff_gain * error,
        # mapped to tendon velocities through the minimum-norm joint motion)
        self.friction_ff_direction = self.declare_parameter('friction_ff_direction', 'qp')
        self.friction_ff_gain = float(self.declare_parameter('friction_ff_gain', 1.0))
        if self.friction_ff_direction not in ('qp', 'task'):
            raise ValueError("friction_ff_direction must be 'qp' or 'task'")
        self.r_spool = float(self.declare_parameter('r_spool', 0.04))
        self._motor_last = {}                 # motor_id -> (stamp, position)
        self.motor_speed = np.zeros(nu_)      # measured tendon speed [m/s], from motor positions
        if self.trajectory_ramp_s < 0 or self.integral_gain < 0 or self.integral_limit < 0:
            raise ValueError('trajectory_ramp_s, integral_gain and integral_limit must be nonnegative')

        # Name of the end-effector site in the MuJoCo model
        self.site_name = self.declare_parameter('site_name', 'ee')
        self.site_id = self.model.site(self.site_name).id
        self.measurement_site_names = list(self.declare_parameter('site_names', []))
        self.tracking_pub = self.create_publisher(TRACKING_STATE)
        self.target_pub = self.create_publisher(TARGET_STATE)

        nv, nu = self.model.nv, self.model.nu
        self._jac6 = np.zeros((6, nv))
        self._jdot6 = np.zeros((6, nv))
        self._M = np.zeros((nv, nv))
        self._H = np.zeros((nv + 1, nv + 1))
        self._H[nv, nv] = 2.0 * self.reg_dl
        self._g = np.zeros(nv + 1)
        self._A = np.zeros((1 + 4 * nu, nv + 1))
        self._A[0, nv] = -1.0
        self._b = np.zeros(1 + 4 * nu)
        self._blower = np.full(1 + 4 * nu, -1e30)
        self._sense = np.zeros(1 + 4 * nu, dtype=ctypes.c_int)

        self.last_u = np.ones(nu) * (-0.1) 
        self.solve_times = []

        # Set once a valid RobotState has been received
        self.have_state = False
        self.start_delay_s = self.declare_parameter('start_delay_s', 2.0)
        self.start_time = None

        # Subscribe to the fused state from state_estimation_node
        self.create_subscription(ROBOT_STATE, self._on_state, latest_only=True)
        self.create_subscription(SITE_MEASUREMENT, self._on_measurement)
        if np.any(self.friction_ff) or np.any(self.friction_ff_coulomb):
            self.create_subscription(MOTOR_STATE, self._on_motor_state)

        # Create publisher for MotorCommand
        self.cmd_pub = self.create_publisher(MOTOR_COMMAND)

        # Create a timer that runs the controller and publishes commands
        self.timer = self.create_timer(1.0 / self.rate_hz, self._tick)

    def _reference(self, stamp):
        if self.tracking == 'setpoint':
            return self.target_pos.copy(), np.zeros(3), np.zeros(3)
        elapsed = (0.0 if self.trajectory_start_stamp is None
                   else max(0.0, stamp - self.trajectory_start_stamp))
        phase = self.omega * elapsed
        position = np.array([-0.025 * np.cos(phase), 0.25, 0.15 - 0.025 * np.sin(phase)])
        if self.trajectory_start_stamp is None or stamp < self.trajectory_start_stamp:
            return position, np.zeros(3), np.zeros(3)
        velocity = self.omega * np.array([0.025 * np.sin(phase), 0.0, -0.025 * np.cos(phase)])
        acceleration = self.omega ** 2 * np.array([0.025 * np.cos(phase), 0.0, 0.025 * np.sin(phase)])
        if self.trajectory_ramp_s > 0 and self.trajectory_start_point is not None and elapsed < self.trajectory_ramp_s:
            # smoothstep blend s(tau) from the start point onto the circle, with its derivatives
            T = self.trajectory_ramp_s
            tau = elapsed / T
            s, ds, dds = 3 * tau ** 2 - 2 * tau ** 3, (6 * tau - 6 * tau ** 2) / T, (6 - 12 * tau) / T ** 2
            offset = position - self.trajectory_start_point
            acceleration = dds * offset + 2 * ds * velocity + s * acceleration
            velocity = ds * offset + s * velocity
            position = self.trajectory_start_point + s * offset
        return position, velocity, acceleration

    def _integrate(self, stamp, point):
        """Accumulate the integral action from a tracked-site position at time stamp."""
        if self.integral_gain <= 0 or self.start_time is None or self.now() - self.start_time < self.start_delay_s:
            self._integral_stamp = None
            return
        if self._integral_stamp is not None:
            dt = min(max(stamp - self._integral_stamp, 0.0), 0.1)
            target, _, _ = self._reference(stamp)
            self.integral_offset += self.integral_gain * (target - point) * dt
            norm = np.linalg.norm(self.integral_offset)
            if norm > self.integral_limit:
                self.integral_offset *= self.integral_limit / norm
        self._integral_stamp = stamp

    def _on_state(self, msg):
        if not msg['is_valid']:
            return
        self.state_stamp = float(msg['stamp'])
        self.state_received_time = self.now()
        if self.start_time is None:
            self.start_time = self.state_received_time
            self.trajectory_start_stamp = self.state_stamp + self.start_delay_s
        self.data.qpos[:] = msg['q']
        self.data.qvel[:] = msg['dq']
        self.have_state = True
        mujoco.mj_kinematics(self.model, self.data)
        self._publish_tracking(msg['stamp'], self.data.site_xpos[self.site_id], 'estimated')

    def _on_motor_state(self, msg):
        motor_id = msg.get('motor_id')
        if motor_id not in self.motor_ids:
            return
        i = list(self.motor_ids).index(motor_id)
        stamp, position = float(msg['stamp']), float(msg['position'])
        last = self._motor_last.get(motor_id)
        if last is None or stamp - last[0] >= 0.02:   # differentiate over >= 20 ms
            if last is not None:
                v = self.r_spool * (position - last[1]) / (stamp - last[0])
                self.motor_speed[i] = 0.5 * self.motor_speed[i] + 0.5 * v
            self._motor_last[motor_id] = (stamp, position)

    def _on_measurement(self, msg):
        names = msg.get('site_names', self.measurement_site_names)
        if self.site_name not in names:
            self.get_logger().warn(
                f'Tracked site {self.site_name!r} is absent from marker measurements',
                throttle_duration_sec=2.0)
            return
        positions = np.asarray(msg['data'], dtype=float)
        if positions.size != 3 * len(names):
            self.get_logger().warn('Invalid marker measurement size for tracking log')
            return
        point = positions.reshape(-1, 3)[names.index(self.site_name)]
        self._publish_tracking(msg['stamp'], point, msg.get('source', 'measured'))
        if np.isfinite(point).all():
            self._integral_measured_time = self.now()
            self._integrate(float(msg['stamp']), point)

    def _publish_tracking(self, stamp, point, source):
        target, _, _ = self._reference(stamp)
        valid = bool(np.isfinite(point).all())
        position = point.tolist() if valid else [None, None, None]
        error = (point - target).tolist() if valid else [None, None, None]
        if not valid:
            self.get_logger().warn(
                f'Non-finite {source} position for tracked site {self.site_name!r}',
                throttle_duration_sec=2.0)
        self.tracking_pub.publish({
            'stamp': stamp,
            'site_name': self.site_name,
            'source': source,
            'frame': 'mujoco',
            'position_m': position,
            'target_m': target.tolist(),
            'error_m': error,
            'error_norm_m': float(np.linalg.norm(point - target)) if valid else None,
            'is_valid': valid,
        })

    def _tick(self):
        if not self.have_state:
            return
        now = self.now()
        stamp = self.state_stamp + now - self.state_received_time
        target, target_velocity, target_acceleration = self._reference(stamp)
        self.target_pub.publish({
            'stamp': stamp,
            'site_name': self.site_name,
            'frame': 'mujoco',
            'position_m': target.tolist(),
            'velocity_m_s': target_velocity.tolist(),
            'acceleration_m_s2': target_acceleration.tolist(),
        })
        if now - self.start_time < self.start_delay_s:
            return

        model, data = self.model, self.data
        nv, m = model.nv, self.task_dim

        # Refresh kinematics/dynamics at the current estimated state
        data.ctrl[:] = self.last_u
        mujoco.mj_forward(model, data)

        # End-effector Jacobian and its time derivative
        point = data.site_xpos[self.site_id]
        mujoco.mj_jacSite(model, data, self._jac6[:3], self._jac6[3:], self.site_id)
        mujoco.mj_jacDot(model, data, self._jdot6[:3], self._jdot6[3:], point,
                         model.site_bodyid[self.site_id])
        J, dJ_dt = self._jac6[:m], self._jdot6[:m]

        # Mass matrix
        M = self._M
        mujoco.mj_fullM(model, data, M)

        moment = actuator_moment(model, data)
        act_vel = data.actuator_velocity
        B = moment.T * self.law.k_v
        pinv_B = (np.linalg.pinv(B) if self.pinv_rcond is None
                  else np.linalg.pinv(B, rcond=self.pinv_rcond))
        h = data.qfrc_bias - data.qfrc_passive
        if self.include_constraint_forces:
            h = h - data.qfrc_constraint
        P = pinv_B @ M
        p = pinv_B @ h

        if self.tracking == 'trajectory' and self.trajectory_start_point is None:
            # first active tick: the ramp starts from here
            self.trajectory_start_point = point.copy()
            target, target_velocity, target_acceleration = self._reference(stamp)
        # Integral action: from the estimate only while the tracked site is not being measured
        if self._integral_measured_time is None or now - self._integral_measured_time > 0.2:
            self._integrate(stamp, point)

        # Task-space error (twist)
        dq = data.qvel
        twist = np.zeros(m)
        twist[:3] = target + self.integral_offset - point
        Jdq = J @ dq
        reference_velocity = np.zeros(m)
        reference_acceleration = np.zeros(m)
        reference_velocity[:3] = target_velocity
        reference_acceleration[:3] = target_acceleration
        velocity_error = Jdq - reference_velocity
        mu_des = reference_acceleration + self.K * twist - self.Kd * velocity_error
        task_const = dJ_dt @ dq - mu_des

        # Lyapunov function and its derivative
        eta = np.concatenate((-twist, velocity_error))
        V = float(eta @ self.Pe @ eta)
        eta_T_PeG = eta @ self.PeG
        clf_a = 2.0 * eta_T_PeG @ J
        clf_const = float(eta @ self.FTPe_PeF @ eta
                  + 2.0 * eta_T_PeG @ (dJ_dt @ dq - reference_acceleration))

        N = np.eye(nv) - J.T @ np.linalg.solve(J @ J.T, J)
        qdd_ref = -self.null_gain * (N @ dq)

        # Current bounds: ctrlrange / u_min / u_max, intersected with what the 24 V bus can drive
        lb, ub = self.law.ctrl_bounds(self.u_min, self.u_max, act_vel)
        v_lo, v_hi = self.vlimit.bounds(act_vel)
        lb = np.maximum(lb, v_lo)
        ub = np.maximum(np.minimum(ub, v_hi), lb)

        # Per-step control change limit: |u - last_u| <= u_step_max (component-wise).
        du = self.u_step_max
        u_prev = self.last_u

        # QP in z = [qdd; dl]: min 1/2 z'Hz + g'z  s.t.  A z <= b
        H, g, A, b = self._H, self._g, self._A, self._b
        H[:nv, :nv] = 2.0 * (self.task_weight * J.T @ J + self.reg_u * P.T @ P + self.reg_null * N)
        H[np.arange(nv), np.arange(nv)] += 2.0 * self.reg_qdd
        g[:nv] = 2.0 * (self.task_weight * J.T @ task_const
                   + self.reg_u * P.T @ p - self.reg_null * qdd_ref)

        if self.reg_dyn > 0:
            # Penalize the planned acceleration the actuators cannot produce: M^-1 (I - B B^+) (M qdd + h)
            R = np.linalg.solve(M, np.eye(nv) - B @ pinv_B)
            Q, c = R @ M, R @ h
            H[:nv, :nv] += 2.0 * self.reg_dyn * Q.T @ Q
            g[:nv] += 2.0 * self.reg_dyn * Q.T @ c

        nu = model.nu
        A[0, :nv] = clf_a
        A[1:1 + nu, :nv] = P
        A[1 + nu:1 + 2 * nu, :nv] = -P
        b[0] = -V / self.e - clf_const
        b[1:1 + nu] = ub - p
        b[1 + nu:1 + 2 * nu] = p - lb

        # Additional box constraints on the commanded control increment.
        A[1 + 2 * nu:1 + 3 * nu, :nv] = P
        A[1 + 3 * nu:1 + 4 * nu, :nv] = -P
        b[1 + 2 * nu:1 + 3 * nu] = (u_prev + du) - p
        b[1 + 3 * nu:1 + 4 * nu] = -u_prev + du + p

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
        if np.any(self.friction_ff) or np.any(self.friction_ff_coulomb):
            # tendon velocity of the QP's next state, in tendon units (actuator moment = gear * tendon Jacobian)
            if self.friction_ff_direction == 'task':
                v_task = target_velocity + self.friction_ff_gain * twist[:3]
                act_vel_next = moment @ (np.linalg.pinv(J[:3], rcond=1e-3) @ v_task) / model.actuator_gear[:, 0]
            else:
                act_vel_next = moment @ (dq + z[:nv] / self.rate_hz) / model.actuator_gear[:, 0]
            moving = np.abs(self.motor_speed) > self.friction_ff_stick_velocity
            level = np.where(moving, self.friction_ff_coulomb, self.friction_ff)
            u_ff = level * np.tanh(act_vel_next / self.friction_ff_velocity)
            u_ctrl = np.clip(u_ctrl + u_ff, lb, np.maximum(lb, np.minimum(self.friction_ff_u_max, v_hi)))

        # Planned tendon velocities [m/s] for a drive-side inner loop: the QP's next state, and the
        # task-space demand through the minimum-norm joint motion
        gear = model.actuator_gear[:, 0]
        v_task = target_velocity + self.friction_ff_gain * twist[:3]
        # Publish the motor command to hardware_node
        self.cmd_pub.publish({
            'stamp': self.now(),
            'motor_ids': list(self.motor_ids),
            'u': u_ctrl,
            'tendon_velocity_qp': moment @ (dq + z[:nv] / self.rate_hz) / gear,
            'tendon_velocity_task': moment @ (np.linalg.pinv(J[:3], rcond=1e-3) @ v_task) / gear,
            'tendon_velocity_ref': moment @ (np.linalg.pinv(J[:3], rcond=1e-3) @ target_velocity) / gear,
            'tendon_length': data.ten_length[model.actuator_trnid[:, 0]],
        })


def main(argv=None):
    run_node(ControlNode, argv)


if __name__ == '__main__':
    main()
