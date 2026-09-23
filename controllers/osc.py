"""OSC controller implementations."""
import time
import numpy as np
import cvxpy as cp
from .base import BaseController, ControllerResult


class OSCController(BaseController):
    
    
    def __call__(self, robot, target_vel, target_acc, twist, previous_solution=None):
        """OSC controller using on-demand robot physics interface"""
        
        # Update input matrix for dynamic robots
        robot.update_input_matrix()

        # Get physics data on-demand
        M_inv = robot.get_mass_matrix_inverse()
        jac = robot.get_jacobian()
        dJ_dt = robot.get_jacobian_derivative()
        dq = robot.get_joint_velocities()
        
        # Impedance control using robot attributes
        Kp = robot.Kp
        Kd = robot.Kd

        t_ctrl_start = time.time()
        Mx_inv = jac @ M_inv @ jac.T
        if abs(np.linalg.det(Mx_inv)) >= 1e-2:
            Mx = np.linalg.inv(Mx_inv)
        else:
            Mx = np.linalg.pinv(Mx_inv, rcond=1e-2)
        Jbar = M_inv @ jac.T @ Mx
        C, g = robot.get_coriolis_and_gravity()
        ydd = target_acc + Kp * twist +  Kd * (target_vel - jac @ dq)
        f = Mx @ (ydd - dJ_dt @ dq) + Jbar.T @ (C @ dq + g + robot.get_passive_forces().flatten())
        u = np.linalg.pinv(jac @ M_inv @ robot.B, rcond=1e-8) @ jac @ M_inv @ jac.T @ f
        t_ctrl = time.time() - t_ctrl_start
        try:
            robot.apply_control_input(u)
        except:
            print(f"failed convergence\n")

        return ControllerResult(
            task_error=np.linalg.norm(twist[:3]),
            control_input=u.copy(),
            t_ctrl=t_ctrl
        )