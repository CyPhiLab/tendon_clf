"""Impedance controller implementations."""
import time
import numpy as np
import cvxpy as cp
from .base import BaseController, ControllerResult


class OSCController(BaseController):
    
    def __call__(self, robot, target_vel, target_acc, twist, previous_solution=None):
        """Impedance controller using on-demand robot physics interface"""
        
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
        Cy = Jbar.T @ C @ dq - Mx @ dJ_dt @ dq
        f = Mx @ ydd + Cy
        sigma = np.linalg.pinv(jac @ M_inv @ jac.T, rcond=1e-8) @ (jac @ M_inv @ jac.T @ f)
        tau = jac.T @ sigma + g + robot.get_passive_forces().flatten()
        u = robot.pinv_B @ tau
        t_ctrl = time.time() - t_ctrl_start
        try:
            robot.apply_control_input(u)
        except:
            print(f"failed convergence\n")
            pass
        
        return ControllerResult(
            task_error=np.linalg.norm(twist[:3]),
            control_input=u.copy(),
            t_ctrl=t_ctrl
        )