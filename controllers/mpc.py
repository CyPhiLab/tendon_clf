"""MPC controller implementation."""
import time
import numpy as np
import cvxpy as cp
from scipy import linalg
from .base import BaseController, ControllerResult


class MPCController(BaseController):
    """
    Model Predictive Control with receding horizon optimization.
    
    Implements finite-horizon optimal control with CLF constraints for stability
    and actuator limit handling. The controller predicts future system behavior
    over a finite horizon and optimizes control inputs accordingly.
    
    Mathematical Formulation:
    - State Evolution: eta_k+1 = A*eta_k + B*mu_k
    - Optimization: minimize ||mu_k||^2 + lambda*||qdd_k||^2 + lambda*||u_k||^2
    - Subject to: dynamics constraints, CLF constraints, actuator limits
    
    References:

    """
    
    def __call__(self, robot, target_vel, target_acc, twist, previous_solution=None):
        """MPC controller using on-demand robot physics interface"""
        
        # Update input matrix for dynamic robots
        robot.update_input_matrix()
        
        # Get physics data on-demand
        M = robot.get_mass_matrix()
        jac = robot.get_jacobian()
        dJ_dt = robot.get_jacobian_derivative()
        dq = robot.get_joint_velocities()
        null = np.eye(robot.model.nv) - np.linalg.pinv(jac) @ jac

        
        Pe = robot.Pe
        e = robot.e
        F = robot.F
        G = robot.G
        m = robot.task_dim
        Kp = robot.Kp
        Kd = robot.Kd

        # eta_0 (numeric)
        eta_0 = np.concatenate((-twist, jac @ dq - target_vel)).reshape(2*m, 1)

        N = 10  # prediction horizon
        gamma = 0.95
        dt = 0.005

        # Use original constraint formulation

        # Generic MPC formulation (robot-agnostic)
        nu = robot.nu
        nq = robot.model.nq
        qdd_k  = cp.Variable((nq, N))  # joint accelerations
        mu   = cp.Variable((m, N))      # mu[:,k] = mu_k
        u_k  = cp.Variable((nu, N))     # control inputs over horizon
        eta_k = cp.Variable((2*m, N))    # eta_k[:,k] = eta at step k
        qdd_ref = -50 *null @ dq

        # Initial condition constraint
        constraints = []
        constraints += [eta_k[:, 0:1] == eta_0]

        objective = 0
        saved_state = [robot.data.qpos.copy(), robot.data.qvel.copy()]
        # get discrete jacobian
        # Linear discrete dynamics rollout as constraints
        for k in range(N - 1):
            constraints += [eta_k[:, k+1:k+2] == eta_k[:, k:k+1] + dt * (F @ eta_k[:, k:k+1] + G @ mu[:, k:k+1])]
            constraints += [mu[:, k:k+1] == jac @ qdd_k[:, k:k+1] + dJ_dt @ dq.reshape(-1, 1) - target_acc.reshape(-1, 1)] 
            constraints += [robot.pinv_B @ (M @ qdd_k[:, k:k+1] + robot.get_bias_forces().reshape(-1, 1) 
                                       + robot.get_passive_forces().reshape(-1, 1)) == u_k[:, k:k+1]]
            # Add robot-specific control constraints
            for constraint in robot.get_control_constraints(u_k[:, k:k+1]):
                constraints += [constraint]
            # add linearized dynamics constraint
            eta_k1 = eta_k[:, k+1:k+2]
            mu_k1  = mu[:, k:k+1]
            mu_des_k = - Kp * eta_k[0:m, k:k+1] - Kd * eta_k[m:2*m, k:k+1]
            test = eta_k1
            
            objective += (gamma**k) * (robot.mpc_task_weight * cp.sum_squares(mu_k1 - mu_des_k) 
                                       + cp.sum_squares(eta_k1) 
                                    + robot.reg_u * cp.sum_squares(u_k[:, k:k+1]) 
                                    + robot.reg_qdd * cp.sum_squares(qdd_k[:, k:k+1])
                                    )
            # 

        # Terminal penalty (use eta_k at terminal, not eta_next)
        eta_N = eta_k[:, N-1:N]
        objective += robot.mpc_terminal_weight * cp.quad_form(eta_N, Pe)
        objective = cp.Minimize(objective)
        
        robot.data.qpos[:] = saved_state[0]
        robot.data.qvel[:] = saved_state[1]
        
        prob = cp.Problem(objective, constraints)
        # Warm start with previous solution if available
        if previous_solution is not None:
            try:
                u_k.value[:, 0] = previous_solution['u']
            except:
                pass  # If warm start fails, proceed without it    
        try:
            t_ctrl_start = time.time()
            prob.solve(solver=cp.SCS, verbose=False, warm_start=True)
            t_ctrl = time.time() - t_ctrl_start
            if u_k.value is not None:
                robot.apply_control_input(u_k.value[:, 0])
                # Cache solution for next iteration
                current_solution = {
                    'u': u_k.value.copy(),
                }
                return ControllerResult(
                    task_error=np.linalg.norm(twist[:3]),
                    control_input=u_k.value[:, 0].copy(),
                    previous_solution=current_solution,
                    t_ctrl=t_ctrl
                )
            else:
                print(f"failed convergence - no solution\n")
                return ControllerResult(
                    task_error=np.linalg.norm(twist[:3]),
                    control_input=np.zeros((nu,)),
                    previous_solution=previous_solution,
                    t_ctrl=t_ctrl
                )
        except Exception as e:
            print(f"failed convergence - exception: {e}\n")
            return ControllerResult(
                task_error=np.linalg.norm(twist[:3]),
                control_input=np.zeros((nu,)),
                previous_solution=previous_solution,
                t_ctrl=t_ctrl
            )
