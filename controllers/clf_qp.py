"""CLF-QP controller implementation."""

import time

import numpy as np
import cvxpy as cp
from .base import BaseController, ControllerResult


class CLFQPController(BaseController):
    """
    Control Lyapunov Function Quadratic Programming controller.
    
    This controller uses feedback linearization with Control Lyapunov Functions (CLF)
    to achieve exponential convergence in task space while respecting actuator 
    constraints through quadratic programming optimization.
    
    Mathematical Formulation:
    ========================
    
    Task Space Dynamics via Feedback Linearization:
        ẍ = L₂fy + LgLfy·u
    
    where:
        L₂fy = J̇q̇ - JM⁻¹(C + g + τₚ)  (drift term)
        LgLfy = JM⁻¹B                    (control matrix)
    
    Control Lyapunov Function:
        η = [-e; ė - ẋd]  ∈ ℝ²ⁿ
        V(η) = ηᵀPₑη
    
    where e = xd - x is task error, ė = ẋ - ẋd is velocity error, and Pₑ > 0.
    
    CLF Derivative:
        V̇(η) = ηᵀ(FᵀPₑ + PₑF)η + 2ηᵀPₑGμ
    
    where F, G are CLF system matrices and μ is the achieved task acceleration.
    
    Optimization Problem:
        minimize: ‖V̇‖² + ‖μ - μd‖² + λδ‖δ‖² + λᵤ‖u‖₁
        subject to: V̇ ≤ -γV + δ              (CLF constraint)
                   LgLfy·u = -L₂fy + μ       (feedback linearization)
                   umin ≤ u ≤ umax            (actuator limits)
    
    where:
        μd = ẍd + Kp·e + Kd·ė  (desired task acceleration with PD feedback)
        γ = 1/(e·10)            (convergence rate)
        δ ≥ 0                   (CLF relaxation variable)

    Parameters:
    -----------
    robot : Robot
        Robot instance providing physics interface and configuration
    target_vel : np.ndarray
        Desired task space velocity ẋd ∈ ℝⁿ
    target_acc : np.ndarray  
        Desired task space acceleration ẍd ∈ ℝⁿ
    twist : np.ndarray
        Current task space error e = xd - x ∈ ℝⁿ
    previous_solution : dict, optional
        Warm start solution from previous timestep containing 'u', 'dl'
        
    Returns:
    --------
    ControllerResult
        Contains:
        - task_error: L2 norm of position error ‖e‖
        - control_input: Computed control inputs u ∈ ℝᵐ  
        - lyapunov_value: Current CLF value V(η)
        - previous_solution: Solution cache for warm starting
        - t_ctrl: Control computation time
        
    """
    
    def __call__(self, robot, target_vel, target_acc, twist, previous_solution=None):
        """ID-CLF-QP controller using on-demand robot physics interface"""
        
        # Update input matrix for dynamic robots
        robot.update_input_matrix()
        
        # Get physics data on-demand
        M = robot.get_mass_matrix()
        jac = robot.get_jacobian()
        dJ_dt = robot.get_jacobian_derivative()
        dq = robot.get_joint_velocities()
        
        # Use robot attributes directly for configuration
        F = robot.F
        G = robot.G
        Pe = robot.Pe
        e = robot.e
        Kp = robot.Kp
        Kd = robot.Kd

        # Desired task acceleration 
        mu_des = target_acc + Kp * twist + Kd * (target_vel - jac @ dq)
        eta = np.concatenate((-twist, jac @ dq - target_vel), axis=0)
        # Lyapunov function
        V = eta.T @ Pe @ eta

        # Generic optimization formulation (robot-agnostic)
        nu = robot.nu
        nq = robot.model.nq

        u = cp.Variable(shape=(nu,))
        mu = cp.Variable(shape=(robot.task_dim,))
        dl = cp.Variable(shape=(1,))

        M_inv = robot.get_mass_matrix_inverse()
        L2fy = dJ_dt @ dq - jac @ M_inv @ (robot.get_bias_forces() + robot.get_passive_forces())
        LgLfy = jac @ M_inv @ robot.B
        dV = eta.T @ (F.T @ Pe + Pe @ F) @ eta + 2 * eta.T @ Pe @ G @ mu

        objective = cp.Minimize(cp.square(dV) + cp.square(cp.norm(mu - mu_des)) + robot.reg_dl * cp.square(dl) + robot.reg_u * cp.square(cp.norm(u,1)))
        # Vdot for our main CLF
        constraints = [dV <= - 1/e * V + dl, 
                       LgLfy @ u == -L2fy + mu]
        constraints += robot.get_control_constraints(u)

        prob = cp.Problem(objective=objective, constraints=constraints)
        
        # Warm start with previous solution if available
        if previous_solution is not None:
            try:
                u.value = previous_solution['u']
                # qdd.value = previous_solution['qdd'] 
                dl.value = previous_solution['dl']
            except:
                pass  # If warm start fails, proceed without it
        
        try:
            t_ctrl_start = time.time()
            prob.solve(solver=cp.SCS, verbose=False, warm_start=True)
            t_ctrl = time.time() - t_ctrl_start
            
            if u.value is not None:
                robot.apply_control_input(u.value)
                
                current_solution = {
                    'u': u.value.copy(),
                    # 'qdd': qdd.value.copy(),
                    'dl': dl.value.copy()
                }

                return ControllerResult(
                    task_error=np.linalg.norm(twist[:3]),
                    control_input=u.value.copy(),
                    lyapunov_value=float(V),
                    previous_solution=current_solution,
                    t_ctrl=t_ctrl
                )
            else:
                print(f"failed convergence - no solution\\n")
                return ControllerResult(
                    task_error=np.linalg.norm(twist[:3]),
                    control_input=np.zeros((nu, 1)),
                    lyapunov_value=float(V),
                    previous_solution=previous_solution,
                    t_ctrl=t_ctrl
                )
        except Exception as e:
            print(f"failed convergence - exception: {e}\\n")
            return ControllerResult(
                task_error=np.linalg.norm(twist[:3]),
                control_input=np.zeros((nu, 1)),
                lyapunov_value=float(V),
                previous_solution=previous_solution,
                t_ctrl=t_ctrl
            )

