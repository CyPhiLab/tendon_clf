"""ID-CLF-QP controller implementation."""

import numpy as np
import cvxpy as cp
from .base import BaseController, ControllerResult


class IDCLFQPController(BaseController):
    """
    Inverse Dynamics Control Lyapunov Function Quadratic Programming controller.
    
    This controller combines inverse dynamics with Control Lyapunov Functions (CLF)
    to ensure exponential convergence while respecting actuator constraints through
    quadratic programming optimization.
    
    Mathematical Formulation:
    ========================
    
    Task Space Dynamics:
        ẍ = J(q)q̈ + J̇(q)q̇
    
    where x ∈ ℝⁿ is the task space position, q ∈ ℝᵐ is joint space.
    
    Control Lyapunov Function:
        η = [-e; ė - ẋd]  ∈ ℝ²ⁿ
        V(η) = ηᵀPₑη
    
    where e = xd - x is the task error and Pe > 0.
    
    CLF Constraint:
        V̇(η) = ηᵀ(FᵀPₑ + PₑF)η + 2ηᵀPₑG(J(q)q̈ + J̇(q)q̇ - ẍd) ≤ -γV(η) + δ
    
    where F, G are system matrices, γ = 1/e > 0 is the convergence rate,
    and δ ≥ 0 is a relaxation variable.
    
    Optimization Problem:
        minimize: ‖J(q)q̈ + J̇(q)q̇ - μd‖² + λq‖q̈‖² + λu‖u‖² + λδ‖δ‖²
        subject to: V̇(η) ≤ -γV(η) + δ
                   Mq̈ + C(q,q̇) + g(q) + τp = Bu  (inverse dynamics)
                   umin ≤ u ≤ umax                   (actuator limits)
    
    where μd = ẍd + Kp*e + Kd*(ẋd - ẋ) is the desired task acceleration.
    
    Parameters:
    -----------
    robot : Robot
        Robot instance providing physics interface and configuration
    target_vel : np.ndarray
        Desired task space velocity ẋd
    target_acc : np.ndarray  
        Desired task space acceleration ẍd
    twist : np.ndarray
        Current task space error e = xd - x
    previous_solution : dict, optional
        Warm start solution from previous timestep
        
    Returns:
    --------
    ControllerResult
        Contains task error, control input, Lyapunov value, and solution cache
        
    References:
    -----------
    [1] Ames et al. "Control Lyapunov Function Based Quadratic Programs for Safety 
        Critical Systems." IEEE TAC, 2017.
    [2] Nguyen & Sreenath. "Exponential Control Barrier Functions for Enforcing High 
        Relative-Degree Safety-Critical Constraints." ACC, 2016.
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
        qdd = cp.Variable(shape=(nq,))
        dl = cp.Variable(shape=(1,))

        objective = cp.Minimize(cp.square(cp.norm(dJ_dt @ dq + jac @ qdd - mu_des)) + robot.reg_qdd * cp.square(cp.norm(qdd))  
                                + robot.reg_u * cp.square(cp.norm(u,1)) + robot.reg_dl * cp.square(dl)) 

        # Vdot for our main CLF
        dV = eta.T @ (F.T @ Pe + Pe @ F) @ eta + 2 * eta.T @ Pe @ G @ (dJ_dt @ dq + jac @ qdd - target_acc)
        constraints = [dV <= - 1/e * V + dl, 
                       robot.pinv_B @ (M @ qdd + robot.get_bias_forces() + robot.get_passive_forces()) == u]
        constraints += robot.get_control_constraints(u)

        prob = cp.Problem(objective=objective, constraints=constraints)
        
        # Warm start with previous solution if available
        if previous_solution is not None:
            try:
                u.value = previous_solution['u']
                qdd.value = previous_solution['qdd'] 
                dl.value = previous_solution['dl']
            except:
                pass  # If warm start fails, proceed without it
        
        try:
            prob.solve(solver=cp.SCS, verbose=False, warm_start=True)

            if u.value is not None:
                robot.apply_control_input(u.value)

                current_solution = {
                    'u': u.value.copy(),
                    'qdd': qdd.value.copy(),
                    'dl': dl.value.copy()
                }

                return ControllerResult(
                    task_error=np.linalg.norm(twist[:3]),
                    control_input=u.value.copy(),
                    lyapunov_value=float(V),
                    previous_solution=current_solution
                )
            else:
                print(f"failed convergence - no solution\\n")
                return ControllerResult(
                    task_error=np.linalg.norm(twist[:3]),
                    control_input=np.zeros((nu, 1)),
                    lyapunov_value=float(V),
                    previous_solution=previous_solution
                )
        except Exception as e:
            print(f"failed convergence - exception: {e}\\n")
            return ControllerResult(
                task_error=np.linalg.norm(twist[:3]),
                control_input=np.zeros((nu, 1)),
                lyapunov_value=float(V),
                previous_solution=previous_solution
            )

