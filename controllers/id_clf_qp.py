"""ID-CLF-QP controller implementation."""

import time

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
    
    def __init__(self):
        """Initialize controller with lazy problem compilation."""
        self._prob = None
        self._u_var = None
        self._qdd_var = None
        self._dl_var = None
        self._params = {}

    def _build_problem(self, robot, task_dim):
        """Build the CVXPY problem once, using Parameters for all per-step data.

        Uses auxiliary variables (y_task, y_null) so that parameterized matrix
        products appear only in affine equality constraints — making the problem
        DPP-compliant and allowing cvxpy to canonicalize exactly once.
        """
        nu = robot.nu
        nq = robot.model.nq
        nv = robot.model.nv

        # Decision variables
        u   = cp.Variable(shape=(nu,),        name='u')
        qdd = cp.Variable(shape=(nq,),        name='qdd')
        dl  = cp.Variable(shape=(1,),         name='dl')
        # Auxiliary variables for parameterized quadratic terms
        y_task = cp.Variable(shape=(task_dim,), name='y_task')
        y_null = cp.Variable(shape=(nv,),       name='y_null')

        # Parameters — updated each step, never trigger recompilation
        p_jac        = cp.Parameter(shape=(task_dim, nv), name='jac')
        p_task_const = cp.Parameter(shape=(task_dim,),    name='task_const')  # dJ@dq - mu_des
        # Null-space term: instead of a (nv,nv) N parameter, introduce auxiliary variable
        # w ∈ R^task_dim satisfying JJT @ w = J @ qdd.  Then N @ qdd = qdd - J^T @ w,
        # which avoids forming or passing the full (nv,nv) projector each step.
        w            = cp.Variable(shape=(task_dim,),     name='w')
        p_JJT        = cp.Parameter(shape=(task_dim, task_dim), name='JJT', symmetric=True)
        p_qdd_ref    = cp.Parameter(shape=(nv,),          name='qdd_ref')
        p_clf_coeff  = cp.Parameter(shape=(nv,),          name='clf_coeff')   # 2*eta'*Pe*G*jac
        p_clf_rhs    = cp.Parameter(shape=(1,),           name='clf_rhs')     # -1/e*V - const
        # ID constraint via pinv(B): pinv_B @ (M qdd + h) == u
        # Rearranged: p_pinvBM @ qdd - u == p_pinvBh  (p_pinvBh = -pinv_B @ h)
        # Avoids T/Tinv/TinvT in the controller; only pinv_B (nu x nv) is needed.
        p_pinvBM     = cp.Parameter(shape=(nu, nq),       name='pinvBM')      # pinv_B @ M
        p_pinvBh     = cp.Parameter(shape=(nu,),          name='pinvBh')      # -pinv_B @ h
        # Control bounds are parameters, not constants: a dcmotor's force limit
        # makes them depend on tendon velocity, so they change every step.  For
        # robots with static limits these simply take the same value each step.
        p_lb         = cp.Parameter(shape=(nu,),          name='lb')
        p_ub         = cp.Parameter(shape=(nu,),          name='ub')

        objective = cp.Minimize(
            cp.sum_squares(y_task)
            + robot.reg_qdd * cp.sum_squares(qdd)
            + robot.reg_u   * cp.sum_squares(u)
            + robot.reg_dl  * cp.sum_squares(dl)
            + robot.reg_null * cp.sum_squares(y_null)
        )

        constraints = [
            # Auxiliary equalities (make parameterized products DPP-compliant)
            y_task == p_jac @ qdd + p_task_const,
            # Null-space: y_null = N @ qdd - qdd_ref = qdd - J^T w - qdd_ref
            y_null == qdd - p_jac.T @ w - p_qdd_ref,
            p_JJT  @ w == p_jac @ qdd,
            # CLF: clf_coeff @ qdd - dl <= clf_rhs
            p_clf_coeff @ qdd - dl <= p_clf_rhs,
            # Inverse dynamics: pinv_B @ M @ qdd - u == -pinv_B @ h
            p_pinvBM @ qdd - u == p_pinvBh,
            p_lb <= u,
            u <= p_ub,
        ]

        prob = cp.Problem(objective, constraints)

        params = {
            'jac':        p_jac,
            'task_const': p_task_const,
            'JJT':        p_JJT,
            'qdd_ref':    p_qdd_ref,
            'clf_coeff':  p_clf_coeff,
            'clf_rhs':    p_clf_rhs,
            'pinvBM':     p_pinvBM,
            'pinvBh':     p_pinvBh,
            'lb':         p_lb,
            'ub':         p_ub,
        }
        return prob, u, qdd, dl, params

    def __call__(self, robot, target_vel, target_acc, twist, previous_solution=None):
        """ID-CLF-QP controller using on-demand robot physics interface"""

        # Update input matrix for dynamic robots
        t_ctrl_start = time.time()
        robot.update_input_matrix()

        # Get physics data on-demand
        M     = robot.get_mass_matrix()
        jac   = robot.get_jacobian()
        dJ_dt = robot.get_jacobian_derivative(J_precomputed=jac)  # skip redundant kinematics
        dq    = robot.get_joint_velocities()
        h     = robot.get_bias_forces() + robot.get_passive_forces()

        Pe  = robot.Pe
        eps = robot.e   # CLF convergence parameter (renamed to avoid shadowing except clause)
        Kp  = robot.Kp
        Kd  = robot.Kd
        nu  = robot.nu

        mu_des = target_acc + Kp * twist + Kd * (target_vel - jac @ dq)
        eta    = np.concatenate((-twist, jac @ dq - target_vel), axis=0)
        V      = float(eta.T @ Pe @ eta)

        # Build (and compile) the problem once on the first call
        if self._prob is None:
            self._prob, self._u_var, self._qdd_var, self._dl_var, self._params = \
                self._build_problem(robot, jac.shape[0])

        # Pre-compute numpy quantities needed by parameters.
        # robot.PeG and robot.FTPe_PeF are constant matrices cached at robot init.
        eta_T_PeG = eta @ robot.PeG                                             # (task_dim,)
        clf_coeff = 2.0 * eta_T_PeG @ jac                                      # (nv,)
        clf_const = float(eta @ robot.FTPe_PeF @ eta
                          + 2.0 * eta_T_PeG @ (dJ_dt @ dq - target_acc))
        clf_rhs_val = -V / eps - clf_const

        # Null-space qdd_ref = -50 * N @ dq = -50 * (dq - J^T (JJ^T)^{-1} J dq)
        # Avoids forming the full (nv,nv) N matrix; uses O(nv * task_dim) instead of O(nv^2).
        JJT     = jac @ jac.T                                                   # (task_dim, task_dim)
        Jdq     = jac @ dq                                                      # (task_dim,)
        JpinvJdq = jac.T @ np.linalg.solve(JJT, Jdq)                          # (nv,)
        qdd_ref = -50.0 * (dq - JpinvJdq)

        # Update parameters (no recompilation)
        p = self._params
        p['jac'].value        = jac
        p['task_const'].value = dJ_dt @ dq - mu_des          # constant part of task residual
        p['JJT'].value        = JJT
        p['qdd_ref'].value    = qdd_ref
        p['clf_coeff'].value  = clf_coeff
        p['clf_rhs'].value    = np.array([clf_rhs_val])
        pinv_B = robot.pinv_B
        p['pinvBM'].value     = pinv_B @ M
        p['pinvBh'].value     = -(pinv_B @ h)
        lb, ub = robot.get_control_bounds()
        p['lb'].value         = lb
        p['ub'].value         = ub

        # Warm start with previous solution if available
        if previous_solution is not None:
            try:
                self._u_var.value   = previous_solution['u']
                self._qdd_var.value = previous_solution['qdd']
                self._dl_var.value  = previous_solution['dl']
            except Exception:
                pass

        try:
            
            self._prob.solve(solver=cp.OSQP, warm_start=True, verbose=False)
            t_ctrl = time.time() - t_ctrl_start

            u_val = self._u_var.value
            if u_val is not None:
                robot.apply_control_input(u_val)

                current_solution = {
                    'u':   u_val.copy(),
                    'qdd': self._qdd_var.value.copy(),
                    'dl':  self._dl_var.value.copy(),
                }
                return ControllerResult(
                    task_error=np.linalg.norm(twist[:3]),
                    control_input=u_val.copy(),
                    lyapunov_value=V,
                    previous_solution=current_solution,
                    t_ctrl=t_ctrl
                )
            else:
                print("failed convergence - no solution\n")
                return ControllerResult(
                    task_error=np.linalg.norm(twist[:3]),
                    control_input=np.zeros((nu,)),
                    lyapunov_value=V,
                    previous_solution=previous_solution,
                    t_ctrl=t_ctrl
                )
        except Exception as exc:
            t_ctrl = time.time() - t_ctrl_start
            print(f"failed convergence - exception: {exc}\n")
            return ControllerResult(
                task_error=np.linalg.norm(twist[:3]),
                control_input=np.zeros((nu,)),
                lyapunov_value=V,
                previous_solution=previous_solution,
                t_ctrl=t_ctrl
            )

