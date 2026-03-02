"""Impedance controller implementations."""
import time
import numpy as np
import cvxpy as cp
from .base import BaseController, ControllerResult


class ImpedanceController(BaseController):
    """
    Traditional Cartesian impedance controller.
    
    Implements impedance control in task space to achieve compliant interaction
    behavior. The controller renders desired mass, damping, and stiffness properties
    in Cartesian coordinates.
    
    Mathematical Formulation:
    ========================
    
    Task Space Impedance Model:
        Mx*ẍ + Cx*ẋ + Kx*x = F_ext + F_control
    
    where:
        - Mx: Desired task space mass matrix
        - Cx: Desired task space damping matrix  
        - Kx: Desired task space stiffness matrix
        - F_ext: External forces
        - F_control: Control forces
    
    Task Space Dynamics:
        ẍ = J(q)q̈ + J̇(q)q̇
        
    Task Space Mass Matrix:
        Mx = (J(q)M⁻¹(q)Jᵀ(q))⁻¹
        
    where M(q) is the joint space mass matrix.
    
    Control Law:
        τ = Jᵀ(q)[Mx(ẍd + Kp*e + Kd*ė) + Cx] + C(q,q̇) + g(q) + τp
        
    where:
        - e = xd - x is the task space error
        - Kp, Kd are proportional and derivative gains
        - ẍd is the desired task space acceleration
        - C(q,q̇) are Coriolis/centripetal forces
        - g(q) are gravitational forces
        - τp are passive forces
        - Cx = J̄ᵀC(q,q̇)q̇ - MxJ̇(q)q̇ (task space Coriolis compensation)
        - J̄ = M⁻¹JᵀMx is the dynamically consistent pseudoinverse
    
    The control input is:
        u = B⁺τ
        
    where B⁺ is the Moore-Penrose pseudoinverse of the input matrix B.
    
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
        
    Returns:
    --------
    ControllerResult
        Contains task error and control input
        
    References:
    -----------
    [1] Hogan, N. "Impedance Control: An Approach to Manipulation." 
        Journal of Dynamic Systems, 1985.
    [2] Khatib, O. "A Unified Approach for Motion and Force Control of Robot 
        Manipulators." IEEE Journal of Robotics, 1987.
    """
    
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
        tau = jac.T @ (Mx @ ydd + Cy) + g + robot.get_passive_forces().flatten()
        lower_bounds = robot.lower_bounds
        upper_bounds = robot.upper_bounds
        u = robot.pinv_B @ tau
        t_ctrl = time.time() - t_ctrl_start
        # u = np.clip(u, lower_bounds, upper_bounds)
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


class ImpedanceQPController(BaseController):
    """
    Impedance controller with quadratic programming for actuator limits.
    
    Extends traditional impedance control by formulating it as a quadratic program
    to handle actuator constraints while maintaining impedance behavior.
    
    Mathematical Formulation:
    ========================
    
    Extends ImpedanceController with constrained optimization:
    
    Optimization Problem:
        minimize: ‖J(q)q̈ + J̇(q)q̇ - μd‖² + λq‖q̈‖² + λu‖u‖²
        subject to: Mq̈ + C(q,q̇) + g(q) + τp = Bu  (inverse dynamics)
                   umin ≤ u ≤ umax                   (actuator limits)
    
    where μd = ẍd + Kp*e + Kd*ė is the desired task acceleration from
    impedance control.
    
    This formulation ensures:
    1. Task space impedance behavior when actuator limits are not active
    2. Graceful degradation when approaching actuator limits
    3. Optimal redistribution of control effort across available actuators
    
    The QP framework allows incorporating additional constraints:
    - Joint position/velocity limits
    - Task space constraints
    - Safety constraints
    
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
        Contains task error, control input, and solution cache
        
    References:
    -----------
    [1] Sentis, L. & Khatib, O. "Synthesis of Whole-Body Behaviors through 
        Hierarchical Control of Behavioral Primitives." IJRR, 2005.
    [2] Kanoun, O. et al. "Kinematic Control of Redundant Manipulators: 
        Generalizing the Task-Priority Framework to Inequality Task." 
        IEEE Transactions on Robotics, 2011.
    """
    
    def __call__(self, robot, target_vel, target_acc, twist, previous_solution=None):
        """Impedance QP controller using Robot class"""
        
        # Update input matrix for dynamic robots
        robot.update_input_matrix()
        
        # Get physics data on-demand
        M = robot.get_mass_matrix()
        jac = robot.get_jacobian()
        dJ_dt = robot.get_jacobian_derivative()
        dq = robot.get_joint_velocities()
        
        Kp = robot.Kp
        Kd = robot.Kd

        # Desired task acceleration 
        mu_des = target_acc + Kp * twist + Kd * (target_vel - jac @ dq)

        # Generic optimization formulation (robot-agnostic)
        nu = robot.nu
        nq = robot.model.nq
        u = cp.Variable(shape=(nu, ))
        qdd = cp.Variable(shape=(nq, ))
        N = np.eye(robot.model.nv) - np.linalg.pinv(jac) @ jac
        qdd_null = N @ qdd
        qdd_ref = -50 *N @ dq

        objective = cp.Minimize(cp.square(cp.norm(dJ_dt @ dq + jac @ qdd - mu_des)) 
                                + robot.reg_qdd * cp.square(cp.norm(qdd))  
                                + robot.reg_u * cp.square(cp.norm(u)) 
                                + robot.reg_null * cp.square(cp.norm(qdd_null - qdd_ref)))

        constraints = [robot.pinv_B @ (M @ qdd + robot.get_bias_forces() + robot.get_passive_forces()) == u]
        constraints += robot.get_control_constraints(u)
        
        # Add workspace CBF constraints if enabled
        if getattr(robot, 'enable_cbf', False):
            cbf_constraints = self.generate_workspace_cbf_constraints(robot, qdd)
            constraints.extend(cbf_constraints)

        prob = cp.Problem(objective=objective, constraints=constraints)
        
        # Warm start with previous solution if available
        if previous_solution is not None:
            try:
                u.value = previous_solution['u']
                qdd.value = previous_solution['qdd'] 
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
                    'qdd': qdd.value.copy()
                }

                return ControllerResult(
                    task_error=np.linalg.norm(twist[:3]),
                    control_input=u.value.copy(),
                    previous_solution=current_solution,
                    t_ctrl=t_ctrl
                )
            else:
                print(f"failed convergence - no solution\n")
                return ControllerResult(
                    task_error=np.linalg.norm(twist[:3]),
                    control_input=np.zeros((nu, 1)),
                    previous_solution=previous_solution,
                    t_ctrl=t_ctrl
                )
        except Exception as e:
            print(f"failed convergence - exception: {e}\n")
            return ControllerResult(
                task_error=np.linalg.norm(twist[:3]),
                control_input=np.zeros((nu, 1)),
                previous_solution=previous_solution,
                t_ctrl=t_ctrl
            )
