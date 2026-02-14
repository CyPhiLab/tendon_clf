import argparse
import matplotlib.pyplot as plt
import mujoco
import mujoco.viewer
import numpy as np
import os
from pathlib import Path
import time
from scipy import linalg
import cvxpy as cp
from scipy.linalg import eigh
import gurobipy
import csv
import pandas as pd


# Configure MuJoCo to use the EGL rendering backend (requires GPU)
os.environ["MUJOCO_GL"] = "egl"


def robot(model_name):
    model_path = Path("mujoco_models") / (str(model_name)) / (str(model_name) + str("_control.xml"))
    model = mujoco.MjModel.from_xml_path(str(model_path.absolute()))
    return model

def get_coriolis_and_gravity(model, data):
    """
    Calculate the Coriolis matrix and gravity vector for a MuJoCo model

    Parameters:
        model: MuJoCo model object
        data: MuJoCo data object

    Returns:
        C: Coriolis matrix (nv x nv)
        g: Gravity vector (nv,)
    """
    nv = model.nv  # number of degrees of freedom

    # Calculate gravity vector
    g = np.zeros(nv)
    dummy = np.zeros(nv,)
    mujoco.mj_factorM(model, data)  # Compute sparse M factorization
    mujoco.mj_rne(model, data, 0, dummy)  # Run RNE with zero acceleration and velocity
    g = data.qfrc_bias.copy()

    # Calculate Coriolis matrix
    C = np.zeros((nv, nv))
    q_vel = data.qvel.copy()

    # Compute each column of C using finite differences
    eps = 1e-6
    for i in range(nv):
        # Save current state
        vel_orig = q_vel.copy()

        # Perturb velocity
        q_vel[i] += eps
        data.qvel = q_vel

        # Calculate forces with perturbed velocity
        mujoco.mj_rne(model, data, 0, dummy)
        tau_plus = data.qfrc_bias.copy()

        # Restore original velocity
        q_vel = vel_orig
        data.qvel = q_vel

        # Compute column of C using finite difference
        C[:, i] = (tau_plus - data.qfrc_bias) / eps

    return C, g

def compute_jacobian_derivative(model, data, site_id, h=1e-6):
    """
    Compute the time derivative of the Jacobian in MuJoCo.
    
    Parameters:
    - model: The MuJoCo model (mjModel).
    - data: The MuJoCo data structure (mjData).
    - jac_func: Function to compute the Jacobian (e.g., mj_jacBody or mj_jacSite).
    - h: Small positive step for numerical differentiation.
    
    Returns:
    - Jdot: The time derivative of the Jacobian.
    """
    # Step 1: Update kinematics
    mujoco.mj_kinematics(model, data)
    mujoco.mj_comPos(model, data)
    
    # Step 2: Compute the initial Jacobian
    J = np.zeros((6, model.nv))  # Assuming a 6xnv Jacobian for full spatial representation
    mujoco.mj_jacSite(model, data, J[:3], J[3:], site_id)
    
    # Step 3: Integrate position using velocity
    qpos_backup = np.copy(data.qpos)  # Backup original qpos
    mujoco.mj_integratePos(model, data.qpos, data.qvel, h)
    
    # Step 4: Update kinematics again
    mujoco.mj_kinematics(model, data)
    mujoco.mj_comPos(model, data)
    
    # Step 5: Compute the new Jacobian
    Jh = np.zeros((6, model.nv))
    mujoco.mj_jacSite(model, data, Jh[:3], Jh[3:], site_id)
    
    # Step 6: Compute Jdot
    Jdot = (Jh - J) / h
    
    # Step 7: Restore qpos
    data.qpos[:] = qpos_backup
    
    return Jdot

def circular_trajectory(t, model_name):
    """
    Circular trajectory through the 4 given points.
    One full revolution in time T.
    """
    if model_name == 'helix':
        L = 0.435/2
        R = L/2
        h = 0.7 - 2*L
    elif model_name == 'tendon':
        L = -0.24/2
        R = L/2
        h = 0
    elif model_name == 'spirob':
        L = 0.5/2
        R = L/2
        h = 0

    # Circle parameters
    cx, cy, cz = L, 0, L+h
    r = R

    # Angle
    omega = 0.25 * np.pi 
    theta = omega * t

    # Position
    x = cx + r * np.cos(theta)
    y = cy
    z = cz + r * np.sin(theta)

    # Velocity
    xd = -r * omega * np.sin(theta)
    yd = 0.0
    zd =  r * omega * np.cos(theta)

    # Acceleration
    xdd = -r * omega**2 * np.cos(theta)
    ydd = 0.0
    zdd = -r * omega**2 * np.sin(theta)

    pos = np.array([x, y, z])
    vel = np.array([xd, yd, zd])
    acc = np.array([xdd, ydd, zdd])

    return omega,{"pos": pos, "vel": vel, "acc": acc}

def set_target(target_pos, model_name):
    
    if model_name == 'helix':
        L = 0.435/2
        R = L/2
        h = 0.7 - 2*L
    elif model_name == 'tendon':
        L = -0.24/2
        R = L/2
        h = 0
    elif model_name == 'spirob':
        L = 0.5/2
        R = L/2
        h = 0

    pos1 = np.array([L-R, 0.0, L+h])
    pos2 = np.array([L+R, 0.0, L+h])
    pos3 = np.array([L, 0.0, L+R+h])
    pos4 = np.array([L, 0.0, L-R+h])

    targets = {
        'pos1': pos1,
        'pos2': pos2,
        'pos3': pos3,
        'pos4': pos4
    }

    return targets[target_pos]

# Pre-compute invariant matrices (computed once, used throughout simulation)
def precompute_invariants(model, model_name=None):
    """Pre-compute matrices that don't change during simulation"""

    # Control input matrix (mapping from tendon forces to joint torques)
    if model_name == 'tendon':
        B = np.array([[0.1, 0.0], [0.1, 0.0], [0.0, 0.1], [0.0, 0.1]])
        Bp = np.array([[1, 0.0], [-1, 0.0], [0.0, 1], [0.0, -1]])
        pinv_B = np.linalg.pinv(B)
        # CLF matrices
        m = 3
        F = np.zeros((2*m, 2*m))
        F[:m, m:] = np.eye(m, m)
        G = np.zeros((2*m, m))
        G[m:, :] = np.eye(m)
        e = 0.05
        Pe = linalg.block_diag(np.eye(m) / e, np.eye(m)).T @ linalg.solve_continuous_are(F, G, np.eye(2*m), np.eye(m)) @ linalg.block_diag(np.eye(m) / e, np.eye(m))
        Kp = 500
        Kd = 2 * np.sqrt(Kp)
        damping = 0.01
        stiffness = 0.01

        return {
            'm': m,
            'F': F,
            'G': G,
            'Pe': Pe,
            'pinv_B': pinv_B,
            'Bp': Bp,
            'e': e,
            'Kp': Kp,
            'Kd': Kd,
            'damping': damping,
            'stiffness': stiffness
        }
    
    elif model_name == 'helix':
        B = np.zeros((36, 9))
        for i in range(3):  # Iterate over u1, u2, u3 blocks
            for j in range(4):  # Repeat each block 4 times
                row_start = i * 12 + j * 3  # Compute row index
                col_start = i * 3  # Compute column index
                B[row_start:row_start+3, col_start:col_start+3] = np.eye(3)  # Assign identity
        pinv_B = np.linalg.pinv(B)

        # CLF matrices
        m = 6
        F = np.zeros((2*m, 2*m))
        F[:m, m:] = np.eye(m, m)
        G = np.zeros((2*m, m))
        G[m:, :] = np.eye(m)
        e = 0.1
        Pe = linalg.block_diag(np.eye(m) / e, np.eye(m)).T @ linalg.solve_continuous_are(F, G, np.eye(2*m), np.eye(m)) @ linalg.block_diag(np.eye(m) / e, np.eye(m))
        Kp = 200
        Kd = 2 * np.sqrt(Kp)
        damping = 0.2
        stiffness = 0.1
        
        # Input matrix pseudoinverse
        pinv_B = np.linalg.pinv(B)
        
        # Selection matrix for compression/extension actuators
        nu = 9
        sel = np.ones((nu, 1))
        sel[[2, 5, 8]] = 0.0

        return {
            'm': m,
            'F': F,
            'G': G,
            'Pe': Pe,
            'B': B,
            'pinv_B': pinv_B,
            'sel': sel,
            'e': e,
            'Kp': Kp,
            'Kd': Kd,
            'damping': damping,
            'stiffness': stiffness
        }
    
    elif model_name == 'spirob':
        # CLF matrices
        m = 6
        F = np.zeros((2*m, 2*m))
        F[:m, m:] = np.eye(m, m)
        G = np.zeros((2*m, m))
        G[m:, :] = np.eye(m)
        e = 0.01
        Pe = linalg.block_diag(np.eye(m) / e, np.eye(m)).T @ linalg.solve_continuous_are(F, G, np.eye(2*m), np.eye(m)) @ linalg.block_diag(np.eye(m) / e, np.eye(m))

        # Selection matrix for control inputs
        nu = model.nu  # Use actual number of actuators from model
        sel = np.ones((nu, 1))  # All control inputs are active

        # PD gains for impedance control (can be tuned)
        Kp = 500.0
        Kd = 2 * np.sqrt(Kp)
        damping = 0.01
        stiffness = 0.01
                
        return {
            'm': m,
            'F': F,
            'G': G,
            'Pe': Pe,
            'sel': sel,
            'e': e,
            'Kp': Kp,
            'Kd': Kd,
            'damping': damping,
            'stiffness': stiffness
        }

def id_clf_qp_control(model_name, model, data, invariants, eta, target_vel, target_acc, twist, jac, M, dJ_dt, previous_solution=None):

    F = invariants['F']
    G = invariants['G']

    if model_name == 'tendon':
        Pe = invariants['Pe']
        pinv_B = invariants['pinv_B']
        Bp = invariants['Bp']
        e = invariants['e']
        Kp = invariants['Kp']
        Kd = invariants['Kd']

        V = eta.T @ Pe @ eta

        # Use original constraint formulation
        dq = data.qvel.reshape(-1,1)
        mu_des = target_acc.reshape(-1,1) + Kp * twist.reshape(-1,1) + Kd * (target_vel.reshape(-1,1) - jac @ dq)

        # define decision variables (create fresh variables each time)
        nu = 2
        nq = model.nq
        u = cp.Variable(shape=(nu, 1))
        qdd = cp.Variable(shape=(nq, 1))
        dl = cp.Variable(shape=(1, 1))

        objective = cp.Minimize(cp.square(cp.norm(dJ_dt @ dq + jac @ qdd - mu_des)) 
                                + 0.02 * cp.square(cp.norm(qdd))  + 0.02 * cp.square(cp.norm(u)) + 1000 * cp.square(dl)) 

        # Vdot for our main CLF
        dV = eta.T @ (F.T @ Pe + Pe @ F) @ eta + 2 * eta.T @ Pe @ G @ (dJ_dt @ dq + jac @ qdd - target_acc.reshape(-1,1))
        constraints = [ dV <= - 1/e * V + dl, 
                        pinv_B @ (M @ qdd + data.qfrc_bias.reshape(-1,1) - data.qfrc_passive.reshape(-1,1)) == u,
                        -1.0 <= u,
                        1.0 >= u]

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
                u = u.value.copy()
                data.ctrl[:] = np.squeeze(Bp @ u)

                current_solution = {
                    'u': u.copy(),
                    'dl': dl.value.copy()
                }

                return V, current_solution, u.copy()
            else:
                print(f"failed convergence - no solution\n")
                return V, previous_solution, None
        except Exception as e:
            print(f"failed convergence - exception: {e}\n")
            return V, previous_solution, None
        
    elif model_name == 'helix':
        Pe = invariants['Pe']
        B = invariants['B']
        pinv_B = invariants['pinv_B']
        sel = invariants['sel']
        e = invariants['e']
        Kp = invariants['Kp']
        Kd = invariants['Kd']

        V = eta.T @ Pe @ eta

        # Use original constraint formulation
        dq = data.qvel.reshape(-1,1)
        mu_des = target_acc.reshape(-1,1) + Kp * twist.reshape(-1,1) + Kd * (target_vel.reshape(-1,1) - jac @ dq)

        # define decision variables (create fresh variables each time)
        nu = 9
        nq = model.nq
        u = cp.Variable(shape=(nu, 1))
        qdd = cp.Variable(shape=(nq, 1))
        dl = cp.Variable(shape=(1, 1))

        # Vdot for our main CLF
        dV = eta.T @ (F.T @ Pe + Pe @ F) @ eta + 2 * eta.T @ Pe @ G @ (dJ_dt @ dq + jac @ qdd - target_acc.reshape(-1,1))

        objective = cp.Minimize(cp.square(cp.norm(dJ_dt @ dq + jac @ qdd - mu_des)) 
                                + 0.5 * cp.square(cp.norm(qdd))  + 0.5 * cp.square(cp.norm(u)) + 1000 * cp.square(dl)) 

        constraints = [ dV <= - 1/e * V + dl, 
                        pinv_B @ (M @ qdd + data.qfrc_bias.reshape(-1,1) + data.qfrc_passive.reshape(-1,1)) == u,
                        -25*sel <= u,
                        25*np.ones((nu,1)) >= u]

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
                u = u.value.copy()
                data.ctrl[:] = np.squeeze(B @ u)

                current_solution = {
                    'u': u.copy(),
                    'qdd': qdd.value.copy(),
                    'dl': dl.value.copy()
                }

                return V, current_solution, u.copy()
            else:
                print(f"failed convergence - no solution\n")
                return V, previous_solution, None
        except Exception as e:
            print(f"failed convergence - exception: {e}\n")
            return V, previous_solution, None
        
    elif model_name == 'spirob':
        Pe = invariants['Pe']
        sel = invariants['sel']
        e = invariants['e']
        Kp = invariants['Kp']
        Kd = invariants['Kd']

        # Sprirob input matrix depends on current configuration, so we compute it at the current state
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

        # Compute the pseudoinverse of B
        pinv_B = np.linalg.pinv(B)

        V = eta.T @ Pe @ eta

        # Use original constraint formulation
        dq = data.qvel.reshape(-1,1)
        mu_des = target_acc.reshape(-1,1) + Kp * twist.reshape(-1,1) + Kd * (target_vel.reshape(-1,1) - jac @ dq)

        # define decision variables (create fresh variables each time)
        nu = model.nu
        nq = model.nq
        u = cp.Variable(shape=(nu, 1))
        qdd = cp.Variable(shape=(nq, 1))
        dl = cp.Variable(shape=(1, 1))
        
        # Null-space projection to handle redundancy
        N = np.eye(model.nv) - np.linalg.pinv(jac) @ jac
        qdd_null = N @ qdd

        # Vdot for our main CLF
        dV = eta.T @ (F.T @ Pe + Pe @ F) @ eta + 2 * eta.T @ Pe @ G @ (dJ_dt @ dq + jac @ qdd - target_acc.reshape(-1,1))

        objective = cp.Minimize(cp.square(cp.norm(dJ_dt @ dq + jac @ qdd - mu_des)) 
                                + 0.2 * cp.square(cp.norm(qdd))  + 0.5 * cp.square(cp.norm(u)) 
                                + 0.5 * cp.square(cp.norm(qdd_null)) + 1000 * cp.square(dl)) 

        constraints = [ dV <= - 1/e * V + dl, 
                        pinv_B @ (M @ qdd + data.qfrc_bias.reshape(-1,1) - data.qfrc_passive.reshape(-1,1)) == u,
                        np.zeros((nu,1)) >= u]

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
                u = u.value.copy()
                data.ctrl[:] = np.squeeze(u)

                current_solution = {
                    'u': u.copy(),
                    'dl': dl.value.copy()
                }

                return V, current_solution, u.copy()
            else:
                print(f"failed convergence - no solution\n")
                return V, previous_solution, None
        except Exception as e:
            print(f"failed convergence - exception: {e}\n")
            return V, previous_solution, None

def impedance_control(model_name, model, data, invariants, target_vel, target_acc, twist, jac, M_inv, dJ_dt):

    # Impedance control
    if model_name == 'tendon':
        pinv_B = invariants['pinv_B']
        Bp = invariants['Bp']
        Kp = invariants['Kp']
        Kd = invariants['Kd']
    elif model_name == 'helix':
        B = invariants['B']
        pinv_B = invariants['pinv_B']
        Kp = invariants['Kp']
        Kd = invariants['Kd']
    elif model_name == 'spirob':
        pinv_B = invariants['pinv_B']
        Kp = invariants['Kp']
        Kd = invariants['Kd']

    Mx_inv = jac @ M_inv @ jac.T
    if abs(np.linalg.det(Mx_inv)) >= 1e-2:
        Mx = np.linalg.inv(Mx_inv)
    else:
        Mx = np.linalg.pinv(Mx_inv, rcond=1e-2)
    Jbar = M_inv @ jac.T @ Mx
    C, g = get_coriolis_and_gravity(model, data)
    ydd = target_acc + Kp * twist +  Kd * (target_vel - jac @ data.qvel)
    Cy = Jbar.T @ C @ data.qvel - Mx @ dJ_dt @ data.qvel
    if model_name == 'helix':
        tau = jac.T @ (Mx @ ydd + Cy) + g + data.qfrc_passive
    else:        
        tau = jac.T @ (Mx @ ydd + Cy) + g - data.qfrc_passive
    u = pinv_B @ tau
    
    try:
        if model_name == 'tendon':
            data.ctrl = Bp @ u
        elif model_name == 'helix':
            data.ctrl = B @ u
        elif model_name == 'spirob':
            data.ctrl = u

    except:
        print(f"failed convergence\n")
        pass
    return u.copy()

def mpc_control(model_name, model, data, invariants, target_vel, target_acc, twist, jac, M, dJ_dt, previous_solution=None):
    F = invariants['F']
    G = invariants['G']
    m = invariants['m']
    # Use original constraint formulation
    dq = data.qvel.reshape(-1,1)

    if model_name == 'tendon':
        Pe = invariants['Pe']
        pinv_B = invariants['pinv_B']
        Bp = invariants['Bp']
        e = invariants['e']
        Kp = invariants['Kp']
        Kd = invariants['Kd']

        # eta_0 (numeric)
        eta_0 = np.concatenate((-twist, jac @ data.qvel - target_vel)).reshape(2*m, 1)

        N = 10  # prediction horizon
        gamma = 0.9
        dt = 0.005

        # Define decision variables (create fresh variables each time)
        nu = 2
        nq = model.nq
        qdd_k  = cp.Variable((nq, N))  # joint accelerations
        mu   = cp.Variable((m, N))      # mu[:,k] = mu_k
        u_k  = cp.Variable((nu, N))     # single applied control (keep as you had)
        eta_k = cp.Variable((2*m, N))    # eta_k[:,k] = eta at step k

        # Since eta is already an error-state [-twist; J qdot], the goal is eta -> 0
        eta_target = np.zeros((2*m, 1))

        # Initial condition constraint
        constraints = []
        constraints += [eta_k[:, 0:1] == eta_0]

        objective = 0
        # Linear discrete dynamics rollout as constraints
        for k in range(N - 1):
            constraints += [eta_k[:, k+1:k+2] == eta_k[:, k:k+1] + dt * (F @ eta_k[:, k:k+1] + G @ mu[:, k:k+1])]
            constraints += [mu[:, k:k+1] == jac @ qdd_k[:, k:k+1] + dJ_dt @ dq - target_acc.reshape(-1,1)] 
            constraints += [pinv_B @ (M @ qdd_k[:, k:k+1] + data.qfrc_bias.reshape(-1,1) 
                                      + data.qfrc_passive.reshape(-1,1)) == u_k[:, k:k+1]]
            constraints += [-1 <= u_k[:, k:k+1], u_k[:, k:k+1] <= 1 * np.ones((nu, 1))]
            eta_k1 = eta_k[:, k+1:k+2]
            mu_k1  = mu[:, k:k+1]
            mu_des_k = - Kp * eta_k[0:m, k:k+1] - 2 * Kd * eta_k[m:2*m, k:k+1]
            objective += (gamma**k) * (cp.sum_squares(mu_k1 - mu_des_k) + cp.sum_squares(eta_k1 - eta_target) 
                                    + 0.1 * cp.sum_squares(u_k[:, k:k+1]) + 0.1 * cp.sum_squares(qdd_k[:, k:k+1]))

        # Terminal penalty (use eta_k at terminal, not eta_next)
        eta_N = eta_k[:, N-1:N]
        objective += cp.quad_form(eta_N, Pe)
        objective = cp.Minimize(objective)
        
        prob = cp.Problem(objective, constraints)
        # Warm start with previous solution if available
        if previous_solution is not None:
            try:
                u_k.value[:, 0] = previous_solution['u']
            except:
                pass  # If warm start fails, proceed without it    
        try:
            prob.solve(solver=cp.SCS, verbose=False, warm_start=True)
            if u_k.value is not None:
                data.ctrl = np.squeeze(Bp @ u_k.value[:, 0]) 
                # Cache solution for next iteration
                current_solution = {
                    'u': u_k.value.copy(),
                }
                # print(f"converged\n")
                return current_solution, u_k.value[:, 0]
            else:
                print(f"failed convergence - no solution\n")
                return previous_solution, u_k.value[:, 0]
        except Exception as e:
            print(f"failed convergence - exception: {e}\n")
            return previous_solution, u_k.value[:, 0]
    
    elif model_name == 'helix':
        Pe = invariants['Pe']
        B = invariants['B']
        pinv_B = invariants['pinv_B']
        sel = invariants['sel']
        e = invariants['e']
        Kp = invariants['Kp']
        Kd = invariants['Kd']

        # eta_0 (numeric)
        eta_0 = np.concatenate((-twist, jac @ data.qvel - target_vel)).reshape(2*m, 1)

        N = 10  # prediction horizon
        gamma = 0.9
        dt = 0.005

        # Define decision variables (create fresh variables each time)
        nu = 9
        nq = model.nq
        qdd_k  = cp.Variable((nq, N))  # joint accelerations
        mu   = cp.Variable((m, N))      # mu[:,k] = mu_k
        u_k  = cp.Variable((nu, N))     # single applied control (keep as you had)
        eta_k = cp.Variable((2*m, N))    # eta_k[:,k] = eta at step k

        # Since eta is already an error-state [-twist; J qdot], the goal is eta -> 0
        eta_target = np.zeros((2*m, 1))

        # Initial condition constraint
        constraints = []
        constraints += [eta_k[:, 0:1] == eta_0]

        objective = 0
        # Linear discrete dynamics rollout as constraints
        for k in range(N - 1):
            constraints += [eta_k[:, k+1:k+2] == eta_k[:, k:k+1] + dt * (F @ eta_k[:, k:k+1] + G @ mu[:, k:k+1])]
            constraints += [mu[:, k:k+1] == jac @ qdd_k[:, k:k+1] + dJ_dt @ dq - target_acc.reshape(-1,1)] 
            constraints += [pinv_B @ (M @ qdd_k[:, k:k+1] + data.qfrc_bias.reshape(-1,1) 
                                      + data.qfrc_passive.reshape(-1,1)) == u_k[:, k:k+1]]
            constraints += [-25 * sel <= u_k[:, k:k+1], u_k[:, k:k+1] <= 25 * np.ones((nu, 1))]
            eta_k1 = eta_k[:, k+1:k+2]
            mu_k1  = mu[:, k:k+1]
            mu_des_k = - Kp * eta_k[0:m, k:k+1] - 2*Kd * eta_k[m:2*m, k:k+1]
            objective += (gamma**k) * (cp.sum_squares(mu_k1 - mu_des_k) + cp.sum_squares(eta_k1 - eta_target) 
                                    + 0.1 * cp.sum_squares(u_k[:, k:k+1]) + 0.1 * cp.sum_squares(qdd_k[:, k:k+1]))

        # Terminal penalty (use eta_k at terminal, not eta_next)
        eta_N = eta_k[:, N-1:N]
        objective += cp.quad_form(eta_N, Pe)
        objective = cp.Minimize(objective)
        
        prob = cp.Problem(objective, constraints)
        # Warm start with previous solution if available
        if previous_solution is not None:
            try:
                u_k.value[:, 0] = previous_solution['u']
            except:
                pass  # If warm start fails, proceed without it    
        try:
            prob.solve(solver=cp.SCS, verbose=False, warm_start=True)
            if u_k.value is not None:
                data.ctrl = np.squeeze(B @ u_k.value[:, 0]) 
                # Cache solution for next iteration
                current_solution = {
                    'u': u_k.value.copy(),
                }
                # print(f"converged\n")
                return current_solution, u_k.value[:, 0]
            else:
                print(f"failed convergence - no solution\n")
                return previous_solution, u_k.value[:, 0]
        except Exception as e:
            print(f"failed convergence - exception: {e}\n")
            return previous_solution, u_k.value[:, 0]
    
    elif model_name == 'spirob':
        Pe = invariants['Pe']
        sel = invariants['sel']
        e = invariants['e']
        Kp = invariants['Kp']
        Kd = invariants['Kd']

        # Sprirob input matrix depends on current configuration, so we compute it at the current state
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

        pinv_B = np.linalg.pinv(B)

        # eta_0 (numeric)
        eta_0 = np.concatenate((-twist, jac @ data.qvel - target_vel)).reshape(2*m, 1)

        N = 10  # prediction horizon
        gamma = 0.9
        dt = 0.005

        # Define decision variables (create fresh variables each time)
        nu = model.nu
        nq = model.nq
        qdd_k  = cp.Variable((nq, N))  # joint accelerations
        mu   = cp.Variable((m, N))      # mu[:,k] = mu_k
        u_k  = cp.Variable((nu, N))     # single applied control (keep as you had)
        eta_k = cp.Variable((2*m, N))    # eta_k[:,k] = eta at step k

        # Since eta is already an error-state [-twist; J qdot], the goal is eta -> 0
        eta_target = np.zeros((2*m, 1))

        # Initial condition constraint
        constraints = []
        constraints += [eta_k[:, 0:1] == eta_0]

        objective = 0
        # Linear discrete dynamics rollout as constraints
        for k in range(N - 1):
            constraints += [eta_k[:, k+1:k+2] == eta_k[:, k:k+1] + dt * (F @ eta_k[:, k:k+1] + G @ mu[:, k:k+1])]
            constraints += [mu[:, k:k+1] == jac @ qdd_k[:, k:k+1] + dJ_dt @ dq - target_acc.reshape(-1,1)] 
            constraints += [pinv_B @ (M @ qdd_k[:, k:k+1] + data.qfrc_bias.reshape(-1,1) 
                                      + data.qfrc_passive.reshape(-1,1)) == u_k[:, k:k+1]]
            constraints += [np.zeros((nu,1)) >= u_k[:, k:k+1]]
            eta_k1 = eta_k[:, k+1:k+2]
            mu_k1  = mu[:, k:k+1]
            mu_des_k = -Kp * eta_k[0:m, k:k+1] - 2*Kd * eta_k[m:2*m, k:k+1]
            objective += (gamma**k) * (cp.sum_squares(mu_k1 - mu_des_k) + cp.sum_squares(eta_k1 - eta_target) 
                                    + 0.1 * cp.sum_squares(u_k[:, k:k+1]) + 0.1 * cp.sum_squares(qdd_k[:, k:k+1]))

        # Terminal penalty (use eta_k at terminal, not eta_next)
        eta_N = eta_k[:, N-1:N]
        objective += cp.quad_form(eta_N, Pe)
        objective = cp.Minimize(objective)
        
        prob = cp.Problem(objective, constraints)
        # Warm start with previous solution if available
        if previous_solution is not None:
            try:
                u_k.value[:, 0] = previous_solution['u']
            except:
                pass  # If warm start fails, proceed without it    
        try:
            prob.solve(solver=cp.SCS, verbose=False, warm_start=True)
            if u_k.value is not None:
                data.ctrl = np.squeeze(u_k.value[:, 0]) 
                # Cache solution for next iteration
                current_solution = {
                    'u': u_k.value.copy(),
                }
                # print(f"converged\n")
                return current_solution, u_k.value[:, 0]
            else:
                print(f"failed convergence - no solution\n")
                return previous_solution, u_k.value[:, 0]
        except Exception as e:
            print(f"failed convergence - exception: {e}\n")
            return previous_solution, u_k.value[:, 0]

    # # eta_0 (numeric)
    # eta_0 = np.concatenate((-twist, jac @ data.qvel)).reshape(6, 1)

    # N = 10  # prediction horizon
    # gamma = 0.9
    # dt = 1.0 / 2000

    # # Define decision variables (create fresh variables each time)
    # nu = 2
    # mu = cp.Variable(shape=(3, N))      # mu[:,k] = mu_k
    # u_k  = cp.Variable((nu, 1))     # single applied control (keep as you had)
    # eta_k = cp.Variable((6, N))    # eta_k[:,k] = eta at step k

    # # Since eta is already an error-state [-twist; J qdot], the goal is eta -> 0
    # eta_target = np.zeros((6, 1))

    # # qdd must be a CVXPY expression (and keep only k=0 since you apply u_k once)
    # Jpinv = cp.Constant(np.linalg.pinv(jac))
    # qdd = Jpinv @ (mu[:, 0:1] - dJ_dt @ dq)   # (nv x 1)

    # # Initial condition constraint
    # constraints = []
    # constraints += [eta_k[:, 0:1] == eta_0]
    # objective = 0.0

    # # Linear discrete dynamics rollout as constraints
    # for k in range(N - 1):
    #     constraints += [eta_k[:, k+1:k+2] == eta_k[:, k:k+1] + dt * (F @ eta_k[:, k:k+1] + G @ mu[:, k:k+1])]
    #     eta_k1 = eta_k[:, k+1:k+2]
    #     mu_k1  = mu[:, k:k+1]
    #     mu_des_k = -Kp * eta_k[0:3, k:k+1] - Kd * eta_k[3:6, k:k+1]
    #     objective += (gamma**k) * (cp.sum_squares(mu_k1 - mu_des_k) + (gamma**k)*cp.sum_squares(eta_k1 - eta_target))

    # # Terminal penalty (use eta_k at terminal, not eta_next)
    # eta_N = eta_k[:, N-1:N]
    # objective += 10*cp.quad_form(eta_N, Pe)
    # objective = cp.Minimize(objective + 0.2 * cp.sum_squares(u_k) )# + 0.2 * cp.square(cp.norm(qdd)))

    # # Inverse dynamics constraint
    # constraints += [pinv_B @ (M @ qdd + data.qfrc_bias.reshape(-1,1) - data.qfrc_passive.reshape(-1,1)) == u_k,
    #     -1.0 <= u_k,
    #     1.0 >= u_k,
    # ]

    # prob = cp.Problem(objective=objective, constraints=constraints)
    
    # # Warm start with previous solution if available
    # if previous_solution is not None:
    #     try:
    #         u_k.value = previous_solution['u']
    #         qdd.value = previous_solution['qdd'] 
    #     except:
    #         pass  # If warm start fails, proceed without it
    # try:
    #     prob.solve(solver=cp.SCS, verbose=False, warm_start=True)

    #     if u_k.value is not None:
    #         u = u_k.value.copy()

    #         # data.ctrl[:] = np.squeeze(u_tendon)
    #         data.ctrl[:] = np.squeeze(Bp @ u)

    #         current_solution = {
    #             'u': u.copy(),
    #         }

    #         return current_solution, u.copy()

    #     else:
    #         print(f"failed convergence - no solution\n")
    #         return previous_solution
    # except Exception as e:
    #     print(f"failed convergence - exception: {e}\n")
    #     return previous_solution




def simulate_model(headless=False,control_scheme=None, target_pos=None,controller=None, experiment = None, model_name=None):
    
    model = robot(model_name)
    model.opt.gravity = (0, 0, -9.81)
    data = mujoco.MjData(model)

    # Pre-compute invariant matrices
    print("Pre-computing invariant matrices...")
    invariants = precompute_invariants(model, model_name)

    model.jnt_stiffness[:] = invariants['stiffness']
    model.dof_damping[:] = invariants['damping']

    if model_name == 'helix':
        data.qpos[2] = 0.0
        model.jnt_range[range(2,len(data.qpos),3)] = [[-0.001, 0.03/2] for i in range(2,len(data.qpos),3)]
        model.jnt_stiffness[range(2,len(data.qpos),3)] = 50
    
    if model_name == 'spirob':
        # For the SpiRob, this should be a straight configuration
        data.qpos[:] = 0.0
        
    # Initialize solution cache
    previous_solution = None

    sim_ts = dict(
        ts=[],
        base_pos=[],
        base_vel=[],
        base_acc=[],
        base_force=[],
        base_torque=[],
        q=[],
        qvel=[],
        ctrl=[],
        actuator_force=[],
        qfrc_fluid=[],
        q_des=[],
    )

    q_des = np.ones(data.qpos.shape[0])*0.2

    V_log = []
    task_error_log = []
    u_log = []
    time_log = []
    t = 0.0
    dt = model.opt.timestep

    log_frequency = 5  # Log every 5 steps instead of every step
    step_count = 0
    threshold  = 1e-3

    if headless:
        while True:
            step_start = time.time()
            if experiment == 'tracking':
                omega,target = circular_trajectory(t, model_name)
            elif experiment == 'set':
                target = set_target(target_pos, model_name)
            if control_scheme == 'id_clf_qp':
                V, task_error, previous_solution, u = controller(experiment, control_scheme, model_name, 
                                                                 target, model, data, invariants, previous_solution)
            elif control_scheme == 'impedance':
                task_error, u = controller(experiment, model_name, control_scheme, model_name, target, model, data, invariants)
            elif control_scheme == 'mpc':
                task_error, previous_solution, u = controller(experiment, control_scheme, model_name, 
                                                              target, model, data, invariants, previous_solution)
            mujoco.mj_step(model, data)
            
            # Only log every N steps to reduce overhead
            if step_count % log_frequency == 0:
                # print(f"Sim time: {data.time:.3f}s")
                
                sim_ts["ts"].append(data.time)
                # extract the sensor data
                sim_ts["base_pos"].append(data.sensordata[:3].copy())
                sim_ts["base_vel"].append(data.sensordata[3:6].copy())
                sim_ts["base_acc"].append(data.sensordata[6:9].copy())
                sim_ts["base_force"].append(data.sensordata[9:12].copy())
                sim_ts["base_torque"].append(data.sensordata[12:15].copy())
                sim_ts["q"].append(data.qpos.copy())
                sim_ts["qvel"].append(data.qvel.copy())
                sim_ts["ctrl"].append(data.ctrl.copy())
                sim_ts["actuator_force"].append(data.actuator_force.copy())
                sim_ts["qfrc_fluid"].append(data.qfrc_fluid.copy())
                sim_ts["q_des"].append(q_des.copy())

                if control_scheme == 'id_clf_qp':
                    V_log.append(V)

                task_error_log.append(task_error)
                u_log.append(u.squeeze().copy())
                time_log.append(t)
            
            step_count += 1

            if experiment == 'tracking' and len(task_error_log) >= 2 and t >= 4*np.pi/omega:
                break
            elif experiment == 'set' and len(task_error_log) >= 2 and abs(task_error_log[-1] - task_error_log[-2]) / dt <= threshold:
                break
            t += dt

    else:
        # Run simulation with viewer
        with mujoco.viewer.launch_passive(model, data) as viewer:
            while viewer.is_running():
                if experiment == 'tracking':
                    omega,target = circular_trajectory(t, model_name)
                elif experiment == 'set':
                    target = set_target(target_pos, model_name)
                if control_scheme == 'id_clf_qp':
                    V, task_error, previous_solution, u = controller(experiment, control_scheme, model_name, 
                                                                     target, model, data, invariants, previous_solution)
                elif control_scheme == 'impedance':
                    task_error, u = controller(experiment, control_scheme, model_name, target, model, data, invariants)
                elif control_scheme == 'mpc':
                    task_error, previous_solution, u = controller(experiment, control_scheme, model_name, 
                                                                  target, model, data, invariants, previous_solution)
                mujoco.mj_step(model, data)
                
                # Only log every N steps to reduce overhead
                if step_count % log_frequency == 0:
                    # print(f"Sim time: {data.time:.3f}s")
                    
                    sim_ts["ts"].append(data.time)
                    # extract the sensor data
                    sim_ts["base_pos"].append(data.sensordata[:3].copy())
                    sim_ts["base_vel"].append(data.sensordata[3:6].copy())
                    sim_ts["base_acc"].append(data.sensordata[6:9].copy())
                    sim_ts["base_force"].append(data.sensordata[9:12].copy())
                    sim_ts["base_torque"].append(data.sensordata[12:15].copy())
                    sim_ts["q"].append(data.qpos.copy())
                    sim_ts["qvel"].append(data.qvel.copy())
                    sim_ts["ctrl"].append(data.ctrl.copy())
                    sim_ts["actuator_force"].append(data.actuator_force.copy())
                    sim_ts["qfrc_fluid"].append(data.qfrc_fluid.copy())
                    sim_ts["q_des"].append(q_des.copy())

                    if control_scheme == 'id_clf_qp':
                        V_log.append(V)

                    task_error_log.append(task_error)
                    u_log.append(u.squeeze().copy())
                    time_log.append(t)
                
                step_count += 1

                if experiment == 'tracking' and len(task_error_log) >= 2 and t >= 4*np.pi/omega:
                    break
                elif experiment == 'set' and len(task_error_log) >= 2 and abs(task_error_log[-1] - task_error_log[-2]) / dt <= threshold:
                    break

                # Pick up changes to the physics state, apply perturbations, update options from GUI.
                viewer.sync()
                t += dt

                # Rudimentary time keeping, will drift relative to wall clock.
                # Removed sleep to run as fast as possible for 1 second of sim time
                # time_until_next_step = model.opt.timestep - (time.time() - step_start)
                # if time_until_next_step > 0:
                #     time.sleep(time_until_next_step)

    
    print(f"Simulation finished after {sim_ts['ts'][-1]} seconds")

    if control_scheme == 'id_clf_qp':
        return V_log, task_error_log, time_log, sim_ts, u_log
    else:
        return task_error_log, time_log, sim_ts, u_log