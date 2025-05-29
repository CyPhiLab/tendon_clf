import mujoco
import numpy as np
from mujoco import viewer
# from mujoco import mjx
import time
import cvxpy as cp
from scipy import linalg

# MJCF XML as string
mjcf = """
<mujoco model="4link_tendon_planar">
    <compiler angle="radian"/>
    <option gravity="0 0 -9.81" integrator="implicitfast"/>

    <default>
        <joint type="hinge" axis="0 1 0" limited="true" range="-1.57 1.57" damping="0.01"/>
        <geom type="capsule" size="0.005 0.03" rgba="0.5 0.5 0.5 1" mass="0.01"/>
    </default>


    <worldbody>
        <camera name="side_view" pos="0 0.1 0.05" xyaxes="1 0 0  0 0 1"/>
        <body name="link1" pos="0 0 0">
            <joint name="joint1" />
            <geom fromto="0 0 0 0.06 0 0" size="0.005"/>
            <body name="link2" pos="0.06 0 0">
                <joint name="joint2"/>
                <geom fromto="0 0 0 0.06 0 0" size="0.005"/>
                <body name="link3" pos="0.06 0 0">
                    <joint name="joint3"/>
                    <geom fromto="0 0 0 0.06 0 0" size="0.005"/>
                    <body name="link4" pos="0.06 0 0">
                        <joint name="joint4"/>
                        <geom fromto="0 0 0 0.06 0 0" size="0.005"/>
                        <body name="attachment" pos="0.03 0.0 0.0">
                              <site name="ee" rgba="1 0 0 1" size="0.001" group="1"/>
                        </body>
                    </body>
                </body>
            </body>
        </body>
    </worldbody>

    <!-- Tendon-based antagonistic actuation -->
    <tendon>
        <!-- Tendon pair A -->
        <fixed name="tendon_a_flex">
            <joint joint="joint1" coef="1"/>
            <joint joint="joint2" coef="1"/>
        </fixed>
        <fixed name="tendon_a_ext">
            <joint joint="joint1" coef="-1"/>
            <joint joint="joint2" coef="-1"/>
        </fixed>

        <!-- Tendon pair B -->
        <fixed name="tendon_b_flex">
            <joint joint="joint3" coef="1"/>
            <joint joint="joint4" coef="1"/>
        </fixed>
        <fixed name="tendon_b_ext">
            <joint joint="joint3" coef="-1"/>
            <joint joint="joint4" coef="-1"/>
        </fixed>
    </tendon>


    <!-- Actuators for tendons -->
    <actuator>
        <motor tendon="tendon_a_flex" ctrlrange="0 1" gear="0.1"/>
        <motor tendon="tendon_a_ext" ctrlrange="0 1" gear="0.1"/>
        <motor tendon="tendon_b_flex" ctrlrange="0 1" gear="0.1"/>
        <motor tendon="tendon_b_ext" ctrlrange="0 1" gear="0.1"/>
    </actuator>
</mujoco>

"""  # Replace with the MJCF content provided above


# Load model from string
model = mujoco.MjModel.from_xml_string(mjcf)
data = mujoco.MjData(model)
model.jnt_stiffness[:] = 0.01

damping_ratio = 1.0

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

def controller(model, data):
    jac = np.zeros((6, model.nv))
    twist = np.zeros(3)
    M_inv = np.zeros((model.nv, model.nv))
    M = np.zeros((model.nv, model.nv))
    Kp_null = np.asarray([1] * model.nv)

    Kd_null = damping_ratio * 2 * np.sqrt(Kp_null)

    m = 3
    F = np.zeros((2*m,2*m))
    F[:m,m:] = np.eye(m,m)
    G = np.zeros((2*m,m))
    G[m:,:] = np.eye(m)
    e = 0.03
    Pe = linalg.block_diag(np.eye(m) / e, np.eye(m) ).T @ linalg.solve_continuous_are(F, G, np.eye(2*m), np.eye(m)) @ linalg.block_diag(np.eye(m) / e, np.eye(m) )
    pinv_B = np.linalg.pinv(B)
    # print(pinv_B)
    site_name = "ee"
    site_id = model.site(site_name).id
    dx = np.array([0.1, 0.0, -0.15]) - data.site(site_id).xpos
    twist = dx 

    # u = np.zeros(np.shape(data.qpos[7:]))
    # print(f"[DEBUG] error: ", (data.qpos - q_des), "\n")
    # print(data.qpos)
    # print(u[2])
    # mujoco.mj_solveM(model, data, M_inv, np.eye(model.nv))
    # Jacobian.
    q = data.qpos
    mujoco.mj_kinematics(model,data)
    mujoco.mj_comPos(model,data)
    mujoco.mj_jacSite(model, data, jac[:3], jac[3:], site_id)
    
    # Compute the task-space inertia matrix.
    mujoco.mj_solveM(model, data, M_inv, np.eye(model.nv))
    mujoco.mj_fullM(model, M, data.qM)
    
    dJ_dt_full = compute_jacobian_derivative(model, data, site_id)
    dJ_dt = dJ_dt_full[:3,:]
    jac = jac[:3,:]
    nu = 4
    nq = model.nq
    u = cp.Variable(shape=(2, 1))
    qdd = cp.Variable(shape=(nq, 1))
    dl = cp.Variable(shape=(1, 1))

    eta = np.concatenate((-twist,jac @ data.qvel))
    V = eta.T @ Pe @ eta


    _, g = get_coriolis_and_gravity(model, data)

    statics = (g + data.qfrc_spring)
    dq = data.qvel.reshape(-1,1)
    ddq = Kp_null * (- data.qpos) - 4*Kd_null * data.qvel
    null = 0.1*B @ pinv_B @ ddq.reshape(-1,1)
    # damp = cp.square(cp.norm(B @ u - null))
    
    Bp = np.array([[0.1, 0.0], [0.1, 0.0], [0.0, 0.1], [0.0, 0.1]])
    stat = cp.square(cp.norm(Bp @ u - statics.reshape(-1,1)))
    qdd_cl = (Bp @ u - data.qfrc_bias.reshape(-1,1) - data.qfrc_passive.reshape(-1,1))
    Vw = (dq.T @ qdd_cl + (q*model.jnt_stiffness[:]).T @ dq)
    # print(np.shape(pinv_B @ (M @ qdd + data.qfrc_bias.reshape(-1,1) + data.qfrc_passive.reshape(-1,1))))
    dV = eta.T @ (F.T @ Pe + Pe @ F) @ eta + 2 * eta.T @ Pe @ G @ (dJ_dt @ dq + jac @ qdd)
    u_des = np.linalg.pinv(Bp)  @ (M @ np.linalg.pinv(jac) @ (100 * twist - 20 * (jac @ data.qvel) - dJ_dt @ data.qvel) + data.qfrc_bias + data.qfrc_passive) #+ 0.1 * pinv_B @ (np.eye(model.nv) - jac.T @ Jbar.T) @ ddq
    objective = cp.Minimize(cp.square(cp.norm(dJ_dt @ dq + jac @ qdd)) + 0.02*cp.square(cp.norm(qdd)) + 0.01*cp.square(cp.norm(u)) + 1000*cp.square(dl)  ) # objective
    # print(np.linalg.pinv(Bp))
    constraints = [ dV <= -2/e * V + dl,
                #    Vw <= - 0.001 * dq.T @ dq + 0.1*dl,
                    np.linalg.pinv(Bp) @ (M @ qdd + data.qfrc_bias.reshape(-1,1) + data.qfrc_passive.reshape(-1,1)) == u,
                    -1.0 <= u,
                    1.0 >= u]
    prob = cp.Problem(objective=objective, constraints=constraints)
    
    
    
    # try:
    prob.solve(verbose=False)
    # print([u.value[0][0],0.0])
    # print([np.max([u.value[0][0],0.0]), np.min([u.value[0][0],0.0]), np.max([u.value[1][0],0.0]), np.min([u.value[1][0],0.0])])
    u = np.array([np.max([u.value[0][0],0.0]), - np.min([u.value[0][0],0.0]), np.max([u.value[1][0],0.0]), - np.min([u.value[1][0],0.0])])
    # input()
    # print(u)
    # input()
    data.ctrl = np.squeeze(u)
        # data.ctrl = np.array([0.0]*4)
        # print(u.value)
        # print(qdd.value)
        # print(B @ u.value)
    print(dx)
        # print("damping: ", np.linalg.norm(B @ u.value - null), "statics: ", np.linalg.norm(B @ u.value - statics.reshape(-1,1)))
    # except:
    #     print(u.value)
    #     print(f"failed convergence\n")
    #     pass

def compute_B_matrix(model, data):
    nv, nu = model.nv, model.nu
    B = np.zeros((nv, nu))

    ctrl_backup = data.ctrl.copy()

    for i in range(nu):
        data.ctrl[:] = 0.0
        data.ctrl[i] = 1.0
        mujoco.mj_forward(model, data)
        B[:, i] = data.qfrc_actuator

    data.ctrl[:] = ctrl_backup  # Restore control inputs
    return B


# Print actuator matrix
B = compute_B_matrix(model, data)
print("Actuator matrix B (joint_accels = B @ actuator_forces):\n", B)

# Simulate
try:
    with viewer.launch_passive(model, data) as v:
        v.cam.azimuth = 90
        v.cam.elevation = 5
        v.cam.distance =  1
        v.cam.lookat = np.array([0.0 , 0.0 , -0.1])
        print("Press ESC to exit viewer.")
        while v.is_running():
            # You can set control inputs here, for example:
            # data.ctrl[:] = [0.5, 0.3, 0.1, 0.2]  # Example tendon control values
            controller(model,data)
            mujoco.mj_step(model, data)
            v.sync()
            time.sleep(0.001)
except ImportError:
    print("Viewer requires glfw, and a GUI environment.")

