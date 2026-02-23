import numpy as np
import mujoco
from pathlib import Path
from scipy import linalg


class Robot:
    """Unified robot class that encapsulates robot-specific configurations and MuJoCo model"""
    
    def __init__(self, model_name: str):
        self.model_name = model_name
        self.model = self._load_model()
        self.data = mujoco.MjData(self.model)
        self.model.opt.gravity = (0, 0, -9.81)
        self._setup_robot_config()
        
    def _load_model(self):
        """Load MuJoCo model from standard path convention"""
        model_path = Path("mujoco_models") / self.model_name / f"{self.model_name}_control.xml"
        return mujoco.MjModel.from_xml_path(str(model_path.absolute()))
    
    def _setup_robot_config(self):
        """Configure robot-specific parameters"""
        if self.model_name == 'tendon':
            self.task_dim = 3
            self.control_limits = (-1.0, 1.0)
            self.nu = 2
            # Static input matrices
            self.B = np.array([[0.1, 0.0], [0.1, 0.0], [0.0, 0.1], [0.0, 0.1]])
            self.B_applied = np.array([[1, 0.0], [-1, 0.0], [0.0, 1], [0.0, -1]])
            self.pinv_B = np.linalg.pinv(self.B)
            # Control gains
            self.Kp = 500
            self.Kd = 2 * np.sqrt(self.Kp)
            self.damping, self.stiffness = 0.02, 0.01
            self.e = 0.2
            # Passive force sign (tendon uses -data.qfrc_passive)
            self.passive_sign = -1
            # Regularization coefficients for optimization
            self.reg_qdd = 0.1
            self.reg_u = 0.1
            self.reg_dl = 1000
            # MPC-specific coefficients
            self.mpc_task_weight = 1.0
            self.mpc_null_weight = 0.0
            self.mpc_terminal_weight = 10.0
            # Control constraint bounds
            self.lower_bounds = np.full((self.nu, ), self.control_limits[0])
            self.upper_bounds = np.full((self.nu, ), self.control_limits[1])
            
        elif self.model_name == 'helix':
            self.task_dim = 6
            self.control_limits = (-25.0, 25.0)
            self.nu = 9
            # Static input matrix
            self.B = np.zeros((36, 9))
            for i in range(3):
                for j in range(4):
                    row_start = i * 12 + j * 3
                    col_start = i * 3
                    self.B[row_start:row_start+3, col_start:col_start+3] = np.eye(3)
            self.B_applied = self.B  # Use same matrix
            self.pinv_B = np.linalg.pinv(self.B)
            # Selection matrix
            self.sel = np.ones((self.nu,))
            self.sel[[2, 5, 8]] = 0.0
            # Control gains
            self.Kp, self.Kd = 500, 2 * np.sqrt(500)
            self.damping, self.stiffness = 0.2, 0.1
            self.e = 0.5
            # Passive force sign (helix uses +data.qfrc_passive)
            self.passive_sign = 1
            # Regularization coefficients for optimization
            self.reg_qdd = 0.5
            self.reg_u = 0.5
            self.reg_dl = 1000
            # MPC-specific coefficients
            self.mpc_task_weight = 1.0
            self.mpc_null_weight = 0.0
            self.mpc_terminal_weight = 1.0
            # Control constraint bounds (incorporate selection logic)
            self.lower_bounds = self.control_limits[0] * self.sel
            self.upper_bounds = np.full((self.nu, ), self.control_limits[1])
            
        elif self.model_name == 'spirob':
            self.task_dim = 6
            self.control_limits = (-100.0, 0.0)
            self.nu = self.model.nu
            # Dynamic B matrix - computed at runtime
            self.B = None
            self.B_applied = np.eye(self.nu)  # Use same as B
            self.pinv_B = None
            self.sel = np.ones((self.nu, 1))
            # Control gains
            self.Kp, self.Kd = 1000.0, 2 * np.sqrt(1000.0)
            self.damping, self.stiffness = 0.01, 0.01
            self.e = 0.01
            # Passive force sign (spirob uses -data.qfrc_passive)
            self.passive_sign = -1
            # Regularization coefficients for optimization  
            self.reg_qdd = 0.5
            self.reg_u = 0.5
            self.reg_dl = 1000
            # MPC-specific coefficients
            self.mpc_task_weight = 1.0
            self.mpc_null_weight = 0.0
            self.mpc_terminal_weight = 10.0
            # Passive force sign (spirob uses -data.qfrc_passive)
            self.passive_sign = -1
            # Control constraint bounds
            self.lower_bounds = np.full((self.nu,), self.control_limits[0])
            self.upper_bounds = np.full((self.nu,), self.control_limits[1])
            
        # Compute Control Lyapunov Function matrices
        self._setup_clf_matrices()
        
    def _setup_clf_matrices(self):
        """Pre-compute invariant matrices for control"""
        m = self.task_dim
        
        # Control Lyapunov Function matrices
        self.F = np.zeros((2*m, 2*m))
        self.F[:m, m:] = np.eye(m, m)
        self.G = np.zeros((2*m, m))
        self.G[m:, :] = np.eye(m)
        
        self.Pe = linalg.block_diag(np.eye(m) / self.e, np.eye(m)).T @ linalg.solve_continuous_are(self.F, self.G, np.eye(2*m), np.eye(m)) @ linalg.block_diag(np.eye(m) / self.e, np.eye(m))
        
    def get_passive_forces(self):
        """Get passive forces with correct sign for each robot"""
        return self.passive_sign * self.data.qfrc_passive
    
    def get_control_constraints(self, u_var):
        """Get control constraints for optimization (robot-agnostic)"""
        constraints = []
        constraints.append(self.lower_bounds <= u_var)
        constraints.append(u_var <= self.upper_bounds)
        return constraints
    
    def update_input_matrix(self):
        """Update input matrix - static for tendon/helix, dynamic for spirob"""
        if self.model_name == 'spirob':
            # Compute B matrix dynamically for spirob
            nv = self.model.nv
            self.B = np.zeros((nv, self.nu))
            
            data_temp = mujoco.MjData(self.model)
            data_temp.qpos[:] = self.data.qpos
            data_temp.qvel[:] = self.data.qvel
            
            mujoco.mj_forward(self.model, data_temp)
            
            for i in range(self.nu):
                data_temp.ctrl[:] = 0.0
                data_temp.ctrl[i] = 1.0
                mujoco.mj_forward(self.model, data_temp)
                self.B[:, i] = data_temp.qfrc_actuator.copy()
                
            self.pinv_B = np.linalg.pinv(self.B)
        # For tendon/helix, B matrix is static and already computed
    
    def initialize_simulation_state(self):
        """Initialize robot-specific simulation state and model parameters."""
        print("Initializing robot configuration...")
        self.model.jnt_stiffness[:] = self.stiffness
        self.model.dof_damping[:] = self.damping
        if self.model_name == 'helix':
            self.data.qpos[2] = 0.0
            self.model.jnt_range[range(2,len(self.data.qpos),3)] = [[-0.001, 0.03/2] for i in range(2,len(self.data.qpos),3)]
            self.model.jnt_stiffness[range(2,len(self.data.qpos),3)] = 50
        elif self.model_name == 'spirob':
            # For the SpiRob, this should be a straight configuration
            self.data.qpos[:] = 0.0

    def compute_jacobian_derivative(self, site_id, h=1e-6):
        """
        Compute the time derivative of the Jacobian for this robot
        
        Parameters:
        - site_id: ID of the site for Jacobian computation
        - h: Small positive step for numerical differentiation
        
        Returns:
        - Jdot: The time derivative of the Jacobian
        """
        # Step 1: Update kinematics
        mujoco.mj_kinematics(self.model, self.data)
        mujoco.mj_comPos(self.model, self.data)
        
        # Step 2: Compute the initial Jacobian
        J = np.zeros((6, self.model.nv))  # Assuming a 6xnv Jacobian for full spatial representation
        mujoco.mj_jacSite(self.model, self.data, J[:3], J[3:], site_id)
        
        # Step 3: Integrate position using velocity
        qpos_backup = np.copy(self.data.qpos)  # Backup original qpos
        mujoco.mj_integratePos(self.model, self.data.qpos, self.data.qvel, h)
        
        # Step 4: Update kinematics again
        mujoco.mj_kinematics(self.model, self.data)
        mujoco.mj_comPos(self.model, self.data)
        
        # Step 5: Compute the new Jacobian
        Jh = np.zeros((6, self.model.nv))
        mujoco.mj_jacSite(self.model, self.data, Jh[:3], Jh[3:], site_id)
        
        # Step 6: Compute Jdot
        Jdot = (Jh - J) / h
        
        # Step 7: Restore qpos
        self.data.qpos[:] = qpos_backup
        
        return Jdot
    
    def get_coriolis_and_gravity(self):
        """Get Coriolis and gravity forces"""
        nv = self.model.nv  # number of degrees of freedom

        # Calculate gravity vector
        g = np.zeros(nv)
        dummy = np.zeros(nv,)
        mujoco.mj_factorM(self.model, self.data)  # Compute sparse M factorization
        mujoco.mj_rne(self.model, self.data, 0, dummy)  # Run RNE with zero acceleration and velocity
        g = self.data.qfrc_bias.copy()

        # Calculate Coriolis matrix
        C = np.zeros((nv, nv))
        q_vel = self.data.qvel.copy()

        # Compute each column of C using finite differences
        eps = 1e-6
        for i in range(nv):
            # Save current state
            vel_orig = q_vel.copy()

            # Perturb velocity
            q_vel[i] += eps
            self.data.qvel = q_vel

            # Calculate forces with perturbed velocity
            mujoco.mj_rne(self.model, self.data, 0, dummy)
            tau_plus = self.data.qfrc_bias.copy()
            # Restore original velocity
            q_vel = vel_orig
            self.data.qvel = q_vel

            # Compute column of C using finite difference
            C[:, i] = (tau_plus - self.data.qfrc_bias) / eps
        return C, g
    
    # ===== ON-DEMAND PHYSICS INTERFACE =====
    
    def get_mass_matrix(self):
        """Compute and return mass matrix on-demand"""
        M = np.zeros((self.model.nv, self.model.nv))
        mujoco.mj_fullM(self.model, M, self.data.qM)
        return M
        
    def get_mass_matrix_inverse(self):
        """Compute and return mass matrix inverse on-demand"""
        M_inv = np.zeros((self.model.nv, self.model.nv))
        mujoco.mj_solveM(self.model, self.data, M_inv, np.eye(self.model.nv))
        return M_inv
        
    def get_jacobian(self, site_name="ee"):
        """Compute and return end-effector Jacobian on-demand"""
        jac = np.zeros((6, self.model.nv))
        site_id = self.model.site(site_name).id
        mujoco.mj_jacSite(self.model, self.data, jac[:3], jac[3:], site_id)
        return jac[:self.task_dim, :]
        
    def get_jacobian_derivative(self, site_name="ee", h=1e-6):
        """Compute and return Jacobian derivative on-demand"""
        site_id = self.model.site(site_name).id
        dJ_dt = self.compute_jacobian_derivative(site_id, h)
        return dJ_dt[:self.task_dim, :]
        
    def get_joint_velocities(self):
        """Get current joint velocities"""
        return self.data.qvel.copy()
        
    def get_bias_forces(self):
        """Get bias forces (Coriolis + gravity)"""
        return self.data.qfrc_bias.copy()
        
    def compute_target_data(self, experiment, target):
        """Compute target position, velocities and task error"""
        mocap_name = "target"
        mocap_id = self.model.body(mocap_name).mocapid[0]
        site_name = "ee"
        site_id = self.model.site(site_name).id
        m = self.task_dim
        
        # Update target position in simulation
        if experiment == 'tracking':
            self.data.mocap_pos[mocap_id] = target['pos']
            if m == 3:
                target_vel = target['vel']
                target_acc = target['acc']
            elif m == 6:
                target_vel = np.hstack([target['vel'], [0, 0, 0]])
                target_acc = np.hstack([target['acc'], [0, 0, 0]])
        else:  # experiment == 'set'
            self.data.mocap_pos[mocap_id] = target
            target_vel = np.zeros(m)
            target_acc = np.zeros(m)
            
        # Compute task error (twist)
        twist = np.zeros(m)
        twist[:3] = self.data.mocap_pos[mocap_id] - self.data.site(site_id).xpos
        twist[3:] = 0.0
        
        return target_vel, target_acc, twist
        
    def update_kinematics(self):
        """Update robot kinematics and COM"""
        mujoco.mj_kinematics(self.model, self.data)
        mujoco.mj_comPos(self.model, self.data)
        
    def step_simulation(self):
        """Advance simulation by one timestep"""
        mujoco.mj_step(self.model, self.data)
        
    def apply_control_input(self, u):
        """Apply control input to robot actuators"""
        self.data.ctrl[:] = self.B_applied @ np.clip(u, self.lower_bounds, self.upper_bounds)
