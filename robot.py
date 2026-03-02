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
            self.T, _ = self.complete_basis(self.B.T)
            self.Tinv = np.linalg.inv(self.T)
            self.TinvT = self.Tinv.T 
            # Control gains
            self.Kp = 500
            self.Kd = 2 * np.sqrt(self.Kp)
            self.damping, self.stiffness = 0.02, 0.01
            self.e = 0.03
            # Passive force sign (tendon uses -data.qfrc_passive)
            self.passive_sign = -1
            # Regularization coefficients for optimization
            self.reg_qdd = 0.2
            self.reg_u = 0.2
            self.reg_null = 0.1
            self.reg_dl = 1000
            # MPC-specific coefficients
            self.mpc_task_weight = 1.0
            self.mpc_null_weight = 0.0
            self.mpc_terminal_weight = 10.0
            # Control constraint bounds
            self.lower_bounds = np.full((self.nu, ), self.control_limits[0])
            self.upper_bounds = np.full((self.nu, ), self.control_limits[1])
            
            # Default workspace bounds for CBF (tendon specific)  
            self.workspace_bounds = {
                'x_min': -0.3, 'x_max': 0.1,
                'y_min': -0.3, 'y_max': 0.3,
                'z_min': 0.05, 'z_max': 0.8
            }
            
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
            self.T, _ = self.complete_basis(self.B.T)
            self.Tinv = np.linalg.inv(self.T)
            self.TinvT = self.Tinv.T 
            # Selection matrix
            self.sel = np.ones((self.nu,))
            self.sel[[2, 5, 8]] = 0.0
            # Control gains
            self.Kp, self.Kd = 500, 2 * np.sqrt(500)
            self.damping, self.stiffness = 0.2, 0.1
            self.e = 0.9
            # Passive force sign (helix uses +data.qfrc_passive)
            self.passive_sign = 1
            # Regularization coefficients for optimization
            self.reg_qdd = 10.0
            self.reg_u = 1.0
            self.reg_null = 10.0
            self.reg_dl = 1000
            # MPC-specific coefficients
            self.mpc_task_weight = 1.0
            self.mpc_null_weight = 0.0
            self.mpc_terminal_weight = 10.0
            # Control constraint bounds (incorporate selection logic)
            self.lower_bounds = self.control_limits[0] * self.sel
            self.upper_bounds = np.full((self.nu, ), self.control_limits[1])
            
            # Default workspace bounds for CBF (helix specific)
            self.workspace_bounds = {
                'x_min': -0.1, 'x_max': 0.2,
                'y_min': -0.1, 'y_max': 0.0, 
                'z_min': -100, 'z_max': 0.4
            }
            # self.model.opt.disableflags |= mujoco.mjtDisableBit.mjDSBL_CONTACT
            
        elif self.model_name == 'spirob':
            self.task_dim = 6
            self.control_limits = (-100.0, 0.0)
            self.nu = self.model.nu
            # Dynamic B matrix - computed at runtime
            self.update_input_matrix()
            self.T, _ = self.complete_basis(self.B.T)
            self.Tinv = np.linalg.inv(self.T)
            self.TinvT = self.Tinv.T 
            # print("Initial B matrix for SpiRob:\n", np.round(self.B, 3))
            self.B_applied = np.eye(self.nu)  # Use same as B
            # self.pinv_B = None
            self.sel = np.ones((self.nu, 1))
            # Control gains
            self.Kp, self.Kd = 1000.0, 2 * np.sqrt(1000.0)
            self.damping, self.stiffness = 0.015, 0.01
            self.e = 0.05

            # Passive force sign (spirob uses -data.qfrc_passive)
            self.passive_sign = -1
            # Regularization coefficients for optimization  
            self.reg_qdd = 0.5
            self.reg_u = 0.5
            self.reg_null = 0.1
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
            
            # Default workspace bounds for CBF (spirob specific)
            self.workspace_bounds = {
                'x_min': -0.4, 'x_max': 0.2,
                'y_min': -0.4, 'y_max': 0.4,
                'z_min': 0.0, 'z_max': 0.3 
            }
            
        # Compute Control Lyapunov Function matrices
        self._setup_clf_matrices()
        
        # CBF configuration (disabled by default)
        self.enable_cbf = False
        self.cbf_alpha = 100.0  # CBF convergence parameter
        
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
            self.T, _ = self.complete_basis(self.B.T)
            self.Tinv = np.linalg.inv(self.T)
            self.TinvT = self.Tinv.T 
            self.B_applied = self.B  # Use same as B
        # For tendon/helix, B matrix is static and already computed

    def complete_basis(self, B, tol=1e-10, return_full=True):
        """
        Given B in R^{m x n} with m < n, compute J in R^{(n-r) x n}
        whose rows span the nullspace of B, where r = rank(B).

        If B has full row rank (r = m), then stacking [B; J] gives an n x n
        invertible matrix.
        """
        B = np.asarray(B, dtype=float)
        if B.ndim != 2:
            raise ValueError("B must be 2D")
        m, n = B.shape
        if m >= n:
            raise ValueError("Require m < n for basis completion by stacking rows.")

        U, s, Vt = np.linalg.svd(B, full_matrices=True)
        r = np.sum(s > tol * (s[0] if s.size else 1.0))

        N = Vt[r:, :]   # rows span nullspace(B)

        if return_full:
            T = np.vstack((B, N))
            return T, N
        return N
    
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
    
    def discrete_jacobian(self, x, u):
        """
        Use MuJoCo's mjd_transitionFD to compute A, B at (x,u).
        This is 1st derivative wrt x, u of the discrete transition function x_{k+1}=f(x_k,u_k).
        By default it uses the dimension 2*nv (position and velocity).
        Adjust if your system dimension is different.
        """
        nq = self.model.nq
        nv = self.model.nv

        Nx = 2 * nv
        Nu = self.model.nu
        
        # Set the state for this linearization point
        self.data.qpos[:] = x[:nq]
        self.data.qvel[:] = x[nq:nq+nv]
        mujoco.mj_forward(self.model, self.data)
        self.data.ctrl[:] = u
        
        # We now call mjd_transitionFD
        A = np.zeros((Nx, Nx))
        B = np.zeros((Nx, Nu))
        eps = 1e-5
        flg_centered = 1
        mujoco.mjd_transitionFD(self.model, self.data, eps, flg_centered, A, B, None, None)
        return A, B
    
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
    
    def get_site_position(self, site_name="ee"):
        """Get current position of a site"""
        site_id = self.model.site(site_name).id
        return self.data.site(site_id).xpos.copy()
        
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
        
    def get_safe_sites(self):
        """Get all sites named 'safe_n' for CBF constraints"""
        safe_sites = []
        for i in range(self.model.nsite):
            site_name = self.model.site(i).name
            if site_name and site_name.startswith('safe_'):
                try:
                    # Verify it follows safe_n pattern
                    int(site_name.split('_')[1])
                    safe_sites.append(site_name)
                except (IndexError, ValueError):
                    continue
        return sorted(safe_sites, key=lambda x: int(x.split('_')[1]))
        
    def get_workspace_cbf_data(self, workspace_bounds, alpha=100.0):
        """Compute workspace CBF constraints for all safe sites
        
        Args:
            workspace_bounds: dict with 'x_min', 'x_max', 'y_min', 'y_max', 'z_min', 'z_max'
            alpha: CBF parameter for exponential convergence
            
        Returns:
            list of dicts containing {c, J, dJ} for each active constraint
        """
        safe_sites = self.get_safe_sites()
        cbf_constraints = []
        dq = self.get_joint_velocities()
        
        for site_name in safe_sites[-1:]:  # Only consider the last few safe sites for workspace constraints
            print(f"Evaluating workspace CBF for site: {site_name}")
            pos = self.get_site_position(site_name)
            jac = self.get_jacobian(site_name)  # Only position part (3x nv)
            jac_dot = self.get_jacobian_derivative(site_name)
            
            # Check each workspace boundary
            bounds = [
                # (pos[0] - workspace_bounds['x_min'], jac[0, :], jac_dot[0, :]),  # x_min constraint
                (workspace_bounds['x_max'] - pos[0], -jac[0, :], -jac_dot[0, :]), # x_max constraint  
                # (pos[1] - workspace_bounds['y_min'], jac[1, :], jac_dot[1, :]),  # y_min constraint
                # (workspace_bounds['y_max'] - pos[1], -jac[1, :], -jac_dot[1, :]), # y_max constraint
                # (pos[2] - workspace_bounds['z_min'], jac[2, :], jac_dot[2, :]),  # z_min constraint
                # (workspace_bounds['z_max'] - pos[2], -jac[2, :], -jac_dot[2, :])  # z_max constraint
            ]
            print(f"  Position: {pos}")
            
            for c, J, dJ in bounds:
                if c < 0.5:  # Only add constraints that might become active
                    cbf_constraints.append({
                        'c': c,
                        'J': J,
                        'dJ': dJ,
                        'site': site_name
                    })
                    
        return cbf_constraints
        
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

