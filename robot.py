from xml.parsers.expat import model

import numpy as np
import mujoco
from pathlib import Path
from scipy import linalg


# Height the spirob_horz base is lifted to so the arm hangs in free space.
# utils.set_target/circular_trajectory place targets relative to this, so it
# lives here as the single source of truth.
SPIROB_HORZ_BASE_HEIGHT = 0.55


class Robot:
    """Unified robot class that encapsulates robot-specific configurations and MuJoCo model"""
    
    # Robots whose scene file does not live at the default
    # mujoco_models/<name>/<name>_control.xml path.  spirob_horz shares the
    # spirob/ directory because meshdir="assets" resolves relative to it.
    MODEL_PATH_OVERRIDES = {
        'spirob_horz': Path("mujoco_models") / "spirob" / "spirob_horz_control.xml",
    }

    # Robots driven through tendons, where B must be rebuilt from
    # data.actuator_moment every step rather than held static.
    TENDON_ROBOTS = ('spirob', 'spirob_horz')

    def __init__(self, model_name: str, control_scheme: str):
        self.model_name = model_name
        self.model = self._load_model()
        self.data = mujoco.MjData(self.model)
        self.model.opt.gravity = (0, 0, -9.81)
        self.control_scheme = control_scheme
        # Defaults that individual robot configs may override.
        self.pinv_rcond = None       # None -> numpy's default pinv cutoff
        self.override_passives = True  # blanket-assign stiffness/damping at init
        self.base_height = None      # None -> leave the base where the XML puts it
        self.k_v, self.k_e = 1.0, 0.0  # dcmotor gain / back-EMF (probed when needed)
        self.include_constraint_forces = False  # carry qfrc_constraint in h
        self._setup_robot_config()
        self.site_id = self.model.site('ee').id


    def _load_model(self):
        """Load MuJoCo model from standard path convention"""
        model_path = self.MODEL_PATH_OVERRIDES.get(
            self.model_name,
            Path("mujoco_models") / self.model_name / f"{self.model_name}_control.xml")
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
            self.damping, self.stiffness = 0.02, 0.01
            if self.control_scheme == 'impedance_QP':
                self.Kp = 1500
            else:
                self.Kp = 500
            self.Kd = 2 * np.sqrt(self.Kp)
            if self.control_scheme == 'clf_qp':
                self.e = 0.05
            else:
                self.e = 0.03
                
            # Passive force sign (tendon uses -data.qfrc_passive)
            self.passive_sign = -1
            # Regularization coefficients for optimization
            if self.control_scheme == 'impedance_QP':
                self.reg_qdd = 0.02
                self.reg_null = 0.0
            else:
                self.reg_qdd = 0.2
                self.reg_null = 0.1
            if self.control_scheme == 'clf_qp':
                self.reg_u = 0.5
            else:
                self.reg_u = 0.2

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
            self.T, _ = self.complete_basis(self.B.T)
            self.Tinv = np.linalg.inv(self.T)
            self.TinvT = self.Tinv.T 

            # Selection matrix
            self.sel = np.ones((self.nu,))
            self.sel[[2, 5, 8]] = 0.0
            # Control gains
            self.Kp, self.Kd = 500, 2 * np.sqrt(500)
            if self.control_scheme == 'osc' or self.control_scheme == 'impedance':
                self.Kp, self.Kd = 2000, 2 * np.sqrt(2000)
            elif self.control_scheme == 'uosc':
                self.Kp, self.Kd = 1000, 2 * np.sqrt(1000)

            self.damping, self.stiffness = 0.2, 0.2
            self.e = 0.05
            # Passive force sign (helix uses +data.qfrc_passive)
            self.passive_sign = 1
            # Regularization coefficients for optimization
            self.reg_qdd = 0.1
            self.reg_u = 0.2
            self.reg_null = 0.1
            self.reg_dl = 1000
            # MPC-specific coefficients
            self.mpc_task_weight = 1.0
            self.mpc_null_weight = 0.0
            self.mpc_terminal_weight = 10.0
            # Control constraint bounds (incorporate selection logic)
            self.lower_bounds = self.control_limits[0] * self.sel
            self.upper_bounds = np.full((self.nu, ), self.control_limits[1])
            # self.model.opt.disableflags |= mujoco.mjtDisableBit.mjDSBL_CONTACT
            
        elif self.model_name == 'spirob':
            self.task_dim = 6
            self.control_limits = (-100.0, 0.0)
            self.nu = self.model.nu
            # Dynamic B matrix - computed at runtime (via update_input_matrix after mj_fwdPosition)
            self.update_input_matrix()
            self.T, _ = self.complete_basis(self.B.T)
            self.Tinv = np.linalg.inv(self.T)
            self.TinvT = self.Tinv.T 
            # print("Initial B matrix for SpiRob:\n", np.round(self.B, 3))
            self.B_applied = np.eye(self.nu)  # Use same as B
            # self.pinv_B = None
            self.sel = np.ones((self.nu, 1))
            # Control gains
            if self.control_scheme == 'impedance_QP':
                self.Kp, self.Kd = 2000.0, 2 * np.sqrt(2000.0)
            else:
                self.Kp, self.Kd = 200.0, 2 * np.sqrt(200.0)
            self.damping, self.stiffness = 0.05, 0.01
            self.e = 0.01
        



            # Passive force sign (spirob uses -data.qfrc_passive)
            self.passive_sign = -1
            # Regularization coefficients for optimization  
            self.reg_qdd = 0.2
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

        elif self.model_name == 'spirob_horz':
            self.task_dim = 3
            # ctrl is a dcmotor voltage here, not a force: ctrlrange is (-12, 0) V.
            self.control_limits = (-12.0, 0.0)
            self.nu = self.model.nu
            # dcmotor force law: actuator_force = k_v*ctrl - k_e*actuator_velocity.
            # Probed from the compiled model so a retuned nominal="..." is picked up.
            self._probe_actuator_constants()
            # B's common mode (all three tendons pulling equally) barely moves the
            # arm, so B is near-singular; truncate it out of the pseudo-inverse.
            self.pinv_rcond = 1e-2
            # Adjacent spiral segments rest against each other in every
            # configuration (~30 contacts even with the arm hanging in free
            # space), and the resulting constraint force is on average larger
            # than qfrc_bias.  Carrying it in h makes M qdd + h = B u exact
            # (residual 1e-12 vs 1e-1 without it).
            self.include_constraint_forces = True
            # The XML tapers stiffness along the spiral and sets damping per joint;
            # keep those instead of overwriting them with scalars.
            self.override_passives = False
            # Lift the base so the arm hangs in free space.  ID-CLF-QP has no
            # contact term in h, so resting on the floor would make it fight
            # unmodelled constraint forces.
            self.base_body = 'segment_1__configuration_default'
            self.base_height = SPIROB_HORZ_BASE_HEIGHT
            # Let gravity settle the arm before the controller engages, so the
            # run does not start at B's singular point.  Specified in seconds so
            # it survives a timestep change upstream (the model moved from
            # 0.002 to 0.001 when it became a submodule).
            self.settle_time = 2.0
            # Dynamic B matrix - computed at runtime (via update_input_matrix after mj_fwdPosition)
            self.update_input_matrix()
            self.B_applied = np.eye(self.nu)
            self.sel = np.ones((self.nu, 1))
            # Control gains
            if self.control_scheme == 'impedance_QP':
                self.Kp, self.Kd = 2000.0, 2 * np.sqrt(2000.0)
            else:
                self.Kp, self.Kd = 200.0, 2 * np.sqrt(200.0)
            self.damping, self.stiffness = 0.05, 0.01  # unused: override_passives is False
            self.e = 0.01
            # Passive force sign (spirob uses -data.qfrc_passive)
            self.passive_sign = -1
            # Regularization coefficients for optimization
            self.reg_qdd = 0.2
            self.reg_u = 0.5
            self.reg_null = 0.1
            self.reg_dl = 1000
            # MPC-specific coefficients
            self.mpc_task_weight = 1.0
            self.mpc_null_weight = 0.0
            self.mpc_terminal_weight = 10.0
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
        # Cache constant matrix products used every control step
        self.PeG = self.Pe @ self.G
        self.FTPe_PeF = self.F.T @ self.Pe + self.Pe @ self.F
        
    def _probe_actuator_constants(self):
        """Measure the dcmotor force law from the compiled model.

        A `dcmotor` transmission produces an affine, velocity-dependent force

            actuator_force = k_v * ctrl - k_e * actuator_velocity

        where k_v = tau_nom/V_nom and k_e = k_v * (V_nom/omega_nom) follow from
        the actuator's `nominal` attribute.  Probing rather than hardcoding means
        retuning the XML does not silently invalidate the controller model.
        Saturation (|actuator_force| <= forcerange) is not modelled; it only binds
        at tendon speeds above ~0.4 m/s.
        """
        qpos, qvel, ctrl = (self.data.qpos.copy(), self.data.qvel.copy(),
                            self.data.ctrl.copy())
        # k_v: unit ctrl at rest.
        self.data.qpos[:] = 0.0
        self.data.qvel[:] = 0.0
        self.data.ctrl[:] = -1.0
        mujoco.mj_forward(self.model, self.data)
        self.k_v = float(-np.mean(self.data.actuator_force))
        # k_e: nonzero velocity with no ctrl.
        self.data.qvel[:] = 0.1
        self.data.ctrl[:] = 0.0
        mujoco.mj_forward(self.model, self.data)
        moving = np.abs(self.data.actuator_velocity) > 1e-9
        self.k_e = float(np.mean(-self.data.actuator_force[moving]
                                 / self.data.actuator_velocity[moving]))
        # Restore whatever state the caller had.
        self.data.qpos[:] = qpos
        self.data.qvel[:] = qvel
        self.data.ctrl[:] = ctrl
        mujoco.mj_forward(self.model, self.data)

    def get_passive_forces(self):
        """Get passive forces with correct sign for each robot.

        For dcmotor-driven robots this also carries the back-EMF term.  MuJoCo's
        dynamics are

            M qdd + qfrc_bias - qfrc_passive = moment.T @ (k_v*u - k_e*v_act)

        so the controller's `M qdd + h = B u` form needs the velocity-dependent
        part of the actuator force folded into h (B carries the k_v scaling).
        The term is not small: at qdot = 0.2 with ctrl = -6 it flips the sign of
        a tendon's force.
        """
        passive = self.passive_sign * self.data.qfrc_passive
        if self.k_e:
            moment = self.data.actuator_moment.reshape(self.nu, self.model.nv)
            passive = passive + moment.T @ (self.k_e * self.data.actuator_velocity)
        if self.include_constraint_forces:
            passive = passive - self.data.qfrc_constraint
        return passive

    def get_control_constraints(self, u_var):
        """Get control constraints for optimization (robot-agnostic)"""
        constraints = []
        constraints.append(self.lower_bounds <= u_var)
        constraints.append(u_var <= self.upper_bounds)
        return constraints
    
    def update_input_matrix(self):
        """Update input matrix - static for tendon/helix, dynamic for spirob"""
        if self.model_name in self.TENDON_ROBOTS:
            # actuator_moment is flat (nu*nv,); reshape to (nu, nv) and multiply by gear.
            # mj_step keeps actuator_moment current during the sim loop.
            # On the very first call (init), data is fresh so we run mj_forward once.
            if not np.any(self.data.actuator_moment):
                mujoco.mj_forward(self.model, self.data)
            # actuator_moment is flat (nu*nv,) and already includes the gear ratio.
            # Do not multiply by gear again.  k_v converts ctrl to actuator force
            # (1.0 for plain `motor` transmissions, tau_nom/V_nom for a dcmotor).
            self.B = self.k_v * self.data.actuator_moment.reshape(self.nu, self.model.nv).T
            if self.pinv_rcond is None:
                self.pinv_B = np.linalg.pinv(self.B)
            else:
                self.pinv_B = np.linalg.pinv(self.B, rcond=self.pinv_rcond)
            self.T, _ = self.complete_basis(self.B.T)
            self.Tinv = np.linalg.inv(self.T)
            self.TinvT = self.Tinv.T
            self.B_applied = self.B
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
        if self.override_passives:
            # spirob_horz opts out: its XML tapers stiffness per joint along the
            # spiral, and a blanket assignment would erase that.
            self.model.jnt_stiffness[:] = self.stiffness
            self.model.dof_damping[:] = self.damping
        if self.base_height is not None:
            # Done here rather than in the XML: spirob_horz.xml is generated by
            # onshape-to-robot and would lose the edit on regeneration.
            self.model.body_pos[self.model.body(self.base_body).id][2] = self.base_height
        if self.model_name == 'helix':
            self.data.qpos[2] = 0.0
            self.model.jnt_range[range(2,len(self.data.qpos),3)] = [[-0.001, 0.03/2] for i in range(2,len(self.data.qpos),3)]
            self.model.jnt_stiffness[range(2,len(self.data.qpos),3)] = 50
        elif self.model_name == 'spirob':
            # For the SpiRob, this should be a straight configuration
            self.data.qpos[:] = 0.0
        elif self.model_name == 'spirob_horz':
            # qpos = 0 is exactly the worst-conditioned point for B (cond ~950),
            # so start from the gravity-settled configuration instead: it is the
            # state the robot would actually be in at rest, and B is far better
            # conditioned there.
            self.data.qpos[:] = 0.0
            self.data.qvel[:] = 0.0
            self.data.ctrl[:] = 0.0
            for _ in range(int(self.settle_time / self.model.opt.timestep)):
                mujoco.mj_step(self.model, self.data)
            self.data.qvel[:] = 0.0
            self.data.time = 0.0
            mujoco.mj_forward(self.model, self.data)

    def compute_jacobian_derivative(self, site_id, h=1e-6, J_precomputed=None):
        """
        Compute the time derivative of the Jacobian for this robot
        
        Parameters:
        - site_id: ID of the site for Jacobian computation
        - h: Small positive step for numerical differentiation
        - J_precomputed: optional (task_dim, nv) Jacobian already computed this step.
          If provided, skips the initial kinematics update and Jacobian evaluation.
        
        Returns:
        - Jdot: The time derivative of the Jacobian
        """
        # Step 1 & 2: Get the Jacobian at the current state.
        # If the caller already computed it this step, reuse it to skip redundant
        # mj_kinematics + mj_comPos + mj_jacSite calls.
        if J_precomputed is not None:
            J = np.zeros((6, self.model.nv))
            J[:self.task_dim] = J_precomputed
        else:
            mujoco.mj_kinematics(self.model, self.data)
            mujoco.mj_comPos(self.model, self.data)
            J = np.zeros((6, self.model.nv))
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
        mujoco.mj_fullM(self.model, self.data, M)
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
        
    def get_jacobian_derivative(self, site_name="ee", h=1e-6, J_precomputed=None):
        """Compute and return Jacobian derivative on-demand.

        Parameters
        ----------
        J_precomputed : ndarray of shape (task_dim, nv), optional
            If provided, skips the initial kinematics update and Jacobian computation
            inside compute_jacobian_derivative, saving ~2 mj_kinematics + 1 mj_jacSite
            calls. Pass the result of get_jacobian() when it was just computed.
        """
        site_id = self.model.site(site_name).id
        dJ_dt = self.compute_jacobian_derivative(site_id, h, J_precomputed=J_precomputed)
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
        if self.model_name in self.TENDON_ROBOTS:
            self.data.ctrl[:] = np.clip(u, self.lower_bounds, self.upper_bounds)
        else:
            self.data.ctrl[:] = self.B_applied @ np.clip(u, self.lower_bounds, self.upper_bounds)