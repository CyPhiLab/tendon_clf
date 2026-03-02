"""Base controller interface for robot-agnostic control implementations."""

import numpy as np
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Optional


@dataclass
class ControllerResult:
    """Standardized return type for all controller functions"""
    task_error: float
    control_input: np.ndarray
    lyapunov_value: Optional[float] = None
    previous_solution: Optional[dict] = None
    t_ctrl: Optional[float] = None


class BaseController(ABC):
    """Abstract base class for all controllers."""
    
    @abstractmethod
    def __call__(self, robot, model, data, target_vel, target_acc, twist, jac, M, dJ_dt, previous_solution=None) -> ControllerResult:
        """Execute controller and return standardized result."""
        pass

    def cbf(self, robot, c, J, dJ, dq, ddq, alpha=100.0):
        """Optional callback function for additional processing."""
        c_dot = dJ @ dq + J @ ddq
        cbf = c_dot + 2 * alpha * J @ dq + alpha**2 * c
        return cbf
        
    def generate_workspace_cbf_constraints(self, robot, qdd_var, workspace_bounds=None, alpha=100.0):
        """Generate workspace CBF constraints for QP optimization
        
        Args:
            robot: Robot instance
            qdd_var: Joint acceleration variable from QP
            workspace_bounds: Workspace limits dict (uses robot default if None)
            alpha: CBF parameter
            
        Returns:
            list of constraint expressions for QP
        """
        if workspace_bounds is None:
            workspace_bounds = getattr(robot, 'workspace_bounds', None)
            if workspace_bounds is None:
                return []  # No workspace constraints defined
                
        cbf_data = robot.get_workspace_cbf_data(workspace_bounds, alpha)
        constraints = []
        dq = robot.get_joint_velocities()
        
        for data in cbf_data:
            c, J, dJ = data['c'], data['J'], data['dJ']
            # CBF constraint: c_dot + 2*alpha*J*dq + alpha^2*c >= 0
            c_dot = dJ @ dq + J @ qdd_var
            cbf_constraint = c_dot + 2 * alpha * J @ dq + alpha**2 * c >= 0
            constraints.append(cbf_constraint)
            print(f"    → Added CBF constraint: c={c:.3f}")
        return constraints
