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
