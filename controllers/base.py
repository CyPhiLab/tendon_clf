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


class BaseController(ABC):
    """Abstract base class for all controllers."""
    
    @abstractmethod
    def __call__(self, robot, model, data, target_vel, target_acc, twist, jac, M, dJ_dt, previous_solution=None) -> ControllerResult:
        """Execute controller and return standardized result."""
        pass
