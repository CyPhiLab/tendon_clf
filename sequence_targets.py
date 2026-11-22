"""Shared ordered set-point sequences for MuJoCo experiments."""

import numpy as np


SEQUENCE_HOLD_SECONDS = 20.0

# Clockwise ordering for the four set points on each robot's test loop.
SEQUENCE_TARGETS = {
    "helix": np.array([
        [0.25, 0.05, 0.50],
        [0.30, 0.00, 0.55],
        [0.25, -0.05, 0.50],
        [0.20, 0.00, 0.45],
    ]),
    "spirob_horz": np.array([
        [0.00, 0.250, 0.125],
        [0.05, 0.225, 0.150],
        [0.00, 0.200, 0.175],
        [-0.05, 0.225, 0.150],
    ]),
    "tendon": np.array([
        [0.15, 0.00, 0.25],
        [0.20, 0.00, 0.15],
        [0.15, 0.00, 0.05],
        [0.10, 0.00, 0.15],
    ]),
}
