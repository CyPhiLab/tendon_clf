"""The SpiRob MuJoCo scene used to render logged hardware states (needs MuJoCo >= 3.14)."""

from pathlib import Path

import mujoco

SCENE_XML = Path(__file__).resolve().parents[2] / 'mujoco_models' / 'spirob' / 'spirob_horz_control.xml'
ROBOT_NQ = 48   # x_i, y_i for the 24 segments; the logged EKF q covers exactly these


def load():
    return mujoco.MjModel.from_xml_path(str(SCENE_XML))
