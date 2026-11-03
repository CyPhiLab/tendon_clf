"""Viewer startup configuration without opening a GUI."""

import sys
from contextlib import nullcontext
from pathlib import Path
from types import SimpleNamespace

import mujoco
import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from spirob_zmq.core import Logger, Node  # noqa: E402
from spirob_zmq.nodes import simulation_node  # noqa: E402
from spirob_zmq.robots import DEFAULT_ROBOT, ROBOTS  # noqa: E402


@pytest.mark.parametrize('overrides, expected', [
    ({}, (0.0, -20.0)),
    ({'viewer_azimuth': 45.0, 'viewer_elevation': -30.0,
      'site_name': 'marker_1_seg_4'}, (45.0, -30.0)),
])
def test_camera_startup_and_manual_rotation(monkeypatch, overrides, expected):
    def init_node(self, name, params=None):
        self._params = params or {}
        self._profile = ROBOTS[DEFAULT_ROBOT]
        self._logger = Logger(name)

    viewer = SimpleNamespace(
        cam=mujoco.MjvCamera(), opt=mujoco.MjvOption(), lock=nullcontext,
        is_running=lambda: True, sync=lambda: None,
    )
    monkeypatch.setattr(Node, '__init__', init_node)
    monkeypatch.setattr(Node, 'create_subscription', lambda *args, **kwargs: None)
    monkeypatch.setattr(Node, 'create_timer', lambda *args: None)
    monkeypatch.setattr(simulation_node.mujoco.viewer, 'launch_passive', lambda *args: viewer)
    node = simulation_node.SimulationNode(overrides)
    site_name = overrides.get('site_name', ROBOTS[DEFAULT_ROBOT]['site_name'])
    tracked_site = node.model.site(site_name)
    assert tracked_site.group[0] == 0
    np.testing.assert_array_equal(tracked_site.rgba, [0.0, 1.0, 0.0, 1.0])
    assert viewer.opt.sitegroup[0] == 1
    assert viewer.opt.sitegroup[3] == 0
    assert (viewer.cam.azimuth, viewer.cam.elevation) == expected
    azimuth = np.deg2rad(expected[0])
    np.testing.assert_allclose(
        viewer.cam.lookat, 0.08 * np.array([-np.sin(azimuth), np.cos(azimuth), 0.0]))
    lookat = viewer.cam.lookat.copy()

    viewer.cam.azimuth = 70.0
    viewer.cam.elevation = -10.0
    node._update_viewer()
    assert (viewer.cam.azimuth, viewer.cam.elevation) == (70.0, -10.0)
    np.testing.assert_array_equal(viewer.cam.lookat, lookat)
