"""Tracked-position publishing and export without motors or a broker."""

import csv
import json
import sys
from pathlib import Path
from types import SimpleNamespace

import mujoco
import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from spirob_zmq.common import ROBOT_STATE, SITE_MEASUREMENT, TRACKING_STATE  # noqa: E402
from spirob_zmq.core import Logger, Node, encode  # noqa: E402
from spirob_zmq.ekf_report import export_tracking_csv, plot_tracking  # noqa: E402
from spirob_zmq import ekf_report  # noqa: E402
from spirob_zmq.nodes.control_node import ControlNode  # noqa: E402
from spirob_zmq.robots import DEFAULT_ROBOT, ROBOTS  # noqa: E402


@pytest.fixture
def controller(monkeypatch):
    def init(self, name, params=None):
        self._params = params or {}
        self._profile = ROBOTS[DEFAULT_ROBOT]
        self._logger = Logger(name)

    messages = []
    subscriptions = {}
    monkeypatch.setattr(Node, '__init__', init)
    monkeypatch.setattr(Node, 'create_publisher', lambda self, topic: SimpleNamespace(
        publish=lambda msg: messages.append((topic, json.loads(encode(topic, msg)[1])))))
    monkeypatch.setattr(Node, 'create_subscription',
                        lambda self, topic, cb, **kw: subscriptions.update({topic: cb}))
    monkeypatch.setattr(Node, 'create_timer', lambda *args: None)
    node = ControlNode({'site_name': 'marker_2_seg_11', 'target_pos': [0.1, 0.2, 0.3]})
    assert ROBOT_STATE in subscriptions and SITE_MEASUREMENT in subscriptions
    return node, messages


def test_positions_and_dropouts_are_recorded(controller):
    node, messages = controller
    q = node.model.key_qpos[0].copy()
    node._on_state({'stamp': 42., 'q': q, 'dq': np.zeros(node.model.nv), 'is_valid': False})
    assert not messages
    node._on_state({'stamp': 42., 'q': q, 'dq': np.zeros(node.model.nv), 'is_valid': True})
    d = mujoco.MjData(node.model)
    d.qpos[:] = q
    mujoco.mj_kinematics(node.model, d)
    estimated = messages[-1][1]
    np.testing.assert_allclose(estimated['position_m'], d.site_xpos[node.site_id])
    assert estimated['source'] == 'estimated'
    assert estimated['stamp'] == 42.
    # Incoming metadata, not the current profile order, selects the measured marker.
    node._on_measurement({
        'stamp': 42., 'source': 'vicon',
        'site_names': ['marker_2_seg_11', 'marker_1_seg_2'],
        'data': [0.12, 0.23, 0.34, 9., 9., 9.],
    })
    measured = messages[-1][1]
    np.testing.assert_allclose(measured['position_m'], [0.12, 0.23, 0.34])
    np.testing.assert_allclose(measured['error_m'], [0.02, 0.03, 0.04])
    assert measured['target_m'] == [0.1, 0.2, 0.3]
    assert all(topic == TRACKING_STATE for topic, _ in messages)
    node._on_measurement({
        'stamp': 43., 'site_names': ['marker_2_seg_11'], 'source': 'vicon',
        'data': [float('nan')] * 3,
    })
    assert messages[-1][1]['is_valid'] is False
    assert messages[-1][1]['position_m'] == [None] * 3


def test_missing_or_malformed_measurements_log_errors(controller, capsys):
    node, messages = controller
    node._on_measurement({'stamp': 1., 'site_names': [], 'data': []})
    node._on_measurement({'stamp': 1., 'site_names': [node.site_name], 'data': [1.]})
    assert not messages
    output = capsys.readouterr().err
    assert 'absent from marker measurements' in output
    assert 'Invalid marker measurement size' in output


def test_csv_and_plot_use_recorded_metadata(controller, tmp_path, monkeypatch, capsys):
    node, messages = controller
    for source, stamp, point in [
        ('estimated', 12., np.array([0.11, 0.2, 0.3])),
        ('vicon', 12., np.array([0.1, 0.22, 0.3])),
        ('vicon', 13., np.full(3, np.nan)),
    ]:
        node._publish_tracking(stamp, point, source)
    path = tmp_path / 'run.jsonl'
    path.write_text(''.join(json.dumps({'t': 100. + i, 'topic': t, 'msg': m}) + '\n'
                            for i, (t, m) in enumerate(messages)))
    out = export_tracking_csv(path)
    assert out == tmp_path / 'run_tracking.csv'
    with out.open() as f:
        rows = list(csv.DictReader(f))
    assert len(rows) == 3
    assert rows[0]['site_name'] == node.site_name
    assert float(rows[0]['x_m']) == 0.11
    assert float(rows[1]['error_y_m']) == pytest.approx(0.02)
    assert rows[2]['x_m'] == ''
    assert rows[2]['is_valid'] == 'False'
    assert float(rows[2]['elapsed_s']) == 1.
    # Plot must not use a subsequently changed target/site.
    monkeypatch.setitem(ROBOTS[DEFAULT_ROBOT], 'target_pos', [10., 20., 30.])
    monkeypatch.setitem(ROBOTS[DEFAULT_ROBOT], 'site_name', 'ee')
    plot_tracking(path, tmp_path / 'tracking.png')
    output = capsys.readouterr().out
    assert '10.0 mm' in output and '20.0 mm' in output
    assert (tmp_path / 'tracking.png').stat().st_size > 0
    out.unlink()
    monkeypatch.setattr(ekf_report, 'plot_hardware', lambda *args: None)
    monkeypatch.setattr(ekf_report, 'extra_plots', lambda *args: None)
    ekf_report.main([str(path), '--plot', str(tmp_path / 'run.png')])
    assert out.exists(), 'launcher report invocation must also export CSV'


def test_old_recording_does_not_invent_tracking_data(tmp_path):
    path = tmp_path / 'old.jsonl'
    path.write_text('')
    assert export_tracking_csv(path) is None
    assert not (tmp_path / 'old_tracking.csv').exists()
