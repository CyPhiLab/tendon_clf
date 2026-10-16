"""RealMeasurementNode end to end against a fake Vicon SDK replaying synthetic unlabelled frames.

    python tests/test_real_measurement_node.py        (or pytest tests/)
"""

import sys
import tempfile
import types
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import test_startup as synth  # noqa: E402

FRAMES = []


def _install_fake_sdk():
    class DataStreamException(Exception):
        pass

    class Client:
        class StreamMode:
            EClientPull = 0

        def __init__(self):
            self.i, self.rng = -1, np.random.default_rng(0)

        def Connect(self, ip): pass
        def IsConnected(self): return True
        def EnableMarkerData(self): pass
        def EnableUnlabeledMarkerData(self): pass
        def SetStreamMode(self, mode): pass
        def GetFrameNumber(self): return 1000 + self.i

        def GetFrame(self):
            if self.i + 1 >= len(FRAMES):
                return False
            self.i += 1
            return True

        def GetUnlabeledMarkers(self):
            pts = FRAMES[self.i][self.rng.permutation(len(FRAMES[self.i]))]
            return [((p[0], p[1], p[2]), k) for k, p in enumerate(pts)]

    sdk = types.ModuleType('vicon_dssdk')
    sdk.ViconDataStream = types.SimpleNamespace(Client=Client, DataStreamException=DataStreamException)
    sys.modules['vicon_dssdk'] = sdk


def test_node_starts_from_unlabelled_frames_and_tracks():
    from spirob_zmq.calibrate_frame import calibrate
    from spirob_zmq.pose_init import save_frame, stationary_clusters, transform

    _install_fake_sdk()
    rng = np.random.default_rng(7)
    model, data = synth._model()
    rest = synth._sites(model, data, synth.NAMES + synth.EXTRA)
    curl = synth._curl(model, data, current=(-0.7, 0.0, 0.0), duration=0.6)
    truth = np.vstack([np.repeat(rest[None], 150, axis=0), curl])           # model frame

    R_true = np.array([[np.cos(0.1), -np.sin(0.1), 0], [np.sin(0.1), np.cos(0.1), 0], [0, 0, 1]])
    t_true = np.array([0.03, -0.02, 0.01])
    FRAMES.clear()
    for frame in truth:
        vicon = (frame - t_true) @ R_true + rng.normal(0, 2e-4, frame.shape)
        keep = rng.random(len(vicon)) > 0.05
        FRAMES.append(1e3 * vicon[keep])

    with tempfile.TemporaryDirectory() as tmp:
        centres, still = stationary_clusters([f / 1e3 for f in FRAMES[:100]])
        assert still
        _, (R, t, res, _) = calibrate(centres)
        path = f'{tmp}/vicon_frame.json'
        save_frame(path, R, t, res, 'test')

        from spirob_zmq.nodes.real_measurement_node import RealMeasurementNode
        published = []
        node = RealMeasurementNode({'vicon_frame_path': path, 'warmup_frames': 100})
        node._publish = lambda topic, msg: published.append((topic, msg))
        node._publish_initial()
        while node.vicon.client.i + 1 < len(FRAMES):
            node._tick()
        node.destroy_node()

    init = [m for topic, m in published if topic == '/spirob/initial_state']
    meas = np.array([np.reshape(m['data'], (-1, 3)) for topic, m in published if topic == '/spirob/site_measurement'])
    assert init and init[0]['rms'] < 2e-3
    expected = truth[-len(meas):, :9]
    ok = ~np.isnan(meas[..., 0])
    err = np.linalg.norm(meas - expected, axis=2)
    assert np.sum(err[ok] > 3e-3) == 0, 'mislabelled markers'
    assert np.mean(~ok) < 0.15, f'{100 * np.mean(~ok):.1f} % NaN'


if __name__ == '__main__':
    test_node_starts_from_unlabelled_frames_and_tracks()
    print('PASS test_node_starts_from_unlabelled_frames_and_tracks')
