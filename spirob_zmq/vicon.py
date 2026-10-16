"""Thin wrapper around the Vicon DataStream SDK: unlabelled markers in meters, in the Vicon frame."""

import time

import numpy as np


class ViconClient:
    def __init__(self, ip, logger, rate_hz=100.0):
        from vicon_dssdk import ViconDataStream  # imported here so offline tools don't need the SDK
        self._vds = ViconDataStream
        self.rate_hz = rate_hz
        self.client = ViconDataStream.Client()
        logger.info(f'Connecting to Vicon at {ip}...')
        while not self.client.IsConnected():
            try:
                self.client.Connect(ip)
            except ViconDataStream.DataStreamException:
                logger.warn('Vicon connect failed; retrying...', throttle_duration_sec=1.0)
                time.sleep(1)
        logger.info('Connected to Vicon.')
        self.client.EnableMarkerData()
        self.client.EnableUnlabeledMarkerData()
        self.client.SetStreamMode(ViconDataStream.Client.StreamMode.EClientPull)
        self._last_frame = None

    def read(self):
        """(stamp, points (M, 3) in meters) for a new frame, or None if there is none yet.

        The stamp comes from the Vicon frame number when available, so frame-to-frame spacing does
        not pick up host scheduling jitter.
        """
        try:
            if not self.client.GetFrame():
                return None
            frame = self.client.GetFrameNumber()
            if frame == self._last_frame:
                return None
            self._last_frame = frame
            markers = self.client.GetUnlabeledMarkers()
        except self._vds.DataStreamException:
            return None
        stamp = frame / self.rate_hz if frame is not None else time.time()
        pts = np.array([[m[0][0], m[0][1], m[0][2]] for m in markers], dtype=float).reshape(-1, 3) / 1000.0
        return stamp, pts

    def collect(self, n_frames):
        """The next n_frames frames as (stamps, list of point arrays)."""
        stamps, frames = [], []
        while len(frames) < n_frames:
            got = self.read()
            if got is None:
                time.sleep(0.001)
                continue
            stamps.append(got[0])
            frames.append(got[1])
        return stamps, frames
