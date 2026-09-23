"""Minimal rclpy-like node runtime on top of ZeroMQ.

Topology
--------
All nodes talk through a single XSUB/XPUB forwarder (``spirob_zmq.broker``):

    node PUB --connect--> [XSUB  broker  XPUB] <--connect-- node SUB

so any node can publish or subscribe to any topic without knowing who else is
running (same as ROS topics). Addresses default to localhost TCP, which works
on every OS (``ipc://`` does not exist on Windows). Override them with the
``SPIROB_ZMQ_PUB`` / ``SPIROB_ZMQ_SUB`` environment variables to run nodes on
different machines.

Messages are sent as two frames: ``[topic, json_payload]``. Payloads are plain
dicts using the same field names as the ROS ``spirob_interfaces`` messages.
"""

import argparse
import json
import os
import sys
import time

import zmq

from spirob_zmq.robots import DEFAULT_ROBOT, REPO_ROOT, ROBOTS

# Nodes publish to PUB_ADDR (broker XSUB side), subscribe from SUB_ADDR (broker XPUB side).
PUB_ADDR = os.environ.get('SPIROB_ZMQ_PUB', 'tcp://127.0.0.1:5555')
SUB_ADDR = os.environ.get('SPIROB_ZMQ_SUB', 'tcp://127.0.0.1:5556')


def _to_jsonable(obj):
    if hasattr(obj, 'tolist'):
        return obj.tolist()
    raise TypeError(f'Object of type {type(obj).__name__} is not JSON serializable')


def encode(topic, msg):
    return [topic.encode(), json.dumps(msg, default=_to_jsonable).encode()]


def decode(frames):
    return frames[0].decode(), json.loads(frames[1])


class Logger:
    def __init__(self, name):
        self.name = name
        self._last = {}

    def _log(self, level, text, throttle_duration_sec=None):
        if throttle_duration_sec is not None:
            now = time.monotonic()
            key = (level, text)
            if now - self._last.get(key, -1e9) < throttle_duration_sec:
                return
            self._last[key] = now
        print(f'[{level}] [{time.time():.3f}] [{self.name}]: {text}', file=sys.stderr, flush=True)

    def debug(self, text, **kw):
        if os.environ.get('SPIROB_DEBUG'):
            self._log('DEBUG', text, **kw)

    def info(self, text, **kw):
        self._log('INFO', text, **kw)

    def warn(self, text, **kw):
        self._log('WARN', text, **kw)

    warning = warn

    def error(self, text, **kw):
        self._log('ERROR', text, **kw)


class _Timer:
    def __init__(self, period, callback):
        self.period = period
        self.callback = callback
        self.next_t = time.monotonic() + period


class Node:
    """Single-threaded node: subscriptions and timers are serviced by ``spin()``.

    Mirrors the subset of the rclpy API the spirob nodes use, so the ported
    node code reads almost line-for-line like the ROS version.
    """

    def __init__(self, name, params=None):
        self.name = name
        self._params = dict(params or {})
        robot = self._params.get('robot', DEFAULT_ROBOT)
        if robot not in ROBOTS:
            raise ValueError(f'unknown robot {robot!r}; choose from {sorted(ROBOTS)}')
        self.robot = robot
        self._profile = ROBOTS[robot]
        self._logger = Logger(name)
        self._ctx = zmq.Context.instance()

        self._pub = self._ctx.socket(zmq.PUB)
        self._pub.setsockopt(zmq.LINGER, 0)
        self._pub.connect(PUB_ADDR)

        self._sub = self._ctx.socket(zmq.SUB)
        self._sub.setsockopt(zmq.LINGER, 0)
        self._sub.connect(SUB_ADDR)

        self._subs = {}      # topic -> list of (callback, latest_only)
        self._timers = []
        self._running = True

    # --- rclpy-like API -------------------------------------------------
    def get_logger(self):
        return self._logger

    def declare_parameter(self, name, default=None):
        """Return the ``-p`` override for ``name``, else the robot profile's
        value (see robots.py), else ``default``."""
        value = self._params.get(name, self._profile.get(name, default))
        if isinstance(default, float) and isinstance(value, int) and not isinstance(value, bool):
            value = float(value)
        return value

    def create_publisher(self, topic):
        return _Publisher(self, topic)

    def create_subscription(self, topic, callback, latest_only=False):
        """Subscribe to ``topic``.

        ``latest_only=True`` behaves like a ROS queue depth of 1: if several
        messages arrived since the last spin iteration only the newest is
        delivered. Otherwise every message is delivered in order.
        """
        if topic not in self._subs:
            self._sub.setsockopt(zmq.SUBSCRIBE, topic.encode())
            self._subs[topic] = []
        self._subs[topic].append((callback, latest_only))

    def create_timer(self, period, callback):
        timer = _Timer(period, callback)
        self._timers.append(timer)
        return timer

    def now(self):
        return time.time()

    def shutdown(self):
        self._running = False

    def ok(self):
        return self._running

    def destroy_node(self):
        self._running = False
        self._pub.close()
        self._sub.close()

    # --- event loop -------------------------------------------------------
    def _publish(self, topic, msg):
        self._pub.send_multipart(encode(topic, msg))

    def _drain(self):
        """Receive everything currently queued and dispatch it."""
        pending = []
        while True:
            try:
                frames = self._sub.recv_multipart(zmq.NOBLOCK)
            except zmq.Again:
                break
            topic, msg = decode(frames)
            if topic in self._subs:   # ZMQ filters by prefix; require exact match
                pending.append((topic, msg))
        if not pending:
            return
        latest = {topic: msg for topic, msg in pending}
        for topic, msg in pending:
            for callback, latest_only in self._subs[topic]:
                if not latest_only:
                    callback(msg)
        for topic, msg in latest.items():
            for callback, latest_only in self._subs[topic]:
                if latest_only:
                    callback(msg)

    def spin(self):
        poller = zmq.Poller()
        poller.register(self._sub, zmq.POLLIN)
        while self._running:
            now = time.monotonic()
            if self._timers:
                next_due = min(t.next_t for t in self._timers)
                timeout_ms = max(0.0, (next_due - now) * 1000.0)
            else:
                timeout_ms = 100.0
            if poller.poll(timeout_ms):
                self._drain()
            now = time.monotonic()
            for timer in self._timers:
                if not self._running:
                    break
                if now >= timer.next_t:
                    timer.callback()
                    timer.next_t += timer.period
                    # If we fell more than a period behind (slow callback), skip
                    # missed ticks instead of bursting to catch up (as rclpy does).
                    if timer.next_t < time.monotonic():
                        timer.next_t = time.monotonic() + timer.period


class _Publisher:
    def __init__(self, node, topic):
        self._node = node
        self.topic = topic

    def publish(self, msg):
        self._node._publish(self.topic, msg)


def parse_params(argv=None, description=None):
    """Parse ``--param key=value`` / ``--config file.json`` into a dict.

    Values are parsed as JSON when possible (``rate_hz=100``, ``motor_ids=[0,1,2]``,
    ``dry_run=false``) and kept as strings otherwise (``model_path=foo.xml``).
    """
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument('--config', help='JSON file with parameter overrides')
    parser.add_argument('-p', '--param', action='append', default=[], metavar='KEY=VALUE',
                        help='parameter override, may be repeated')
    args = parser.parse_args(argv)
    params = {}
    if args.config:
        with open(args.config) as f:
            params.update(json.load(f))
    for item in args.param:
        key, sep, raw = item.partition('=')
        if not sep:
            parser.error(f'--param expects KEY=VALUE, got {item!r}')
        try:
            params[key] = json.loads(raw)
        except json.JSONDecodeError:
            params[key] = raw
    return params


def run_node(node_cls, argv=None):
    """Standard ``main()`` body: build the node from CLI params and spin it."""
    params = parse_params(argv, description=node_cls.__doc__)
    node = node_cls(params)
    try:
        node.spin()
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
