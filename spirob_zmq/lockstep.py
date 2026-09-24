"""Run the nodes deterministically in simulated time."""

import argparse
import heapq
import importlib
import json
import sys
import time

from spirob_zmq.core import _to_jsonable
from spirob_zmq.launch import LAUNCHES, VIEWER_NODES, _split_params
from spirob_zmq.core import parse_params


class Lockstep:
    def __init__(self):
        self.time = 0.0
        self.nodes = []
        self._queue = []
        self._seq = 0
        self.records = []
        self.record_topics = None

    def add(self, node):
        node._publish = lambda topic, msg, _n=node: self._deliver(topic, msg)
        node.now = lambda: self.time
        self.nodes.append(node)
        for timer in node._timers:
            self._push(timer.period, timer)
        return node

    def _push(self, due, timer):
        heapq.heappush(self._queue, (due, self._seq, timer))
        self._seq += 1

    def _deliver(self, topic, msg):
        msg = json.loads(json.dumps(msg, default=_to_jsonable))
        if self.record_topics is None or topic in self.record_topics:
            self.records.append({'t': self.time, 'topic': topic, 'msg': msg})
        for node in self.nodes:
            for callback, _ in node._subs.get(topic, []):
                callback(msg)

    def run(self, duration):
        while self._queue and self._queue[0][0] <= duration + 1e-12:
            due, _, timer = heapq.heappop(self._queue)
            self.time = due
            timer.callback()
            self._push(due + timer.period, timer)


def build(mode, params, per_node, skip=()):
    sim = Lockstep()
    for name in LAUNCHES[mode]:
        if name in VIEWER_NODES or name in skip:
            continue
        module = importlib.import_module(f'spirob_zmq.nodes.{name}')
        cls = next(v for k, v in vars(module).items()
                   if isinstance(v, type) and k.endswith('Node') and v.__module__ == module.__name__)
        node_params = {'jacobian_thread': False}
        node_params.update(params)
        node_params.update(per_node.get(name, {}))
        sim.add(cls(node_params))
    return sim


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--mode', choices=sorted(LAUNCHES), default='ekf')
    parser.add_argument('--duration', type=float, default=5.0, help='simulated seconds')
    parser.add_argument('--out', default='lockstep.jsonl')
    parser.add_argument('-p', '--param', action='append', default=[], metavar='[NODE.]KEY=VALUE')
    args = parser.parse_args(argv)

    shared, per_node_raw = _split_params(args.param)
    params = parse_params(sum((['-p', s] for s in shared), []))
    per_node = {n: parse_params(sum((['-p', s] for s in items), [])) for n, items in per_node_raw.items()}

    sim = build(args.mode, params, per_node)
    t0 = time.perf_counter()
    sim.run(args.duration)
    wall = time.perf_counter() - t0
    with open(args.out, 'w') as f:
        for rec in sim.records:
            f.write(json.dumps(rec) + '\n')
    for node in sim.nodes:
        node.destroy_node()
    print(f'simulated {args.duration:.2f} s in {wall:.1f} s wall; wrote {len(sim.records)} messages to {args.out}',
          file=sys.stderr)


if __name__ == '__main__':
    main()
