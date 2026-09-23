"""Inspect topics while the system runs (``ros2 topic echo/hz`` equivalent).

    python -m spirob_zmq.topic echo /spirob/robot_state
    python -m spirob_zmq.topic hz /spirob/motor_command
    python -m spirob_zmq.topic record out.jsonl              # all topics
    python -m spirob_zmq.topic record out.jsonl /spirob/robot_state /spirob/site_measurement

``record`` writes one JSON object per line: {"t": recv_time, "topic": ..., "msg": {...}}.
"""

import argparse
import json
import time

import zmq

from spirob_zmq.core import SUB_ADDR, decode


def _socket(topics):
    sock = zmq.Context.instance().socket(zmq.SUB)
    sock.connect(SUB_ADDR)
    for topic in topics or ['']:
        sock.setsockopt(zmq.SUBSCRIBE, topic.encode())
    return sock


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest='cmd', required=True)
    p_echo = sub.add_parser('echo')
    p_echo.add_argument('topic')
    p_hz = sub.add_parser('hz')
    p_hz.add_argument('topic')
    p_rec = sub.add_parser('record')
    p_rec.add_argument('output')
    p_rec.add_argument('topics', nargs='*')
    args = parser.parse_args(argv)

    topics = [args.topic] if args.cmd in ('echo', 'hz') else args.topics
    sock = _socket(topics)
    wanted = set(topics)
    try:
        if args.cmd == 'echo':
            while True:
                topic, msg = decode(sock.recv_multipart())
                if topic in wanted:
                    print(json.dumps(msg), flush=True)
        elif args.cmd == 'hz':
            count, t0 = 0, time.monotonic()
            while True:
                if sock.poll(1000):
                    topic, _ = decode(sock.recv_multipart())
                    count += topic in wanted
                now = time.monotonic()
                if now - t0 >= 1.0:
                    print(f'{args.topic}: {count / (now - t0):.1f} Hz', flush=True)
                    count, t0 = 0, now
        else:
            with open(args.output, 'w') as f:
                while True:
                    topic, msg = decode(sock.recv_multipart())
                    if not wanted or topic in wanted:
                        f.write(json.dumps({'t': time.time(), 'topic': topic, 'msg': msg}) + '\n')
    except KeyboardInterrupt:
        pass


if __name__ == '__main__':
    main()
