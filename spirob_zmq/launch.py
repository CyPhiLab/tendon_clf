"""Launch the broker and nodes. -p key=value for all nodes, -p node.key=value for one."""

import argparse
import json
import os
import shutil
import signal
import subprocess
import sys
import time
from pathlib import Path

from spirob_zmq.core import REPO_ROOT

LAUNCHES = {
    'ekf': ['virtual_measurement_node', 'state_estimation_node', 'control_node',
            'simulation_node', 'hardware_node'],
    'real_ekf': ['real_measurement_node', 'state_estimation_node', 'control_node',
                 'simulation_node', 'hardware_node'],
    'no_ekf': ['get_state_node', 'control_node', 'simulation_node'],
    'teleop': ['get_state_node', 'teleop_node', 'simulation_node'],
    'real_teleop': ['teleop_node', 'hardware_node'],
}
VIEWER_NODES = {'simulation_node'}

# Where each mode's recordings and plots go
RESULTS_DIR = REPO_ROOT / 'ekf_results'
RESULTS_SUBDIR = {'ekf': 'sim', 'no_ekf': 'sim', 'teleop': 'sim', 'real_ekf': 'real', 'real_teleop': 'real'}


def _viewer_python():
    if sys.platform != 'darwin':
        return sys.executable
    candidate = Path(sys.executable).with_name('mjpython')
    if candidate.exists():
        return str(candidate)
    found = shutil.which('mjpython')
    if found:
        return found
    sys.exit('macOS: the MuJoCo viewer needs `mjpython` (ships with `pip install mujoco`), '
             'but it was not found. Install mujoco in this environment or use --headless.')


def _split_params(items):
    shared, per_node = [], {}
    for item in items:
        key, sep, _ = item.partition('=')
        if not sep:
            sys.exit(f'-p expects KEY=VALUE, got {item!r}')
        node, dot, _ = key.partition('.')
        if dot and node.endswith('_node'):
            per_node.setdefault(node, []).append(item[len(node) + 1:])
        else:
            shared.append(item)
    return shared, per_node


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--mode', choices=sorted(LAUNCHES), default='ekf')
    parser.add_argument('--headless', action='store_true', help='do not start the MuJoCo viewer')
    parser.add_argument('--duration', type=float, default=None, help='stop after this many seconds')
    parser.add_argument('--no-broker', action='store_true', help='a broker is already running')
    parser.add_argument('-p', '--param', action='append', default=[], metavar='[NODE.]KEY=VALUE')
    parser.add_argument('--config', help='JSON file: {"key": v, "node_name": {"key": v}}')
    parser.add_argument('--no-record', action='store_true', help='do not record topics or plot')
    parser.add_argument('--name', help='run name (default: timestamp)')
    args = parser.parse_args(argv)

    shared, per_node = _split_params(args.param)
    config = {}
    if args.config:
        with open(args.config) as f:
            config = json.load(f)

    env = dict(os.environ)
    env['PYTHONPATH'] = os.pathsep.join(filter(None, [str(REPO_ROOT), env.get('PYTHONPATH')]))
    env.setdefault('PYTHONUNBUFFERED', '1')

    procs = []
    popen_kw = {'cwd': str(REPO_ROOT), 'env': env}
    if sys.platform == 'win32':
        popen_kw['creationflags'] = subprocess.CREATE_NEW_PROCESS_GROUP

    def spawn(name, cmd):
        print(f'[launch] starting {name}: {" ".join(cmd)}', flush=True)
        procs.append((name, subprocess.Popen(cmd, **popen_kw)))

    if not args.no_broker:
        spawn('broker', [sys.executable, '-m', 'spirob_zmq.broker'])
        time.sleep(0.2)

    record_path = None
    if not args.no_record:
        out_dir = RESULTS_DIR / RESULTS_SUBDIR[args.mode]
        out_dir.mkdir(parents=True, exist_ok=True)
        record_path = out_dir / f'{args.name or time.strftime("%Y%m%d_%H%M%S")}.jsonl'
        spawn('recorder', [sys.executable, '-m', 'spirob_zmq.topic', 'record', str(record_path)])
        time.sleep(0.2)  # subscribe before nodes start publishing

    for name in LAUNCHES[args.mode]:
        if name in VIEWER_NODES and args.headless:
            continue
        python = _viewer_python() if name in VIEWER_NODES else sys.executable
        cmd = [python, '-m', f'spirob_zmq.nodes.{name}']
        node_cfg = {k: v for k, v in config.items() if not isinstance(v, dict)}
        node_cfg.update(config.get(name, {}))
        if name == 'control_node' and not args.headless:
            cmd += ['-p', 'start_delay_s=2.0']
        for key, value in node_cfg.items():
            cmd += ['-p', f'{key}={json.dumps(value)}']
        for item in shared + per_node.get(name, []):
            cmd += ['-p', item]
        spawn(name, cmd)

    deadline = None if args.duration is None else time.monotonic() + args.duration
    exit_code = 0
    try:
        while True:
            for name, proc in procs:
                code = proc.poll()
                if code is not None:
                    print(f'[launch] {name} exited with code {code}; shutting down', flush=True)
                    exit_code = code
                    raise StopIteration
            if deadline is not None and time.monotonic() > deadline:
                print('[launch] duration reached; shutting down', flush=True)
                raise StopIteration
            time.sleep(0.1)
    except (KeyboardInterrupt, StopIteration):
        pass
    finally:
        for name, proc in reversed(procs):
            if proc.poll() is None:
                if sys.platform == 'win32':
                    proc.send_signal(signal.CTRL_BREAK_EVENT)
                else:
                    proc.send_signal(signal.SIGINT)
        for name, proc in reversed(procs):
            try:
                proc.wait(timeout=3)
            except subprocess.TimeoutExpired:
                proc.kill()

    if record_path is not None and record_path.exists() and record_path.stat().st_size > 0:
        plot_path = record_path.with_suffix('.png')
        print(f'[launch] recorded {record_path}; plotting {plot_path}', flush=True)
        try:
            subprocess.run([sys.executable, '-m', 'spirob_zmq.ekf_report', str(record_path),
                            '--plot', str(plot_path)], **popen_kw)
        except KeyboardInterrupt:
            pass
    return exit_code


if __name__ == '__main__':
    sys.exit(main())
