#!/usr/bin/env python3
"""
Comprehensive experiment runner for all robot/controller/experiment combinations.
Replaces the hardcoded run_all.sh with systematic coverage.
"""

import argparse
import time
from itertools import product
from utils import *


def run_single_experiment(robot, controller, experiment, target_pos=None, sim_duration=10.0):
    """Run a single experiment configuration."""
    
    print(f"Running: {robot} + {controller} + {experiment}" + (f" + {target_pos}" if target_pos else ""))

    start_time = time.time()
    
    # Run simulation
    results = simulate_model(
        headless=True,  # Always headless for batch runs
        control_scheme=controller,
        target_pos=target_pos,
        controller=None,
        experiment=experiment,
        model_name=robot,
        sim_duration=sim_duration
    )
    
    # Add timing data
    end_time = time.time()
    results['wall_time'] = end_time - start_time
    results['sim_time_total'] = results['sim_time'][-1]

    csv_path = save_results(
        results=results,
        experiment=experiment,
        control_scheme=controller,
        model_name=robot,
        target_pos=target_pos
    )

    print(f"  → Completed in {end_time - start_time:.2f}s, saved to {csv_path}")
    return results


def main():
    parser = argparse.ArgumentParser(description='Run comprehensive experiments across all configurations')
    parser.add_argument('--robots', nargs='+', default=['helix', 'tendon', 'spirob'], 
                       choices=['helix', 'tendon', 'spirob'],
                       help='Robots to test (default: all)')
    parser.add_argument('--controllers', nargs='+', default=['id_clf_qp', 'impedance', 'osc', 'impedance_QP', 'clf_qp','uosc'],
                       choices=['id_clf_qp', 'impedance', 'osc', 'impedance_QP', 'clf_qp','uosc'],
                       help='Controllers to test (default: all)')
    parser.add_argument('--experiments', nargs='+', default=['set', 'tracking'],
                        choices=['set', 'tracking'],
                        help='Experiments to run (default: both)')
    parser.add_argument('--target_positions', nargs='+', default=['pos1', 'pos2', 'pos3', 'pos4'],
                       choices=['pos1', 'pos2', 'pos3', 'pos4'],
                       help='Target positions for set experiments (default: all)')
    parser.add_argument('--sim_duration', type=float, default=10.0, help='Duration of simulations in seconds')
    args = parser.parse_args()

    total_start_time = time.time()
    completed_experiments = 0
    total_experiments = 0

    valid_configs = []

    for robot, controller, experiment in product(args.robots, args.controllers, args.experiments):

        # ---- Filtering Rules ----
        if robot == 'helix' and controller == 'clf_qp':
            continue

        if robot == 'spirob' and experiment == 'set' and controller in ['impedance', 'osc', 'uosc']:
            continue

        if robot == 'spirob' and experiment == 'tracking':
            continue

        valid_configs.append((robot, controller, experiment))

        if experiment == 'set':
            total_experiments += len(args.target_positions)
        else:
            total_experiments += 1

    print(f"Starting comprehensive experiment suite: {total_experiments} total experiments")
    print(f"Robots: {args.robots}")
    print(f"Controllers: {args.controllers}")
    print(f"Experiments: {args.experiments}")
    print(f"Simulation duration: {args.sim_duration}s")
    if 'set' in args.experiments:
        print(f"Target positions: {args.target_positions}")
    print("=" * 80)
    
    # Run all experiments
    for robot, controller, experiment in valid_configs:

        try:
            if experiment == 'set':
                for target_pos in args.target_positions:
                    run_single_experiment(robot, controller, experiment, target_pos, args.sim_duration)
                    completed_experiments += 1
                    print(f"Progress: {completed_experiments}/{total_experiments} ({100*completed_experiments/total_experiments:.1f}%)")
            else:  # tracking
                # Single tracking experiment
                run_single_experiment(robot, controller, experiment, sim_duration=args.sim_duration)
                completed_experiments += 1
                print(f"Progress: {completed_experiments}/{total_experiments} ({100*completed_experiments/total_experiments:.1f}%)")

        except Exception as e:
            print(f"ERROR in {robot} + {controller} + {experiment}: {e}")
            completed_experiments += 1

    total_time = time.time() - total_start_time
    print("=" * 80)
    print(f"Experiment suite completed!")
    print(f"Total time: {total_time:.2f} seconds ({total_time/60:.1f} minutes)")
    print(f"Average time per experiment: {total_time/total_experiments:.2f} seconds")
    print(f"Completed: {completed_experiments}/{total_experiments}")


if __name__ == "__main__":
    main()


