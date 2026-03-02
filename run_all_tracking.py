#!/usr/bin/env python3

import argparse
import time
from itertools import product
from utils import *


def run_single_experiment(robot, controller, omega=None):

    print(
        f"Running: {robot} + {controller} + tracking"
        + (f" + {omega}" if omega else "")
    )

    start_time = time.time()

    results = simulate_model(
        headless=True,
        control_scheme=controller,
        target_pos=None,
        controller=None,
        experiment="tracking",
        model_name=robot,
        omega=omega
    )

    end_time = time.time()

    results["wall_time"] = end_time - start_time
    results["sim_time_total"] = results["sim_time"][-1]

    csv_path = save_results(
        results=results,
        experiment="tracking",
        control_scheme=controller,
        model_name=robot,
        target_pos=None,
        omega=omega
    )

    print(f"  → Completed in {end_time - start_time:.2f}s, saved to {csv_path}")
    return results


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--robots",
        nargs="+",
        default=["helix", "tendon"],
        choices=["helix", "tendon", "spirob"],
    )

    parser.add_argument(
        "--controllers",
        nargs="+",
        default=["id_clf_qp", "impedance", "osc", "impedance_QP", "clf_qp", "uosc"],
        choices=["id_clf_qp", "impedance", "osc", "impedance_QP", "clf_qp", "uosc"],
    )

    args = parser.parse_args()

    omega_list = ["omg1", "omg2", "omg3", "omg4", "omg5"]

    total_start_time = time.time()
    completed_experiments = 0

    valid_configs = []

    for robot, controller in product(args.robots, args.controllers):

        if robot == "helix" and controller == "clf_qp":
            continue

        if robot == "spirob":
            continue

        valid_configs.append((robot, controller))

    total_experiments = len(valid_configs) * len(omega_list)

    print(f"Starting tracking suite: {total_experiments} total experiments")
    print(f"Robots: {args.robots}")
    print(f"Controllers: {args.controllers}")
    print(f"Omegas: {omega_list}")
    print("=" * 80)

    for robot, controller in valid_configs:
        try:
            for omega in omega_list:
                run_single_experiment(
                    robot,
                    controller,
                    omega=omega
                )

                completed_experiments += 1

                print(
                    f"Progress: {completed_experiments}/{total_experiments} "
                    f"({100 * completed_experiments / total_experiments:.1f}%)"
                )

        except Exception as e:
            print(f"ERROR in {robot} + {controller} + tracking: {e}")
            completed_experiments += 1

    total_time = time.time() - total_start_time

    print("=" * 80)
    print("Tracking suite completed!")
    print(f"Total time: {total_time:.2f} seconds ({total_time/60:.1f} minutes)")
    print(
        f"Average time per experiment: "
        f"{total_time/total_experiments:.2f} seconds"
    )
    print(f"Completed: {completed_experiments}/{total_experiments}")


if __name__ == "__main__":
    main()