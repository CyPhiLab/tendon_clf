#!/usr/bin/env python3

import argparse
import time
from itertools import product
from utils import *


def run_single_experiment(robot, controller, experiment,
                          target_pos=None, sim_duration=10.0, omega=None):

    print(
        f"Running: {robot} + {controller} + {experiment}"
        + (f" + {target_pos}" if target_pos else "")
        + (f" + {omega}" if omega else "")
    )

    start_time = time.time()

    results = simulate_model(
        headless=True,
        control_scheme=controller,
        target_pos=target_pos,
        controller=None,
        experiment=experiment,
        model_name=robot,
        sim_duration=sim_duration,
        omega=omega
    )

    end_time = time.time()

    results["wall_time"] = end_time - start_time
    results["sim_time_total"] = results["sim_time"][-1]

    csv_path = save_results(
        results=results,
        experiment=experiment,
        control_scheme=controller,
        model_name=robot,
        target_pos=target_pos,
        omega=omega
    )

    print(f"  → Completed in {end_time - start_time:.2f}s, saved to {csv_path}")
    return results


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--robots",
        nargs="+",
        default=["helix", "tendon", "spirob"],
        choices=["helix", "tendon", "spirob"],
    )
    parser.add_argument(
        "--controllers",
        nargs="+",
        default=["id_clf_qp", "impedance", "osc", "impedance_QP", "clf_qp", "uosc"],
        choices=["id_clf_qp", "impedance", "osc", "impedance_QP", "clf_qp", "uosc"],
    )
    parser.add_argument(
        "--experiments",
        nargs="+",
        default=["set", "tracking"],
        choices=["set", "tracking"],
    )
    parser.add_argument(
        "--target_positions",
        nargs="+",
        default=["pos1", "pos2", "pos3", "pos4"],
        choices=["pos1", "pos2", "pos3", "pos4"],
    )
    parser.add_argument("--sim_duration", type=float, default=10.0)
    args = parser.parse_args()

    omega_list = ["omg1", "omg2", "omg3", "omg4"]

    total_start_time = time.time()
    completed_experiments = 0
    total_experiments = 0

    valid_configs = []

    for robot, controller, experiment in product(
        args.robots, args.controllers, args.experiments
    ):

        if robot == "helix" and controller == "clf_qp":
            continue

        if robot == "spirob" and experiment == "set" and controller in [
            "impedance",
            "osc",
            "uosc",
            "clf_qp",
        ]:
            continue

        if robot == "spirob" and experiment == "tracking":
            continue

        valid_configs.append((robot, controller, experiment))

        if experiment == "set":
            total_experiments += len(args.target_positions)
        else:
            total_experiments += len(omega_list)

    print(f"Starting experiment suite: {total_experiments} total experiments")
    print(f"Robots: {args.robots}")
    print(f"Controllers: {args.controllers}")
    print(f"Experiments: {args.experiments}")
    print(f"Simulation duration: {args.sim_duration}s")
    if "set" in args.experiments:
        print(f"Target positions: {args.target_positions}")
    print("=" * 80)

    for robot, controller, experiment in valid_configs:
        try:
            if experiment == "set":
                for target_pos in args.target_positions:
                    run_single_experiment(
                        robot,
                        controller,
                        experiment,
                        target_pos=target_pos,
                        sim_duration=args.sim_duration,
                    )
                    completed_experiments += 1
                    print(
                        f"Progress: {completed_experiments}/{total_experiments} "
                        f"({100 * completed_experiments / total_experiments:.1f}%)"
                    )
            else:
                for omega in omega_list:
                    run_single_experiment(
                        robot,
                        controller,
                        experiment,
                        omega=omega,
                    )
                    completed_experiments += 1
                    print(
                        f"Progress: {completed_experiments}/{total_experiments} "
                        f"({100 * completed_experiments / total_experiments:.1f}%)"
                    )

        except Exception as e:
            print(f"ERROR in {robot} + {controller} + {experiment}: {e}")
            completed_experiments += 1

    total_time = time.time() - total_start_time

    print("=" * 80)
    print("Experiment suite completed!")
    print(f"Total time: {total_time:.2f} seconds ({total_time/60:.1f} minutes)")
    print(
        f"Average time per experiment: "
        f"{total_time/total_experiments:.2f} seconds"
    )
    print(f"Completed: {completed_experiments}/{total_experiments}")


if __name__ == "__main__":
    main()