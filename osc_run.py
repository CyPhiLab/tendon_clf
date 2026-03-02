#!/usr/bin/env python3

import argparse
import time
from utils import *


def run_tracking(robot, omega):
    print(f"Running: {robot} + osc + tracking + {omega}")

    start_time = time.time()

    results = simulate_model(
        headless=True,
        control_scheme="osc",
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
        control_scheme="osc",
        model_name=robot,
        target_pos=None,
        omega=omega
    )

    print(f"  → Completed in {end_time - start_time:.2f}s, saved to {csv_path}")

    return results


def run_setpoint(robot, target_pos):
    print(f"Running: {robot} + osc + set + {target_pos}")

    start_time = time.time()

    results = simulate_model(
        headless=True,
        control_scheme="osc",
        target_pos=target_pos,
        controller=None,
        experiment="set",
        model_name=robot
    )

    end_time = time.time()

    results["wall_time"] = end_time - start_time
    results["sim_time_total"] = results["sim_time"][-1]

    csv_path = save_results(
        results=results,
        experiment="set",
        control_scheme="osc",
        model_name=robot,
        target_pos=target_pos
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

    args = parser.parse_args()

    omega_list = ["omg1", "omg2", "omg3", "omg4", "omg5"]
    target_list = ["pos1", "pos2", "pos3", "pos4"]

    valid_robots = [r for r in args.robots if r != "spirob"]

    total_experiments = (
        len(valid_robots) * len(omega_list)
        + len(valid_robots) * len(target_list)
    )

    print(f"Starting OSC suite: {total_experiments} total experiments")
    print("=" * 80)

    total_start_time = time.time()
    completed_experiments = 0

    for robot in valid_robots:
        try:
            for target_pos in target_list:
                run_setpoint(robot, target_pos)
                completed_experiments += 1
                print(
                    f"Progress: {completed_experiments}/{total_experiments} "
                    f"({100 * completed_experiments / total_experiments:.1f}%)"
                )

            for omega in omega_list:
                run_tracking(robot, omega)
                completed_experiments += 1
                print(
                    f"Progress: {completed_experiments}/{total_experiments} "
                    f"({100 * completed_experiments / total_experiments:.1f}%)"
                )

        except Exception as e:
            print(f"ERROR in {robot} + osc: {e}")
            completed_experiments += 1

    total_time = time.time() - total_start_time

    print("=" * 80)
    print("OSC suite completed!")
    print(f"Total time: {total_time:.2f} seconds ({total_time/60:.1f} minutes)")
    print(f"Average time per experiment: {total_time/total_experiments:.2f} seconds")
    print(f"Completed: {completed_experiments}/{total_experiments}")


if __name__ == "__main__":
    main()