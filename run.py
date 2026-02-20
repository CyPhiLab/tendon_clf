import argparse
import sys
import time
from utils import *

# Gain for the orientation component of the twist computation. This should be
# between 0 and 1. 0 means no movement, 1 means move the end-effector to the target
# orientation in one integration step.


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Run optimized tendon control simulation')
    parser.add_argument('--headless', action='store_true', help='Run simulation without GUI for maximum performance')
    parser.add_argument('--no-plots', action='store_true', help='Skip generating plots at the end')
    parser.add_argument('--control', type=str, default='id_clf_qp', choices=['id_clf_qp', 'impedance', 'mpc', 'impedance_QP'], help='Controller type to use')
    parser.add_argument('--robot', type=str, default='helix', choices=['helix', 'tendon','spirob'], help='Robot to simulate')
    parser.add_argument('--experiment', type=str, default='set', choices=['set', 'tracking'], help='Experiment name')
    parser.add_argument('--target_pos', type=str, default='pos4', choices=['pos1', 'pos2', 'pos3', 'pos4'], help='Target position for the end-effector')
    parser.add_argument('--verbose', action='store_true', help='Print detailed system information')
    args = parser.parse_args()
    
    # Print system info if requested
    if args.verbose:
        print(f"Running: {args.robot} robot with {args.control} controller")
        print(f"Experiment: {args.experiment}, Target: {args.target_pos if args.experiment == 'set' else 'trajectory'}")
    
    # Basic validation
    if args.robot == 'spirob' and args.experiment == 'tracking':
        print("Error: SpiRob robot only supports 'set' experiments, not 'tracking'")
        sys.exit(1)
    
    # Record start time for performance measurement
    start_time = time.time()

    # Simulate the model
    results = simulate_model(headless=args.headless,
                            control_scheme=args.control, 
                            target_pos=args.target_pos, 
                            controller=None,  # Controller logic now in Robot class
                            experiment=args.experiment, 
                            model_name=args.robot)
    
    # Record end time and add runtime data to results
    end_time = time.time()
    results['wall_time'] = end_time - start_time
    results['sim_time_total'] = results['sim_time'][-1]
    
    print(f"Simulation completed in {end_time - start_time:.2f} seconds (wall clock time)")
    print(f"Simulated {results['sim_time'][-1]:.3f} seconds of physics time")
    print(f"Performance ratio: {results['sim_time'][-1] / (end_time - start_time):.2f}x real-time")
    
    if args.no_plots:
        print("Skipping plot generation")
    
    print(results['task_error'][-1])
    
    csv_path = save_results(
        results=results,
        experiment=args.experiment,
        control_scheme=args.control,
        model_name=args.robot,
        target_pos=args.target_pos
    )

    print(f"Results saved to {csv_path}")


