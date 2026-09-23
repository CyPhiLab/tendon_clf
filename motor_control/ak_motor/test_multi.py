import os
import time
import argparse
import numpy as np
import matplotlib.pyplot as plt

from motor_can import AKController

current_limit = 1.9  # A
speed_limit = 180     # RPM
k_t = 0.127  # Nm/A
k_v = 75     # RPM/V


def emergency_stop(ctrl):
    try:
        ctrl.stop()
        time.sleep(0.02)
        print("Emergency stop executed successfully.")
    except Exception as e:
        print(f"Error during emergency stop: {e}")

    try:
        ctrl.disable()
        time.sleep(0.02)
        print("Motors disabled successfully.")
    except Exception as e:
        print(f"Error during motor disable: {e}")


def plot_logs(motor_ids, t_log, pos_log, velocity_log, current_log):
    """pos_log/velocity_log/current_log are dicts keyed by motor_id -> list."""
    plt.figure(figsize=(12, 8))

    plt.subplot(3, 1, 1)
    for mid in motor_ids:
        plt.plot(t_log, pos_log[mid], label=f'id={mid}')
    plt.ylabel('Position (deg)')
    plt.grid()
    plt.legend()

    plt.subplot(3, 1, 2)
    for mid in motor_ids:
        plt.plot(t_log, velocity_log[mid], label=f'id={mid}')
    plt.ylabel('Velocity (RPM)')
    plt.grid()
    plt.legend()

    plt.subplot(3, 1, 3)
    for mid in motor_ids:
        plt.plot(t_log, current_log[mid], label=f'id={mid}')
    plt.xlabel('Time (s)')
    plt.ylabel('Current (A)')
    plt.grid()
    plt.legend()

    plt.tight_layout()


def main():
    parser = argparse.ArgumentParser(description="Test multiple AK motors on a shared CAN bus")
    parser.add_argument("--motor_ids", type=int, nargs="+", default=[101, 102, 103],
                         help="List of motor IDs, e.g. --motor_ids 101 102 103")
    parser.add_argument("--channel", type=str, default="PCAN_USBBUS1",
                         choices=["PCAN_USBBUS1", "PCAN_USBBUS2"])
    parser.add_argument("--command", type=str, default="position",
                         choices=["position", "velocity", "current"])
    parser.add_argument("--values", type=float, nargs="+", default=None,
                         help="Command value per motor (same order as --motor_ids). "
                              "If a single value is given, it is applied to all motors.")
    parser.add_argument("--duration", type=float, default=5.0)
    parser.add_argument("--dt", type=float, default=0.05)
    args = parser.parse_args()

    motor_ids = args.motor_ids
    n = len(motor_ids)

    # Broadcast / validate command values
    if args.values is None:
        values = [0.0] * n
    elif len(args.values) == 1:
        values = args.values * n
    elif len(args.values) == n:
        values = args.values
    else:
        raise ValueError(f"--values must have length 1 or {n} (matching --motor_ids), "
                          f"got {len(args.values)}")

    # Logs, keyed by motor_id
    t_log = []
    pos_log = {mid: [] for mid in motor_ids}
    velocity_log = {mid: [] for mid in motor_ids}
    current_log = {mid: [] for mid in motor_ids}

    ctrl = AKController.make_servo(motor_ids=motor_ids, channel=args.channel)

    with ctrl:
        t_initial = time.monotonic()
        estop_done = False
        try:
            while True:
                t = time.monotonic() - t_initial
                if t > args.duration:
                    emergency_stop(ctrl)
                    estop_done = True
                    break

                # Send command based on the selected mode (broadcast to all motors)
                if args.command == "position":
                    ctrl.set_position(values)
                elif args.command == "velocity":
                    clipped = list(np.clip(values, -speed_limit, speed_limit))
                    ctrl.set_rpm(clipped)
                elif args.command == "current":
                    clipped = list(np.clip(values, -current_limit, current_limit))
                    ctrl.set_current(clipped)

                # Read all motor states in one pass
                feedbacks = ctrl.get_feedback_all_motors(timeout=args.dt)
                if any(fb is None for fb in feedbacks):
                    missing = [mid for mid, fb in zip(motor_ids, feedbacks) if fb is None]
                    print(f"No feedback received from motor id(s) {missing}. Stopping the test.")
                    emergency_stop(ctrl)
                    estop_done = True
                    break

                for mid, fb in zip(motor_ids, feedbacks):
                    pos_log[mid].append(fb['position'])
                    velocity_log[mid].append(fb['speed'])
                    current_log[mid].append(fb['current'])
                t_log.append(t)

        except KeyboardInterrupt:
            emergency_stop(ctrl)

        except Exception as e:
            print(f"An error occurred: {e}")
            emergency_stop(ctrl)
            estop_done = True
            raise

        finally:
            if not estop_done:
                emergency_stop(ctrl)

            # Save logs to a file (one file per motor)
            for mid in motor_ids:
                log_filename = f"motor_{mid}_log.txt"
                with open(log_filename, 'w') as f:
                    f.write("Time(s),Position(deg),Velocity(RPM),Current(A)\n")
                    for t, pos, vel, curr in zip(t_log, pos_log[mid], velocity_log[mid], current_log[mid]):
                        f.write(f"{t:.3f},{pos:.3f},{vel:.3f},{curr:.3f}\n")
                print(f"Logs saved to {log_filename}")

            plot_logs(motor_ids, t_log, pos_log, velocity_log, current_log)
            ids_str = "_".join(str(mid) for mid in motor_ids)
            plot_path = f"plots/motors_{ids_str}_plot.png"
            os.makedirs("plots", exist_ok=True)
            plt.savefig(plot_path, dpi=150, bbox_inches="tight")
            plt.show()
            print(f"Plot saved to {plot_path}")


if __name__ == "__main__":
    main()

# Example usage:
# python3 cubemars_motor/ak_motor/test_multi.py --motor_ids 101 102 103 --command position --values 90 0 -45
# python3 cubemars_motor/ak_motor/test_multi.py --motor_ids 101 102 --command position --values 90   # same value for all motors
