import os
import sys
import time
import argparse
import numpy as np
import matplotlib.pyplot as plt

from motor_can import AKMotorCAN

current_limit = 1.9 # A
speed_limit = 180 # RPM
k_t = 0.127 # Nm/A
k_v = 75 # RPM/V

def emergency_stop(motor):
    try:
        motor.stop()
        time.sleep(0.02)  # Wait for the motor to stop
        print("Emergency stop executed successfully.")
    except Exception as e:
        print(f"Error during emergency stop: {e}")

    try:
        motor.disable()
        time.sleep(0.02)  # Wait for the motor to disable
        print("Motor disabled successfully.")
    except Exception as e:
        print(f"Error during motor disable: {e}")

def plot_logs(t_log, pos_log, velocity_log, current_log):
    plt.figure(figsize=(12, 8))

    plt.subplot(3, 1, 1)
    plt.plot(t_log, pos_log, label='Position (deg)', color='blue')
    plt.ylabel('Position (deg)')
    plt.grid()
    plt.legend()

    plt.subplot(3, 1, 2)
    plt.plot(t_log, velocity_log, label='Velocity (RPM)', color='orange')
    plt.ylabel('Velocity (RPM)')
    plt.grid()
    plt.legend()

    plt.subplot(3, 1, 3)
    plt.plot(t_log, current_log, label='Current (A)', color='green')
    plt.xlabel('Time (s)')
    plt.ylabel('Current (A)')
    plt.grid()
    plt.legend()

    plt.tight_layout()

def main():
    parser = argparse.ArgumentParser(description="Test AKMotorCAN")
    parser.add_argument("--motor_id", type=int, default=104)
    parser.add_argument("--channel", type=str, default="PCAN_USBBUS1", choices=["PCAN_USBBUS1", "PCAN_USBBUS2"])
    parser.add_argument("--command", type=str, default="position", choices=["position", "velocity", "current"])
    parser.add_argument("--value", type=float, default=0.0)  # Command value
    parser.add_argument("--duration", type=float, default=5.0)   # seconds
    parser.add_argument("--dt", type=float, default=0.05)        # loop period
    args = parser.parse_args()

    # Logs
    t_log = []
    pos_log = []
    velocity_log = []
    current_log = []

    with AKMotorCAN(args.motor_id) as motor:
        t_initial = time.monotonic()
        estop_done = False
        try:
            while True:
                t = time.monotonic() - t_initial
                if t > args.duration:
                    emergency_stop(motor)
                    estop_done = True
                    break

                # Send command based on the selected mode
                if args.command == "position":
                    motor.set_position(args.value)
                elif args.command == "velocity":
                    value = np.clip(args.value, -speed_limit, speed_limit)
                    motor.set_rpm(value)
                elif args.command == "current":
                    value = np.clip(args.value, -current_limit, current_limit)
                    motor.set_current(value)

                # Read motor state
                feedback = motor.receive_feedback()
                if feedback is None:
                    print("No feedback received. Stopping the test.")
                    emergency_stop(motor)
                    estop_done = True
                    break

                pos_log.append(feedback['position'])
                velocity_log.append(feedback['speed'])
                current_log.append(feedback['current'])
                t_log.append(t)
                time.sleep(args.dt)

        except KeyboardInterrupt:
            emergency_stop(motor)

        except Exception as e:
            print(f"An error occurred: {e}")
            emergency_stop(motor)
            estop_done = True
            raise

        finally:
            if not estop_done:
                emergency_stop(motor)

            # Save logs to a file
            log_filename = f"motor_{args.motor_id}_log.txt"
            with open(log_filename, 'w') as f:
                f.write("Time(s),Position(deg),Velocity(RPM),Current(A)\n")
                for t, pos, vel, curr in zip(t_log, pos_log, velocity_log, current_log):
                    f.write(f"{t:.3f},{pos:.3f},{vel:.3f},{curr:.3f}\n")
            print(f"Logs saved to {log_filename}")

            plot_logs(t_log, pos_log, velocity_log, current_log)
            plot_path = f"plots/motor_{args.motor_id}_plot.png"
            os.makedirs("plots", exist_ok=True)
            plt.savefig(plot_path, dpi=150, bbox_inches="tight")
            plt.show()
            print(f"Plot saved to {plot_path}")

if __name__ == "__main__":
    main()

# python3 cubemars_motor/ak_motor/test.py --motor_id 2 --command position --value 90

## Debugging and testing notes:
# 0. Downloading and installing the PCANBasic library is required for this code to run. Make sure to follow the installation instructions provided by PEAK-System.
# 1. Confirm that the PCANBasic library is installed and accessible in your Python environment by running:
# find /usr /usr/local /opt -iname 'libpcanbasic.so*' 2>/dev/null
# 2. Confirm PCANBasic installation: 
# python3 -c "from can.interfaces.pcan.basic import PCANBasic; print('pcanbasic ok')"
# python3 -c "from can.interfaces.pcan.basic import PCANBasic; PCANBasic(); print('init ok')"
# 3. Confirm that the PCAN device is connected and recognized by the system:
# lsusb | grep -i -E 'peak|pcan'
# 4. Confrim that the PCAN driver is installed
# sudo find / -type f \( -name 'pcan.ko' -o -name 'pcan.ko.*' \) 2>/dev/null
# 5. Confirm that the PCAN device is accessible via the can library. Could try BUS1, BUS2, etc. depending on your setup:
# python3 -c "import can; b=can.Bus(interface='pcan', channel='PCAN_USBBUS1', bitrate=1000000); print('open ok'); b.shutdown()"
# 6. If that fails because driver is not loaded to the currently running kernel, check the current kernel version and compare it with the kernel version for which the PCAN driver was built. If they differ, you may need to rebuild the PCAN driver for your current kernel.
# uname -r

## Reinstalling the PCAN driver to match the current kernel version with the same PEAK source tree and rebuild against the currently running kernel.
# # 1) Confirm running kernel
# uname -r

# # 2) Install build tools + matching headers
# sudo apt update
# sudo apt install -y build-essential linux-headers-$(uname -r)

# # 3) Rebuild PEAK driver for this kernel
# cd ~/peak-linux-driver-9.2.0 # this is just an example, use your own path to the PEAK driver source tree
# make clean
# make
# sudo make install

# # 4) Refresh module map and load module
# sudo depmod -a
# sudo modprobe pcan

## 5) Verify that the driver is loaded and matches the current kernel version
# modinfo pcan | grep -E 'filename|vermagic'
# lsmod | grep pcan
# cat /proc/pcan

## 6) Test CAN open again:
# python3 -c "import can; b=can.Bus(interface='pcan', channel='PCAN_USBBUS1', bitrate=1000000); print('open ok'); b.shutdown()"