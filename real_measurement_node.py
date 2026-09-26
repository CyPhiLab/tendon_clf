import time
import numpy as np
from vicon_dssdk import ViconDataStream


# ============================================================
# Configuration
# ============================================================

LAB_PC_IP = "192.168.0.144"


# ============================================================
# Connect to Vicon
# ============================================================

client = ViconDataStream.Client()

print(f"Connecting to Vicon at {LAB_PC_IP}...")

while not client.IsConnected():
    try:
        client.Connect(LAB_PC_IP)
    except ViconDataStream.DataStreamException:
        print("  Retrying...", flush=True)
        time.sleep(1)

print("Connected!")

client.EnableMarkerData()
client.EnableUnlabeledMarkerData()

client.SetStreamMode(
    ViconDataStream.Client.StreamMode.EClientPull
)


# ============================================================
# Measurement loop
# ============================================================

print("\nStarting Vicon measurement stream...")
print("Press Ctrl+C to stop.\n")


try:

    while True:

        # Wait for next Vicon frame
        if not client.GetFrame():
            continue

        # Get all currently visible unlabeled markers
        try:
            unlabeled = client.GetUnlabeledMarkers()
        except ViconDataStream.DataStreamException:
            continue

        # ----------------------------------------------------
        # Extract XYZ positions
        #
        # Vicon returns positions in mm.
        # Convert to meters.
        # ----------------------------------------------------

        markers_vicon = np.array([
            [
                marker[0][0] / 1000.0,
                marker[0][1] / 1000.0,
                marker[0][2] / 1000.0
            ]
            for marker in unlabeled
        ])

        # ----------------------------------------------------
        # Print number of visible markers
        # ----------------------------------------------------

        print(
            f"\nVisible markers: {len(markers_vicon)}"
        )

        # ----------------------------------------------------
        # Print positions
        # ----------------------------------------------------

        for i, position in enumerate(markers_vicon):

            x, y, z = position

            print(
                f"Marker {i:02d}: "
                f"x={x:+.4f}, "
                f"y={y:+.4f}, "
                f"z={z:+.4f}"
            )

        # ~100 Hz
        time.sleep(0.01)


except KeyboardInterrupt:

    print("\nStopping Vicon measurement stream...")


print("Done.")