import time
from pathlib import Path

import matplotlib.pyplot as plt
import mujoco
import mujoco.viewer
import numpy as np
from sklearn.cluster import DBSCAN
from vicon_dssdk import ViconDataStream


# ============================================================
# Configuration
# ============================================================

LAB_PC_IP = "172.20.54.13"
REPO_ROOT = Path(__file__).resolve().parent
OUTPUT_PATH = REPO_ROOT / "vicon_dbscan_summary.txt"
PLOT_PATH = REPO_ROOT / "vicon_mujoco_markers.png"
VIEWER_DURATION_SEC = 10.0

INIT_FRAMES = 100
DBSCAN_EPS = 1.5e-3  # 1.5 mm
DBSCAN_MIN_SAMPLES = 10

MODEL_PATH = REPO_ROOT / "mujoco_models" / "spirob" / "spirob_horz_control.xml"
REFERENCE_MARKER_NAMES = [
    "marker_1_seg_2",
    "marker_2_seg_2",
    "marker_3_seg_2",
    "marker_1_seg_5",
    "marker_2_seg_5",
    "marker_3_seg_5",
    "marker_1_seg_8",
    "marker_2_seg_8",
    "marker_3_seg_8",
]


# ============================================================
# MuJoCo reference geometry
# ============================================================

def load_mujoco_model_with_init():
    model = mujoco.MjModel.from_xml_path(str(MODEL_PATH))
    data = mujoco.MjData(model)

    key_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_KEY, "init")
    if key_id >= 0:
        data.qpos[:] = model.key_qpos[key_id]
        mujoco.mj_forward(model, data)
    else:
        raise RuntimeError("Could not find MuJoCo keyframe named 'init'.")

    return model, data


def load_mujoco_marker_positions():
    model, data = load_mujoco_model_with_init()

    ref = {}
    for name in REFERENCE_MARKER_NAMES:
        site_id = model.site(name).id
        ref[name] = data.site_xpos[site_id].copy()
    return ref


def show_mujoco_viewer():
    model, data = load_mujoco_model_with_init()
    print(f"Launching MuJoCo viewer at keyframe 'init' for {VIEWER_DURATION_SEC:.0f} seconds...")
    with mujoco.viewer.launch_passive(model, data) as viewer:
        end_time = time.time() + VIEWER_DURATION_SEC
        while viewer.is_running() and time.time() < end_time:
            viewer.sync()
            time.sleep(1.0 / 60.0)


# ============================================================
# Clustering logic
# ============================================================


def compute_cluster_centers(points, labels):
    centers = []
    for cluster_id in sorted(set(labels)):
        if cluster_id < 0:
            continue
        cluster_points = points[labels == cluster_id]
        centers.append({
            "cluster_id": cluster_id,
            "count": int(len(cluster_points)),
            "center": cluster_points.mean(axis=0),
        })
    centers.sort(key=lambda item: item["count"], reverse=True)
    return centers


def estimate_rate_hz(frame_times):
    if len(frame_times) < 2:
        return None
    dt = np.diff(frame_times)
    dt = dt[dt > 0]
    if len(dt) == 0:
        return None
    return float(1.0 / np.median(dt))


def emit(lines):
    text = "\n".join(lines)
    print(text)
    OUTPUT_PATH.write_text(text + "\n")


def compute_marker_error(mujoco_pos, vicon_pos):
    error_vec = np.asarray(vicon_pos, dtype=float) - np.asarray(mujoco_pos, dtype=float)
    error_norm = float(np.linalg.norm(error_vec))
    return error_vec, error_norm


def plot_marker_positions(reference_positions, clustered_reference_positions, cluster_centers):
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection="3d")

    mujoco_xyz = np.array([reference_positions[name] for name in REFERENCE_MARKER_NAMES], dtype=float)
    ax.scatter(
        mujoco_xyz[:, 0], mujoco_xyz[:, 1], mujoco_xyz[:, 2],
        c="tab:red", marker="^", s=80, label="MuJoCo markers",
    )

    for name, pos in zip(REFERENCE_MARKER_NAMES, mujoco_xyz):
        ax.text(pos[0], pos[1], pos[2], name, color="tab:red", fontsize=8)

    matched_names = [name for name in REFERENCE_MARKER_NAMES if name in clustered_reference_positions]
    if matched_names:
        vicon_xyz = np.array([clustered_reference_positions[name] for name in matched_names], dtype=float)
        ax.scatter(
            vicon_xyz[:, 0], vicon_xyz[:, 1], vicon_xyz[:, 2],
            c="tab:blue", marker="o", s=80, label="Vicon DBSCAN centers",
        )
        for name, pos in zip(matched_names, vicon_xyz):
            ax.text(pos[0], pos[1], pos[2], name, color="tab:blue", fontsize=8)

    if cluster_centers:
        raw_cluster_xyz = np.array([cluster["center"] for cluster in cluster_centers], dtype=float)
        ax.scatter(
            raw_cluster_xyz[:, 0], raw_cluster_xyz[:, 1], raw_cluster_xyz[:, 2],
            c="gray", marker="x", s=60, alpha=0.7, label="All Vicon clusters",
        )

    ax.set_title("MuJoCo vs Vicon marker positions")
    ax.set_xlabel("x [m]")
    ax.set_ylabel("y [m]")
    ax.set_zlabel("z [m]")
    ax.legend()
    plt.tight_layout()
    fig.savefig(PLOT_PATH, dpi=200, bbox_inches="tight")
    print(f"Saved marker comparison plot to: {PLOT_PATH}")
    plt.show()


def collect_initial_clusters(client):
    frame_times = []
    collected_points = []

    while len(frame_times) < INIT_FRAMES:
        if not client.GetFrame():
            continue

        try:
            unlabeled = client.GetUnlabeledMarkers()
        except ViconDataStream.DataStreamException:
            continue

        frame_time = time.time()
        frame_times.append(frame_time)

        points = np.array([
            [marker[0][0] / 1000.0, marker[0][1] / 1000.0, marker[0][2] / 1000.0]
            for marker in unlabeled
        ], dtype=float)

        if len(points) > 0:
            collected_points.append(points)

        print(
            f"  Init frame {len(frame_times):03d}/{INIT_FRAMES}: visible markers = {len(points)}",
            end="\r",
            flush=True,
        )

    print()

    if not collected_points:
        raise RuntimeError("No Vicon markers were collected during DBSCAN initialization.")

    all_points = np.vstack(collected_points)
    labels = DBSCAN(eps=DBSCAN_EPS, min_samples=DBSCAN_MIN_SAMPLES).fit_predict(all_points)
    cluster_centers = compute_cluster_centers(all_points, labels)
    return cluster_centers, frame_times


# ============================================================
# Matching logic
# ============================================================

def match_vicon_to_reference(reference_positions, vicon_points):
    """Assign each Vicon marker to the nearest reference marker position.

    This is a greedy nearest-neighbor assignment: each reference marker gets the
    closest remaining Vicon point. It is a practical match when the real marker
    identity is unknown but the geometry is approximately known.
    """
    if len(vicon_points) == 0:
        return {}

    ref_names = list(reference_positions.keys())
    vicon_used = set()
    assignment = {}

    for ref_name in ref_names:
        ref_pos = np.asarray(reference_positions[ref_name], dtype=float)
        best_index = None
        best_dist = np.inf

        for idx, point in enumerate(vicon_points):
            if idx in vicon_used:
                continue
            dist = np.linalg.norm(np.asarray(point, dtype=float) - ref_pos)
            if dist < best_dist:
                best_dist = dist
                best_index = idx

        if best_index is not None:
            assignment[ref_name] = np.asarray(vicon_points[best_index], dtype=float)
            vicon_used.add(best_index)

    return assignment


# ============================================================
# Vicon stream
# ============================================================

def main():
    show_mujoco_viewer()

    reference_positions = load_mujoco_marker_positions()
    lines = ["Reference marker positions loaded from MuJoCo init keyframe:"]
    for name in REFERENCE_MARKER_NAMES:
        p = reference_positions[name]
        lines.append(f"  {name}: ({p[0]:.4f}, {p[1]:.4f}, {p[2]:.4f})")

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
    client.SetStreamMode(ViconDataStream.Client.StreamMode.EClientPull)

    cluster_centers, init_frame_times = collect_initial_clusters(client)
    rate_hz = estimate_rate_hz(init_frame_times)

    lines.append("")
    lines.append(
        f"DBSCAN found {len(cluster_centers)} clusters with "
        f"eps={DBSCAN_EPS * 1000:.1f} mm and min_samples={DBSCAN_MIN_SAMPLES}.")
    for cluster in cluster_centers:
        p = cluster["center"]
        lines.append(
            f"  Cluster {cluster['cluster_id']}: count={cluster['count']}, "
            f"center=({p[0]:.4f}, {p[1]:.4f}, {p[2]:.4f})")

    clustered_reference_positions = match_vicon_to_reference(
        reference_positions,
        np.array([cluster["center"] for cluster in cluster_centers], dtype=float),
    )
    if len(clustered_reference_positions) < len(REFERENCE_MARKER_NAMES):
        lines.append(
            f"Warning: only identified {len(clustered_reference_positions)} / "
            f"{len(REFERENCE_MARKER_NAMES)} desired markers from DBSCAN clusters.")

    lines.append("")
    lines.append("Matched Vicon DBSCAN centers vs MuJoCo init positions:")
    error_norms = []
    for name in REFERENCE_MARKER_NAMES:
        mujoco_pos = reference_positions[name]
        if name in clustered_reference_positions:
            vicon_pos = clustered_reference_positions[name]
            error_vec, error_norm = compute_marker_error(mujoco_pos, vicon_pos)
            error_norms.append(error_norm)
            lines.append(
                f"  {name}:\n"
                f"    Vicon : ({vicon_pos[0]:.4f}, {vicon_pos[1]:.4f}, {vicon_pos[2]:.4f})\n"
                f"    MuJoCo: ({mujoco_pos[0]:.4f}, {mujoco_pos[1]:.4f}, {mujoco_pos[2]:.4f})\n"
                f"    Error : (dx={error_vec[0]:+.4f}, dy={error_vec[1]:+.4f}, dz={error_vec[2]:+.4f}) m, "
                f"||e||={error_norm:.4f} m"
            )
        else:
            lines.append(
                f"  {name}:\n"
                f"    Vicon : not identified\n"
                f"    MuJoCo: ({mujoco_pos[0]:.4f}, {mujoco_pos[1]:.4f}, {mujoco_pos[2]:.4f})"
            )

    if error_norms:
        lines.append("")
        lines.append(
            f"Mean marker position error: {np.mean(error_norms):.4f} m "
            f"({1000.0 * np.mean(error_norms):.2f} mm)")
        lines.append(
            f"Max marker position error : {np.max(error_norms):.4f} m "
            f"({1000.0 * np.max(error_norms):.2f} mm)")

    if rate_hz is not None:
        lines.append("")
        lines.append(f"Estimated Vicon frame rate: {rate_hz:.2f} Hz")

    lines.append("")
    lines.append(f"Saved summary to: {OUTPUT_PATH}")
    lines.append("Done.")
    emit(lines)
    plot_marker_positions(reference_positions, clustered_reference_positions, cluster_centers)


if __name__ == "__main__":
    main()