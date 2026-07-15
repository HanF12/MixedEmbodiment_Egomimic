import numpy as np
import csv
import pandas as pd
import os
try:
    import h5py  # type: ignore
except Exception:  # pragma: no cover
    h5py = None

def calculate_fps_array(input: np.array, savepath: str = None):
    times = input.astype(np.float64)
    n = times.size

    # Pre-allocate the fps array
    fps = np.empty_like(times)

    if n == 1:
        fps[0] = np.nan      # not enough data to estimate a rate
    else:
        # Forward-difference for the first element
        fps[0] = 1.0 / (times[1] - times[0])

        if n > 2:
            # Central-difference (two-frame window) for the interior elements
            fps[1:-1] = 2.0 / (times[2:] - times[:-2])
        else:
            # If we have exactly two samples, reuse the first estimate
            fps[1:-1] = fps[0]

        # Backward-difference for the last element
        fps[-1] = 1.0 / (times[-1] - times[-2])

    # Replace any infinite or non-positive values (bad/duplicate timestamps) with NaN
    fps[~np.isfinite(fps) | (fps <= 0)] = np.nan

    # Save the result
    if savepath:
        np.save(f"{savepath}.npy", fps)
    return fps

def first_fps_target_instance(input: np.array, target:float = 30.0, threshold: float = 0.15):
    """
    Calculate the first FPS target instance from a given array of timestamps.
    
    Args:
        input (np.array): Array of timestamps.
        
    Returns:
        float: The first FPS target instance, or NaN if not enough data.
    """
    print(f'Target FPS: {target}, Threshold: {threshold}')
    length = input.size
    for i in range(length):
        if abs(target - input[i]) <= threshold:
            print(f"Found first FPS target instance at index {i}: {input[i]}")
            return input[i]
    print("No valid FPS target instance found.")
    return -1

def hdf5_to_csv(hdf5_path, csv_path):
    """
    Read an HDF5 file containing joint state data and write its contents to a CSV file.

    The HDF5 file is expected to contain these datasets:
      - 'timestamps'   : 1D array of shape (N,)
      - 'joint_names'  : 1D array of byte strings of length (J,)
      - 'positions'    : 2D array of shape (N, J)
      - 'velocities'   : 2D array of shape (N, J)

    The output CSV will have columns:
      timestamp, <joint1>_position, <joint1>_velocity, <joint2>_position, <joint2>_velocity, ...

    Args:
        hdf5_path (str): Path to the input HDF5 file.
        csv_path  (str): Path where the output CSV should be written.
    """
    if h5py is None:
        raise ImportError("h5py is required for hdf5_to_csv(); install with `pip install h5py`.")
    if not os.path.exists(hdf5_path):
        raise FileNotFoundError(f"HDF5 file not found: {hdf5_path}")

    with h5py.File(hdf5_path, 'r') as f:
        # Verify all required datasets are present
        for ds in ('timestamps', 'joint_names', 'positions', 'velocities'):
            if ds not in f:
                raise KeyError(f"Expected dataset '{ds}' not found in {hdf5_path}")

        timestamps = f['timestamps'][:]                # shape: (N,)
        raw_joint_names = f['joint_names'][:]          # shape: (J,)
        joint_names = [name.decode('utf-8') for name in raw_joint_names]
        positions = f['positions'][:]                  # shape: (N, J)
        velocities = f['velocities'][:]                # shape: (N, J)

    num_samples = timestamps.shape[0]
    num_joints  = len(joint_names)

    # Check shapes
    if positions.shape != (num_samples, num_joints):
        raise ValueError(
            f"Positions shape {positions.shape} does not match "
            f"({num_samples}, {num_joints})"
        )
    if velocities.shape != (num_samples, num_joints):
        raise ValueError(
            f"Velocities shape {velocities.shape} does not match "
            f"({num_samples}, {num_joints})"
        )

    # Build CSV header
    header = ['timestamp']
    for name in joint_names:
        header.append(f"{name}_position")
        header.append(f"{name}_velocity")

    # Write CSV
    with open(csv_path, mode='w', newline='') as csv_file:
        writer = csv.writer(csv_file)
        writer.writerow(header)

        for i in range(num_samples):
            row = [timestamps[i]]
            for j in range(num_joints):
                row.append(positions[i, j])
                row.append(velocities[i, j])
            writer.writerow(row)

    print(f"[OK] Wrote {num_samples} rows → '{csv_path}'")

def arm_data_to_npy(csv_path: str, output_npy: str = None):
    """
    Convert a CSV file containing arm joint timestamps and joint data to a NumPy array containing only timestamps
    Returns a NumPy array of timestamps (for joint data)
    """
    # Check that the CSV file exists
    if not os.path.isfile(csv_path):
        raise FileNotFoundError(f"CSV file not found: {csv_path}")

    # Read the 'timestamp' column from the CSV
    with open(csv_path, newline="") as f:
        reader = csv.reader(f)
        header = next(reader)
        try:
            ts_idx = header.index("timestamp")
        except ValueError:
            raise KeyError(f"'system_timestamp' column not found in {csv_path}")

        timestamps_list = []
        for row in reader:
            # Convert each value to float (adjust if your timestamps are integers)
            timestamps_list.append(float(row[ts_idx]))

    # Convert to NumPy array
    timestamps = np.array(timestamps_list)

    # Save the NumPy array to disk
    if output_npy is not None:
        np.save(output_npy, timestamps)
        print(f"Saved {timestamps.shape[0]} timestamps to '{output_npy}'")
    return timestamps

def synchronize(joint_ts:np.array, realsense_ts_l:np.array,realsense_ts_r:np.array, bird_ts: np.array, out_csv: str, max_skew_s: float = 0.050, debug: bool = False):
    """
    Synchronise three timestamp arrays and write a CSV.

    Parameters:
        joints_ts : joint timestamp numpy array
        realsense_ts : realsense timestamp numpy array
        bird_ts : bird view timestamp numpy array
        out_csv : output csv name
        max_skew_s : maximum time different allowed (0.050s for 15fps cameras + 30Hz joints)
        debug : bool
            If True, include raw timestamps and per-row skew in the CSV.
            If False (default), output only the index columns.
    """
    # ---------- monotonicity checks ----------
    assert np.all(np.diff(joint_ts)     > 0), "joint_data not sorted"
    assert np.all(np.diff(realsense_ts_l) > 0), "left_data not sorted"
    assert np.all(np.diff(realsense_ts_r) > 0), "right_data not sorted"
    assert np.all(np.diff(bird_ts)      > 0), "bird_data not sorted"

    # ---------- single-pass sync ----------
    rows = []  # (master, i, j, k, jt, rt, bt, skew)
    i = j = k = m = master = 0
    A, B, C, D = len(joint_ts), len(realsense_ts_l), len(bird_ts), len(realsense_ts_r)

    while i < A and j < B and k < C and m < D:
        pivot = max(joint_ts[i], realsense_ts_l[j], bird_ts[k], realsense_ts_r[m])

        while i < A and pivot - joint_ts[i]     > max_skew_s: i += 1
        while j < B and pivot - realsense_ts_l[j] > max_skew_s: j += 1
        while k < C and pivot - bird_ts[k]      > max_skew_s: k += 1
        while m < D and pivot - realsense_ts_r[m] > max_skew_s: m += 1
        if i == A or j == B or k == C or m == D:
            break

        jt, rt, bt, rrt = joint_ts[i], realsense_ts_l[j], bird_ts[k], realsense_ts_r[m]
        t_min, t_max = min(jt, rt, bt,rrt), max(jt, rt, bt,rrt)

        if t_max - t_min <= max_skew_s:
            rows.append((master, i, j, k,m, jt, rt, bt,rrt, t_max - t_min))
            master += 1
            i += 1; j += 1; k += 1; m += 1
        else:
            earliest = np.argmin([jt, rt, bt,rrt])
            if   earliest == 0: i += 1
            elif earliest == 1: j += 1
            elif earliest == 2: k += 1
            else: m += 1

    # ---------- build DataFrame ----------
    all_cols = [
        "master_index",
        "joint_index", "left_index", "bird_index", "right_index",
        "joint_time",  "realsense_time",  "bird_time", "realsense_r_time",
        "time_diff"
    ]
    df = pd.DataFrame(rows, columns=all_cols)

    # keep only what we need
    if not debug:
        df = df[["master_index",
                 "joint_index", "left_index", "bird_index", "right_index"]]

    # ---------- save ----------
    df.to_csv(out_csv, index=False)
    print(f"Synced {len(df)} triplets → {out_csv} (debug={debug})")


def synchronize_with_front(
    joint_ts: np.array,
    realsense_ts_l: np.array,
    realsense_ts_r: np.array,
    bird_ts: np.array,
    front_ts: np.array,
    out_csv: str,
    max_skew_s: float = 0.050,
    debug: bool = False,
):
    """
    Synchronise joints + 4 camera timestamp arrays (left wrist, right wrist, bird, front) and write a CSV.

    This is backward-compatible with the existing 3-cam pipeline: use `synchronize()` unless
    you are explicitly recording a 4th camera view and want it in the sync CSV.
    """
    assert np.all(np.diff(joint_ts) > 0), "joint_data not sorted"
    assert np.all(np.diff(realsense_ts_l) > 0), "left_data not sorted"
    assert np.all(np.diff(realsense_ts_r) > 0), "right_data not sorted"
    assert np.all(np.diff(bird_ts) > 0), "bird_data not sorted"
    assert np.all(np.diff(front_ts) > 0), "front_data not sorted"

    rows = []
    i = j = k = m = f = master = 0
    A = len(joint_ts)
    B = len(realsense_ts_l)
    C = len(bird_ts)
    D = len(realsense_ts_r)
    E = len(front_ts)

    while i < A and j < B and k < C and m < D and f < E:
        pivot = max(joint_ts[i], realsense_ts_l[j], bird_ts[k], realsense_ts_r[m], front_ts[f])

        while i < A and pivot - joint_ts[i] > max_skew_s:
            i += 1
        while j < B and pivot - realsense_ts_l[j] > max_skew_s:
            j += 1
        while k < C and pivot - bird_ts[k] > max_skew_s:
            k += 1
        while m < D and pivot - realsense_ts_r[m] > max_skew_s:
            m += 1
        while f < E and pivot - front_ts[f] > max_skew_s:
            f += 1
        if i == A or j == B or k == C or m == D or f == E:
            break

        jt, lt, bt, rt, ft = (
            joint_ts[i],
            realsense_ts_l[j],
            bird_ts[k],
            realsense_ts_r[m],
            front_ts[f],
        )
        t_min = min(jt, lt, bt, rt, ft)
        t_max = max(jt, lt, bt, rt, ft)

        if t_max - t_min <= max_skew_s:
            rows.append((master, i, j, k, m, f, jt, lt, bt, rt, ft, t_max - t_min))
            master += 1
            i += 1
            j += 1
            k += 1
            m += 1
            f += 1
        else:
            earliest = int(np.argmin([jt, lt, bt, rt, ft]))
            if earliest == 0:
                i += 1
            elif earliest == 1:
                j += 1
            elif earliest == 2:
                k += 1
            elif earliest == 3:
                m += 1
            else:
                f += 1

    all_cols = [
        "master_index",
        "joint_index",
        "left_index",
        "bird_index",
        "right_index",
        "front_index",
        "joint_time",
        "realsense_time",
        "bird_time",
        "realsense_r_time",
        "front_time",
        "time_diff",
    ]
    df = pd.DataFrame(rows, columns=all_cols)
    if not debug:
        df = df[
            [
                "master_index",
                "joint_index",
                "left_index",
                "bird_index",
                "right_index",
                "front_index",
            ]
        ]
    df.to_csv(out_csv, index=False)
    print(f"Synced {len(df)} quintuplets → {out_csv} (debug={debug})")

if __name__ == "__main__":
    # call with debug=True if you want the extra columns
    pass
