# Gaze-ALOHA (cleaned)

This repo is intentionally minimal and contains only:
- **data recording** scripts (RealSense + bird cam + joint states)
- **timestamp synchronization** utilities
- **one baseline ALOHA/ACT training + inference** implementation

## Repo layout
- `recording/`: recording scripts
  - `realsense_double_record.py`: left/right RealSense RGB recordings + per-frame timestamps
  - `bird_record.py`: bird camera recording + per-frame timestamps
  - `store_joint.py`: ROS joint-state recorder (HDF5 + timestamps)
  - `record_all.sh`: convenience launcher for all three recorders
- `ALOHA-mimic/`: baseline ACT training + inference
  - `training_single.py`: baseline training
  - `inference.py`: baseline ROS inference loop
  - `data_synchronization.py`: sync utilities (writes a CSV mapping aligned indices)

## Quickstart (recording)
From the repo root:

```bash
bash recording/record_all.sh
```

Stop any recorder window with `q`, or stop everything with `Ctrl+C`.

## Notes
- The scripts write data to local folders like `aloha-data/` and `bird-data/` (ignored by git).
- Inference requires a running ROS environment and the correct joint-state topics for your setup.
