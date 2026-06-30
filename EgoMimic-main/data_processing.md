# Using your own data
## SAM Installation
Processing hand and robot data relies on SAM to generate masks for the hand and robot.

![SAM](./assets/SAM_masking.png)

Full SAM [instructions](https://github.com/facebookresearch/segment-anything-2).  

TLDR:
```
cd outsideOfEgomimic
git clone https://github.com/facebookresearch/sam2.git && cd sam2
pip install -e .
cd checkpoints && \
./download_ckpts.sh && \
mv sam2_hiera_tiny.pt /path/to/egomimic/resources/sam2_hiera_tiny.pt
```

## Processing Robot Data for Training
To use your own robot, first follow setup instructions in our hardware repo [Eve](https://github.com/SimarKareer/eve).

**Calibrate Cameras**

To train EgoMimic on your own data you must provide the hand-eye-calibration extrinsics matrix inside [``egomimic/utils/egomimicUtils``](./egomimic/utils/egomimicUtils.py)
- Print a large april tag and tape it to the wrist camera mount
- Collect calibration data for each arm one at a time.  Move the arm in many directions for best results.  This will generate an hdf5 under the `data` directory
```
cd eve
python scripts/record_episides.py --task_name CALIBRATE --arm <left or right>

cd egomimic/scripts/calibrate_camera
python calibrate_egoplay.py --h5py-path <path to hdf5 from previous section.hdf5>
```
Paste this matrix into [``egomimic/utils/egomimicUtils``](./egomimic/utils/egomimicUtils.py) for the appropriate arm

**Recording Demos**

```
cd eve
setup_eve
ros_eve
sh scripts/auto_record.sh <task defined in constants.py> <num_demos> <arm: left, right, bimanual> <starting_index>
```
This creates a folder with many demos in it
```
├── TASK_NAME
│   ├── episode_1.hdf5
...
│   ├── episode_n.hdf5
```

**Process Robot Demos**

To process the demos we've recorded we run.  Here's an example command
```
cd egomimic/scripts/aloha_process
python aloha_to_robomimic.py --dataset /path/to/TASK_NAME --arm <left, right, bimanual> --out /desired/output/path.hdf5 --extrinsics <keyName in egoMimicUtils.py>
```

## Process Human Data for Training
Collect Aria demonstrations via the Aria App, then transfer them to your computer, make the following structure
```
TASK_NAME_ARIA/
├── rawAria
│   ├── demo1.vrs
│   ├── demo1.vrs.json
...
│   ├── demoN.vrs
│   ├── demoN.vrs.json
└── converted (empty folder)
```

This will process your aria data into hdf5 format, and optionally with the `--mask` argument, it will also use SAM to mask out the hand data.
```
cd egomimic
python scripts/aria_process/aria_to_robomimic.py --dataset /path/to/TASK_NAME_ARIA --out /path/to/converted/TASK_NAME.hdf5 --hand <left, right, or bimanual> --mask
```
