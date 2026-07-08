import argparse
import os
import time

import cv2
import matplotlib.pyplot as plt
import numpy as np
from IPython.display import display, clear_output
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, random_split

from core import build, kl_divergence
from dataloader_3cam import *
from data_synchronization import arm_data_to_npy, synchronize, hdf5_to_csv
from torch.cuda.amp import autocast, GradScaler


# ===========================================================================
# LOAD PARSER ARGUMENTS
# ===========================================================================
parser = argparse.ArgumentParser(description="ALOHA single-arm training script")

parser.add_argument('-e', '--epochs', type=int, default=500, help='number of epochs to train model for')
parser.add_argument('-b', '--batch', type=int, default=6, help='number of samples per batch size')
parser.add_argument('-n', '--normalization_path', type=str, default="normalization_stats_r.npz", help='save path for normalization statistics, defaults to "normalization_stats_r.npz"')
parser.add_argument('-q', '--num_queries', type=int, default=100, help='number of queries to sample from future, defaults to 100')
parser.add_argument('-s', '--save_steps', type=int, default=500, help='number of steps between each save, defaults to 500')
parser.add_argument('-t', '--training_loss', type=float, default=1e-4, help='number of epochs to train model for, defaults to 1e-4')
parser.add_argument('-g', '--gpu_number', type=int, default=0, help='GPU number to run on (URIL uses GPU 0-7), defaults to 0')
parser.add_argument('--data_root', type=str, default=".", help="Root folder containing recorded data (defaults to repo root).")
parser.add_argument('--sync_dir', type=str, default="m-synced-csvs", help="Where to write synchronized index CSVs.")
args = parser.parse_args()

# ===========================================================================
# LOAD DATA USING DATALOADER
# ===========================================================================
data_root = os.path.abspath(args.data_root)
bird_vids = os.path.join(data_root, 'bird-realsense-data/mp4')
bird_time = os.path.join(data_root, 'bird-realsense-data/npy')
left_arm_vids = os.path.join(data_root, 'aloha-data/left/mp4')
left_arm_time = os.path.join(data_root, 'aloha-data/left/npy')
right_arm_vids = os.path.join(data_root, 'aloha-data/right/mp4')
right_arm_time = os.path.join(data_root, 'aloha-data/right/npy')
joint_pos = os.path.join(data_root, 'joint-data/right/position')
joint_time = os.path.join(data_root, 'joint-data/right/time')
synced_csvs = os.path.abspath(args.sync_dir)

bird_time_files = sorted_files_in(bird_time, '.npy')
left_arm_time_files = sorted_files_in(left_arm_time, '.npy')
right_arm_time_files = sorted_files_in(right_arm_time, '.npy')
joint_time_files = sorted_files_in(joint_time, '.npy')

# Match demos by shared id (from bird-realsense timestamp filename).
joint_time_by_id = {
    demo_id_from_joint_npy(p, "joint_timestamp_"): p for p in joint_time_files
}
left_arm_time_by_id = {demo_id_from_hash_filename(p): p for p in left_arm_time_files}
right_arm_time_by_id = {demo_id_from_hash_filename(p): p for p in right_arm_time_files}

# create synchronized csv files
os.makedirs(synced_csvs, exist_ok=True)

for bird_ts_path in bird_time_files:
    id_number = demo_id_from_hash_filename(bird_ts_path)
    if id_number not in joint_time_by_id:
        print(f"WARNING: skipping {id_number} — no right joint timestamps")
        continue
    if id_number not in left_arm_time_by_id or id_number not in right_arm_time_by_id:
        print(f"WARNING: skipping {id_number} — missing wrist camera timestamps")
        continue

    joint_ts = np.load(joint_time_by_id[id_number])
    left_arm_ts = np.load(left_arm_time_by_id[id_number])
    right_arm_ts = np.load(right_arm_time_by_id[id_number])
    bird_ts = np.load(bird_ts_path)
    synchronize(
        joint_ts,
        left_arm_ts,
        right_arm_ts,
        bird_ts,
        os.path.join(synced_csvs, f"{id_number}.csv"),
        max_skew_s=0.02,
        debug=False,
    )

device = torch.device(f"cuda:{args.gpu_number}" if torch.cuda.is_available() else "cpu")
if torch.cuda.is_available():
    n_gpus = torch.cuda.device_count()
    print(f"Detected {n_gpus} GPU(s):")
    for i in range(n_gpus):
        print(f"  [{i}] {torch.cuda.get_device_name(i)}")
else:
    print("No GPUs detected, falling back to CPU.")

print('clearing cache')
torch.cuda.empty_cache()
torch.cuda.ipc_collect()

print(f"Using device: {device}")

BATCH_SIZE = args.batch
Train_Epoch = args.epochs
K = args.num_queries  # number of frames to sample from future

dataset = PreloadedMultiVideoJointDatasetNEW(
    bird_vids,
    left_arm_vids,
    right_arm_vids,
    joint_pos,
    synced_csvs,
    num_queries=args.num_queries,
)

qpos_mean = dataset.joint_mean
qpos_std = dataset.joint_std

save_path = "normalization_stats_direct.npz"
np.savez(save_path, qpos_mean=qpos_mean, qpos_std=qpos_std)
print(f"Normalization statistics saved to {save_path}")

# total number of samples
n = len(dataset)
if n == 0:
    raise RuntimeError("Dataset is empty — check sync CSVs have rows and data files exist.")
if Train_Epoch <= 0:
    raise RuntimeError(f"--epochs must be > 0 (got {Train_Epoch}).")

# compute lengths — always keep at least one training sample when n >= 1
train_len = max(1, int(0.95 * n)) if n > 1 else n
test_len = n - train_len
if test_len == 0 and n > 1:
    train_len = n - 1
    test_len = 1
print(f"Dataset: {n} samples ({train_len} train, {test_len} val)")
train_dataset, test_dataset = random_split(
    dataset,
    [train_len, test_len],
    # generator=torch.Generator().manual_seed(42) # May need random for reproducing
)
train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=4, pin_memory=True, persistent_workers=True)
val_loader  = DataLoader(test_dataset,  batch_size=BATCH_SIZE, shuffle=True, num_workers=4, pin_memory=True, persistent_workers=True)

# sanity check
batch_sanity_check(train_loader)
class Args:
    # It looks like the code snippet is attempting to define a Python class constructor method using
    # the `__init__` method. However, the code is incomplete and contains syntax errors. The correct
    # syntax for defining a class constructor in Python is as follows:
    def __init__(self):
        self.num_queries = dataset.num_queries
        self.camera_names = ["cam0", "cam1", "cam2"]  # Assuming 3 cameras
        self.hidden_dim = 512 # 512 originally
        self.dropout = 0.1
        self.nheads = 8 # 8 originally
        self.dim_feedforward = 3200 # 3200 originally
        self.enc_layers = 4 # 4 originally
        self.dec_layers = 7 # 7 originally
        self.pre_norm = False

        # Backbone/DETR args
        self.position_embedding = 'sine'
        self.backbone = 'resnet18'
        self.lr_backbone = 1e-5
        self.masks = False
        self.dilation = False

        # Custom for your use
        self.state_dim = 7

args = Args()
state_dim = 7     # Your actual joint dimension

scaler = GradScaler() # you may see a depcreated warning, ignore the warning, this is fine

model = build(args)
model.to(device)
model.train()


optimizer   = optim.AdamW(model.parameters(), lr=1e-5, weight_decay=0.01)
criterion   = nn.L1Loss()

# ─── Set up interactive plotting ─────────────────────────────────────────────
fig, ax = plt.subplots()
# Initial plot setup (labels and title will be redrawn in the loop)
ax.set_xlabel("Epoch")
ax.set_ylabel("Loss")
ax.set_title("Training Loss")
loss_history, val_loss_history = [], []
batch_counter = 0

try:
    # ─── Training loop with live plot ────────────────────────────────────────────
    for epoch in range(Train_Epoch):
        # ---------------- TRAIN ----------------
        model.train()
        running_loss = 0.0
        train_batches = 0

        loop = tqdm(train_loader, desc=f"Epoch {epoch+1}/{Train_Epoch}", unit="batch")
        for batch_idx, data in enumerate(loop):
            qpos    = data[0].to(device)
            image   = data[1].to(device)
            actions = data[2].to(device)
            is_pad  = data[3].to(device)

            optimizer.zero_grad(set_to_none=True)

            with autocast():
                bs, seq_len, _ = actions.shape

                output, _, (mu, logvar) = model(qpos, image, env_state=None, actions=actions, is_pad=is_pad)
                total_kld, *_ = kl_divergence(mu, logvar)
                # mask out padded timesteps in L1
                all_l1 = F.l1_loss(output[..., :7], actions, reduction='none')          # [bs,seq,7]
                mask   = (~is_pad).unsqueeze(-1)                                         # [bs,seq,1]
                rec_loss = (all_l1 * mask).sum() / mask.sum()                            # average over valid frames

                loss = rec_loss + total_kld[0] * 10

            scaler.scale(loss).backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(optimizer)
            scaler.update()

            running_loss += rec_loss.item()
            train_batches += 1
            loop.set_postfix(avg_loss=running_loss / train_batches)

        if train_batches == 0:
            raise RuntimeError(
                "Training loop completed zero batches — check batch size vs dataset size."
            )
        avg_train = running_loss / train_batches
        loss_history.append(avg_train)

        # ---------------- VALIDATION ----------------
        model.eval()
        val_running = 0.0
        val_batches = 0
        with torch.no_grad():
            for qpos, image, actions, is_pad in val_loader:
                qpos = qpos.to(device)
                actions = actions.to(device)
                image = image.to(device)
                is_pad = is_pad.to(device)

                with autocast():    # ▲ silence deprecation warning
                    pred, _, (mu, logvar) = model(qpos, image, None)
                    # mask out padded timesteps in validation L1
                    all_l1_val = F.l1_loss(pred[..., :7], actions, reduction='none')    # [bs,seq,7]
                    mask_val   = (~is_pad).unsqueeze(-1)                                 # [bs,seq,1]
                    rec_val    = (all_l1_val * mask_val).sum() / mask_val.sum()

                    # The validation loss is purely the reconstruction loss
                    vloss = rec_val

                val_running += vloss.item()
                val_batches += 1

        avg_val = val_running / val_batches if val_batches else float("nan")
        val_loss_history.append(avg_val)

        print(f"→ Epoch {epoch+1} train: {avg_train:.4f} │ val: {avg_val:.4f}")

        # save every 10 epochs, regardless of val score
        if epoch % 3000 == 0 and epoch > 9000:
            torch.save(model.state_dict(), f"weights/Vinilla_ACT_{epoch}.pth")

except Exception as exc:
    print(f"Training stopped early: {type(exc).__name__}: {exc}", flush=True)
    raise

finally:
    # save model
    os.makedirs("weights", exist_ok=True)
    try:
        torch.save(model.state_dict(), "weights/Vinilla_ACT.pth")
        print('Model saved successfully')
    except Exception as e:
        print("Model save failed:", e)

    # 2) now plot & save chart
    fig, ax = plt.subplots(figsize=(10,4))
    ax.plot(loss_history,     label="train")
    #ax.plot(val_loss_history, label="val")
    ax.set_ylim(0, 1.0) # only display losses that are below 0.8
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Loss")
    ax.set_title("Final Loss Curves")
    ax.legend()
    fig.tight_layout()
    if loss_history:
        tail = loss_history[-200:] if len(loss_history) >= 200 else loss_history
        print("loss history (last", len(tail), "epochs):", tail)
    else:
        print("loss history: (empty — training may have exited before any epoch completed)")
    try:
        fig.savefig("final_loss_curve_lr_813.png")
        print('Chart saved successfully')
    except Exception as e:
        print("Chart save failed:", e)
    plt.close(fig)
